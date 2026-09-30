import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from relbench.base import BaseTask, TaskType
from relbench.base import Table as RelbenchTable
from relbench.datasets import get_dataset
from relbench.tasks import get_task

from relicl.config import TASK_TABLE_ID_COL_NAME, TASK_TABLE_NAME, RelICLConfig
from relicl.schema.database import Database
from relicl.schema.table import Table, shape_str
from relicl.schema.table_common import empty_frame_like
from relicl.timing import timed
from relicl.typing import TimingEvent

logger = logging.getLogger(__name__)


########################################################################################
# Utilities. ###########################################################################
########################################################################################


def _combine_task_dfs(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[DataFrame, list[int], list[int], list[int]]:
    # Get counts.
    n_train = train_df.shape[0]
    n_val = val_df.shape[0]
    n_test = test_df.shape[0]

    # Add IDs to train.
    train_df_copy = train_df.copy()
    train_ids = list(range(n_train))
    train_df_copy.insert(0, TASK_TABLE_ID_COL_NAME, np.array(train_ids))

    # Add IDs to val.
    val_df_copy = val_df.copy()
    val_ids = list(range(n_train, n_train + n_val))
    val_df_copy.insert(0, TASK_TABLE_ID_COL_NAME, np.array(val_ids))

    # Add IDs to test.
    test_df_copy = test_df.copy()
    test_ids = list(range(n_train + n_val, n_train + n_val + n_test))
    test_df_copy.insert(0, TASK_TABLE_ID_COL_NAME, np.array(test_ids))

    combined_df = pd.concat(
        [train_df_copy, val_df_copy, test_df_copy], ignore_index=True
    )

    return (
        combined_df,
        train_ids,
        val_ids,
        test_ids,
    )


########################################################################################
# RDLTask. #############################################################################
########################################################################################


@dataclass
class RDLTask:
    """
    A Relational Deep Learning (RDL) relbench_task, characterized by:

    - The relbench_task table must have a time column.
    - Context tables may have a time column.
    - The prediction at time t should only incorporate rows with time t0 <= t (from any table).
    """

    # Tables.
    db: Database

    # Col names.
    task_table_name: str
    task_table_time_col_name: str
    task_table_id_col_name: str
    task_table_entity_key_col_name: str
    target_col_name: str

    # IDs by split.
    train_ids: list[int]
    val_ids: list[int]
    test_ids: list[int]

    # Task Type (regression, classification, etc.).
    task_type: TaskType

    @property
    def task_table(self):
        return self.db.tables[self.task_table_name]

    def val_table(self, time: pd.Timestamp | None = None) -> Table:
        val_table = self.task_table.filter_by_values(
            self.task_table_id_col_name, self.val_ids
        )

        if time is not None:
            val_table = val_table.filter_by_value(self.task_table_time_col_name, time)

        return val_table

    def test_table(self, time: pd.Timestamp | None = None) -> Table:
        test_table = self.task_table.filter_by_values(
            self.task_table_id_col_name, self.test_ids
        )

        if time is not None:
            test_table = test_table.filter_by_value(self.task_table_time_col_name, time)

        return test_table

    @timed(event_type=TimingEvent.preprocessing)
    def cutoff(self, timestamp: pd.Timestamp | None) -> "RDLTask":
        """Returns a view of this RDLTask up to the given timestamp.

        This is done by replacing this task's database with a cutoff view of itself.

        """
        return RDLTask(
            db=self.db.cutoff(timestamp),
            task_table_name=self.task_table_name,
            task_table_time_col_name=self.task_table_time_col_name,
            task_table_id_col_name=self.task_table_id_col_name,
            task_table_entity_key_col_name=self.task_table_entity_key_col_name,
            target_col_name=self.target_col_name,
            train_ids=self.train_ids,
            val_ids=self.val_ids,
            test_ids=self.test_ids,
            task_type=self.task_type,
        )

    @classmethod
    @timed(event_type=TimingEvent.preprocessing)
    def from_relbench(cls) -> "RDLTask":
        config = RelICLConfig.instance()
        relbench_db_name: str = config.relbench.db
        relbench_task_name: str = config.relbench.task

        # Load task and DB.
        relbench_task: BaseTask = get_task(
            relbench_db_name, relbench_task_name, download=True
        )
        relbench_dataset = get_dataset(relbench_db_name)
        relbench_db = relbench_dataset.get_db(upto_test_timestamp=False)

        # Combine relbench task tables.
        train_table = relbench_task.get_table("train", mask_input_cols=False)
        val_table = relbench_task.get_table("val", mask_input_cols=False)
        test_table = relbench_task.get_table("test", mask_input_cols=False)
        task_df, train_ids, val_ids, test_ids = _combine_task_dfs(
            train_table.df
            if not config.run.dry
            else pd.DataFrame(columns=train_table.df.columns),  # empty train table
            val_table.df,
            test_table.df,
        )
        relbench_task_table = RelbenchTable(
            df=task_df,
            fkey_col_to_pkey_table=train_table.fkey_col_to_pkey_table,
            pkey_col=train_table.pkey_col,
            time_col=train_table.time_col,
        )

        # Determine entity ID column name.
        assert len(train_table.fkey_col_to_pkey_table) == 1
        entity_id_col_name = list(train_table.fkey_col_to_pkey_table.keys())[0]

        # Create RelICL database.
        relbench_table_dict = relbench_db.table_dict.copy()
        relbench_table_dict[TASK_TABLE_NAME] = relbench_task_table
        db = Database.from_relbench(relbench_table_dict)

        # Convert
        # - int64 to Int64
        # - time_cols to np.datetime64
        # in task table, as used in all other tables
        from relicl.schema.table_dask import DaskBaseTable
        from relicl.schema.table_pandas import PandasBaseTable

        assert isinstance(db.tables[TASK_TABLE_NAME], DaskBaseTable) or isinstance(
            db.tables[TASK_TABLE_NAME], PandasBaseTable
        )
        # noinspection PyProtectedMember
        df = db.tables[TASK_TABLE_NAME]._cached_df
        assert isinstance(df, pd.DataFrame)
        for col_name in df.columns.to_list():
            if df[col_name].dtype == np.int64:
                df[col_name] = df[col_name].astype(pd.Int64Dtype())
            elif col_name in db.tables[
                TASK_TABLE_NAME
            ].time_col_names and not np.issubdtype(df.dtypes[col_name], np.datetime64):
                df[col_name] = pd.to_datetime(df[col_name])

        # if dry run, drop all data
        if config.run.dry:
            logger.warning("Dry run: truncating all tables")
            for table_name, table in db.tables.items():
                if table_name == TASK_TABLE_NAME:
                    continue
                assert isinstance(table, DaskBaseTable) or isinstance(
                    table, PandasBaseTable
                )
                # noinspection PyProtectedMember
                table._cached_df = empty_frame_like(table._cached_df)

        # Add dummy cutoff tables above each base table. These dummy cutoff tables do
        # not perform time filtering, but they mark places where time filtering should
        # be performed. The actual filtering happens later using db.cutoff().
        for table_name, table in db.tables.items():
            cutoff_table = table.filter_by_time(None)
            db.tables[table_name] = cutoff_table

        # Return RDLTask.
        return cls(
            db,
            TASK_TABLE_NAME,
            task_table_time_col_name=train_table.time_col,
            task_table_id_col_name=TASK_TABLE_ID_COL_NAME,
            task_table_entity_key_col_name=entity_id_col_name,
            target_col_name=relbench_task.target_col,  # type: ignore
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            task_type=relbench_task.task_type,
        )

    @cached_property
    def val_timestamps(self) -> pd.Series:
        """Returns sorted unique validation timestamps."""
        return pd.Series(
            np.sort(
                self.task_table.df.loc[
                    self.task_table.df[self.task_table_id_col_name].isin(self.val_ids),
                    self.task_table_time_col_name,
                ].unique()
            )
        )

    @cached_property
    def test_timestamps(self) -> pd.Series:
        """Returns sorted unique test timestamps."""
        return pd.Series(
            np.sort(
                self.task_table.df.loc[
                    self.task_table.df[self.task_table_id_col_name].isin(self.test_ids),
                    self.task_table_time_col_name,
                ].unique()
            )
        )

    def train_table(self, include_val: bool = False) -> Table:
        return self.task_table.filter_by_values(
            self.task_table_id_col_name,
            self.train_ids if not include_val else self.train_ids + self.val_ids,
        )

    def description(self, eager: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = dict(
            task_table_name=self.task_table_name,
            target_col_name=self.target_col_name,
            time_col_name=self.task_table_time_col_name,
            id_col_name=self.task_table_id_col_name,
            tables={},
        )
        for table_name, table in self.db.tables.items():
            result["tables"][table_name] = shape_str(table.shape(eager=eager))
        return result
