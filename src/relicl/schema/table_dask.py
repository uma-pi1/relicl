from typing import Any

import dask.dataframe as dd
import numpy as np
import pandas as pd
from ordered_set import OrderedSet

import relicl.schema.table_common as c
from relicl.typing import T_JOIN_HOW, T_SHAPE, BackendType

from .table import Table

########################################################################################
# DaskTable. ###########################################################################
########################################################################################


class DaskTable(Table):
    def __init__(
        self,
        name: str,
        pkey_col_names: OrderedSet[str],
        time_col_names: OrderedSet[str],
        children: list[Table] | None = None,
    ) -> None:
        if children is not None:
            assert all([isinstance(t, DaskTable) for t in children]), (
                "children of DaskTables must be DaskTables"
            )

        super().__init__(name, pkey_col_names, time_col_names, children)

        self._cached_df_lazy: dd.DataFrame | None = None
        self._cached_signature_lazy: set[int] = set()

    @property
    def backend(self) -> BackendType:
        return BackendType.dask

    def _df(self) -> pd.DataFrame:
        return self.df_lazy().compute()

    def is_cached_lazy(self) -> bool:
        """Checks whether this table has a cached and valid data frame."""
        if self._cached_df_lazy is None:
            return False
        return self.signature() == self._cached_signature_lazy

    def df_lazy(self) -> dd.DataFrame:
        """Lazily compute this table as a Dask Dataframe."""
        if not self.is_cached_lazy():
            self._cached_df_lazy = self._df_lazy()
            self._cached_signature_lazy = self.signature()

        assert self._cached_df_lazy is not None
        return self._cached_df_lazy

    def _op(self, *args, **kwargs) -> dd.DataFrame:
        raise NotImplementedError("_op not implemented")

    def _df_lazy(self) -> dd.DataFrame:
        # This is the default implementation. If used, the _op method needs to be
        # provided or mixed-in from a class in table_common.

        # Assert that the list comprehension below works. Then, ignore the warning
        # from the type checker (because it cannot pick up this assertion).
        assert all([isinstance(child, DaskTable) for child in self.children]), (
            "children need to be DaskTables as well"
        )

        return self._op(*[child.df_lazy() for child in self.children])  # type: ignore[attr-defined]

    @property
    def columns(self) -> list[str]:
        return self.df_lazy().columns.tolist()

    def join(
        self,
        other: "Table",
        left_on: list[str],
        right_on: list[str],
        how: T_JOIN_HOW,
        drop_left_keys=False,
        drop_right_keys=False,
    ) -> "DaskTable":
        assert isinstance(other, DaskTable), "other must be DaskTable"

        return JoinTable(
            self,
            other,
            left_on=left_on,
            right_on=right_on,
            how=how,
            drop_left_keys=drop_left_keys,
            drop_right_keys=drop_right_keys,
        )

    def query(self, query: str, local_dict: dict[str, Any]) -> "DaskTable":
        return QueryTable(self, query, local_dict)

    def filter_by_time(
        self, cutoff_time: pd.Timestamp | None, *, cutoff_op: str = "<=", keep_na=True
    ) -> "DaskTable":
        return CutoffTable(self, cutoff_time, cutoff_op=cutoff_op, keep_na=keep_na)

    def __getitem__(self, indices: np.ndarray | slice) -> "DaskTable":
        return RowSliceTable(self, indices)

    def __len__(self) -> int:
        return len(self.df)

    def rename(self, new_name: str) -> "DaskTable":
        return RenameTable(new_name, self)

    def rename_columns(self, col_names_map: dict[str, str]) -> "DaskTable":
        return RenameColumnsTable(self, col_names_map)

    def shape(self, eager: bool = True) -> T_SHAPE:
        if self.is_cached():
            assert self._cached_df is not None
            return self._cached_df.shape
        else:
            shape = self.df_lazy().shape
            if eager:
                # this may be slow
                shape = (shape[0].compute(), shape[1])
            else:
                shape = (None, shape[1])
            return shape


########################################################################################
# DaskBaseTable. #######################################################################
########################################################################################


class DaskBaseTable(DaskTable):
    """Stores a fixed input data frame."""

    def __init__(
        self,
        name: str,
        pkey_col_names: OrderedSet[str],
        time_col_names: OrderedSet[str],
        df: pd.DataFrame,
    ):
        super().__init__(name, pkey_col_names, time_col_names)
        self._cached_df: pd.DataFrame = df
        self._cached_signature = self.signature()

    def _df_lazy(self) -> dd.DataFrame:
        assert self._cached_df is not None
        return dd.from_pandas(self._cached_df)


########################################################################################
# JoinTable. ###########################################################################
########################################################################################


class JoinTable(c.JoinTable, DaskTable): ...


########################################################################################
# QueryTable. ##########################################################################
########################################################################################


class QueryTable(c.QueryTable, DaskTable): ...


########################################################################################
# CutoffTable. #########################################################################
########################################################################################


class CutoffTable(c.CutoffTable, DaskTable):
    def shape(self, eager=True):
        if len(self.time_col_names) == 0 or self.cutoff_time is None:
            # no cutoff; shape does not change
            return self.children[0].shape(eager)
        else:
            return DaskTable.shape(self, eager)


########################################################################################
# RowSliceTable. #######################################################################
########################################################################################


class RowSliceTable(DaskTable):
    def __init__(self, child, indices: np.ndarray | slice):
        super().__init__(
            child.name, child.pkey_col_names, child.time_col_names, [child]
        )
        self.indices = indices

    @property
    def df(self) -> pd.DataFrame:
        return self.children[0].df.iloc[self.indices]

    def _df_lazy(self) -> dd.DataFrame:
        """
        Dask does not support arbitrary positional row slicing like pandas iloc.
        But RowSliceTable is only used for batching, i.e., we can be sure that now
        additional operators will be built upon this table. We raise an error if
        someone tries to the latter.
        """
        raise ValueError("df_lazy not allowed on indexed table")

    def _query_graph_label(self):
        return "slice"


########################################################################################
# RenameTable. #########################################################################
########################################################################################


class RenameTable(DaskTable):
    def __init__(self, name, child):
        super().__init__(name, child.pkey_col_names, child.time_col_names, [child])

    def is_cached(self) -> bool:
        return self.children[0].is_cached()

    def is_cached_lazy(self) -> bool:
        return self.children[0].is_cached_lazy()

    @property
    def df(self) -> pd.DataFrame:
        return self.children[0].df

    def df_lazy(self) -> dd.DataFrame:
        assert isinstance(self.children[0], DaskTable), "child must be DaskTable"
        return self.children[0].df_lazy()

    def _df_lazy(self) -> dd.DataFrame:
        assert isinstance(self.children[0], DaskTable), "child must be DaskTable"

        # (covered by the above assertion)
        # noinspection PyProtectedMember
        return self.children[0]._df_lazy()

    def _query_graph_label(self) -> str:
        return f"Rename to {self.name}"

    def view_with_new_children(self, new_children: list[Table]) -> Table:
        assert len(new_children) == 1
        return RenameTable(self.name, new_children[0])


########################################################################################
# RenameColumnsDaskTable. ###############################################################
########################################################################################


class RenameColumnsTable(c.RenameColumnsTable, DaskTable): ...


########################################################################################
# CountFeatureDaskTable. ###############################################################
########################################################################################


class CountFeatureDaskTable(c.CountFeatureTable, DaskTable): ...
