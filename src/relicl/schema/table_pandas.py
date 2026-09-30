from typing import Any

import numpy as np
import pandas as pd
from ordered_set import OrderedSet

import relicl.schema.table_common as c
from relicl.typing import T_JOIN_HOW, T_SHAPE, BackendType

from .table import Table

########################################################################################
# PandasTable. #########################################################################
########################################################################################


class PandasTable(Table):
    def __init__(
        self,
        name: str,
        pkey_col_names: OrderedSet[str],
        time_col_names: OrderedSet[str],
        children: list[Table] | None = None,
    ) -> None:
        super().__init__(name, pkey_col_names, time_col_names, children)

    @property
    def backend(self) -> BackendType:
        return BackendType.pandas

    def _op(self, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError("_op not implemented")

    def _df(self) -> pd.DataFrame:
        # This is the default implementation. If used, the _op method needs to be
        # provided or mixed-in from a class in table_common.
        return self._op(*[child.df for child in self.children])

    @property
    def columns(self) -> list[str]:
        return self.df.columns.tolist()

    def join(
        self,
        other: "Table",
        left_on: list[str],
        right_on: list[str],
        how: T_JOIN_HOW,
        drop_left_keys: bool = False,
        drop_right_keys: bool = False,
    ) -> "PandasTable":
        return JoinTable(
            self,
            other,
            left_on=left_on,
            right_on=right_on,
            how=how,
            drop_left_keys=drop_left_keys,
            drop_right_keys=drop_right_keys,
        )

    def query(self, query: str, local_dict: dict[str, Any]) -> "PandasTable":
        return QueryTable(self, query, local_dict)

    def filter_by_time(
        self, cutoff_time: pd.Timestamp | None, *, cutoff_op: str = "<=", keep_na=True
    ) -> "PandasTable":
        return CutoffTable(self, cutoff_time, cutoff_op=cutoff_op, keep_na=keep_na)

    def __getitem__(self, indices: np.ndarray | slice) -> "PandasTable":
        return RowSliceTable(self, indices)

    def __len__(self) -> int:
        return len(self.df)

    def rename(self, new_name: str) -> "PandasTable":
        return RenameTable(new_name, self)

    def rename_columns(self, col_names_map: dict[str, str]) -> "PandasTable":
        return RenameColumnsTable(self, col_names_map)

    def shape(self, eager: bool = True) -> T_SHAPE:
        return self.df.shape


########################################################################################
# PandasBaseTable. #####################################################################
########################################################################################


class PandasBaseTable(PandasTable):
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

    def _df(self) -> pd.DataFrame:
        raise ValueError("should never be called")


########################################################################################
# JoinTable. ###########################################################################
########################################################################################


class JoinTable(c.JoinTable, PandasTable): ...


########################################################################################
# QueryTable. ##########################################################################
########################################################################################


class QueryTable(c.QueryTable, PandasTable): ...


########################################################################################
# CutoffTable. #########################################################################
########################################################################################


class CutoffTable(c.CutoffTable, PandasTable): ...


########################################################################################
# RowSliceTable. #######################################################################
########################################################################################


class RowSliceTable(PandasTable):
    def __init__(self, child, indices: np.ndarray | slice):
        super().__init__(
            child.name, child.pkey_col_names, child.time_col_names, [child]
        )
        self.indices = indices

    def _df(self) -> pd.DataFrame:
        return self.children[0].df.iloc[self.indices]

    def _query_graph_label(self) -> str:
        return "slice"


class RenameTable(PandasTable):
    def __init__(self, name, child):
        super().__init__(name, child.pkey_col_names, child.time_col_names, [child])

    def is_cached(self) -> bool:
        return self.children[0].is_cached()

    @property
    def df(self) -> pd.DataFrame:
        return self.children[0].df

    def _query_graph_label(self) -> str:
        return f"Rename to {self.name}"

    def view_with_new_children(self, new_children: list[Table]) -> Table:
        assert len(new_children) == 1
        return RenameTable(self.name, new_children[0])


########################################################################################
# RenameColumnsTable. ##################################################################
########################################################################################


class RenameColumnsTable(c.RenameColumnsTable, PandasTable): ...


########################################################################################
# CountFeaturesPandasTable. ############################################################
########################################################################################


class CountFeaturePandasTable(c.CountFeatureTable, PandasTable): ...
