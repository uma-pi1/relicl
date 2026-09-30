from abc import ABC, abstractmethod
from typing import Any

import dask.dataframe as dd
import networkx as nx
import numpy as np
import pandas as pd
from ordered_set import OrderedSet

from relicl.typing import T_JOIN_HOW, T_SHAPE, BackendType

########################################################################################
# Versioning. ##########################################################################
########################################################################################

max_version = 0


def unique_version() -> int:
    global max_version
    max_version += 1
    return max_version


########################################################################################
# Table. ###############################################################################
########################################################################################


class Table(ABC):
    """
    Represents a table and information about that table.

    May represent an original table in the underlying database or a derived table.
    """

    def __init__(
        self,
        name: str,
        pkey_col_names: OrderedSet[str],
        time_col_names: OrderedSet[str],
        children: list["Table"] | None = None,
    ) -> None:
        # information about table
        self.name = name
        self.version = unique_version()
        self.pkey_col_names = pkey_col_names
        self.time_col_names = time_col_names
        self.children = children if children is not None else []

        # cached data frames
        self._cached_df: pd.DataFrame | None = None
        self._cached_signature: set[int] = set()
        self._cached_df_lazy: dd.DataFrame | None = None
        self._cached_signature_lazy: set[int] = set()

    @staticmethod
    def create(
        name: str,
        pkey_col_names: OrderedSet[str],
        time_col_names: OrderedSet[str],
        df: pd.DataFrame,
        backend: BackendType,
    ) -> "Table":
        match backend:
            case BackendType.dask:
                from .table_dask import DaskBaseTable

                return DaskBaseTable(name, pkey_col_names, time_col_names, df)
            case BackendType.pandas:
                from .table_pandas import PandasBaseTable

                return PandasBaseTable(name, pkey_col_names, time_col_names, df)

    @property
    @abstractmethod
    def backend(self) -> BackendType:
        pass

    def signature(self) -> set[int]:
        """Returns a unique signature of this table.

        The signature is given by a set containing the versions of this table and all of
        its children.
        """
        signature = set()
        signature.add(self.version)
        for child in self.children:
            signature.update(child.signature())
        return signature

    def is_cached(self) -> bool:
        """Checks whether this table has a cached and valid data frame."""
        if self._cached_df is None:
            return False
        return self.signature() == self._cached_signature

    def is_cached_lazy(self) -> bool:
        """Checks whether this table has a cached and valid data frame."""
        if self._cached_df_lazy is None:
            return False
        return self.signature() == self._cached_signature_lazy

    @property
    def df(self) -> pd.DataFrame:
        """Compute this table."""
        if not self.is_cached():
            self._cached_df = self._df()
            self._cached_signature = self.signature()

        assert self._cached_df is not None
        return self._cached_df

    @abstractmethod
    def _df(self) -> pd.DataFrame:
        """Force computation of this table (ignore cache)"""

    @property
    @abstractmethod
    def columns(self) -> list[str]:
        pass

    @abstractmethod
    def join(
        self,
        other: "Table",
        left_on: list[str],
        right_on: list[str],
        how: T_JOIN_HOW,
        drop_left_keys: bool = False,
        drop_right_keys: bool = False,
    ) -> "Table":
        pass

    @abstractmethod
    def query(self, query: str, local_dict: dict[str, Any]) -> "Table":
        pass

    def filter_by_values(self, col_name: str, values: list[int]) -> "Table":
        return self.query(f"`{col_name}` in @values", local_dict={"values": values})

    def filter_by_value(self, col_name: str, value: Any) -> "Table":
        return self.query(f"`{col_name}` == @value", local_dict={"value": value})

    @abstractmethod
    def filter_by_time(
        self, cutoff_time: pd.Timestamp | None, *, cutoff_op: str = "<=", keep_na=True
    ) -> "Table":
        pass

    def _update_stored_col_names(self, col_names_map: dict[str, str]) -> None:
        self.pkey_col_names = OrderedSet(
            col_names_map.get(s, s) for s in self.pkey_col_names
        )
        self.time_col_names = OrderedSet(
            col_names_map.get(s, s) for s in self.time_col_names
        )

    @abstractmethod
    def __getitem__(self, indices: np.ndarray | slice) -> "Table":
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def rename(self, new_name: str) -> "Table":
        pass

    @abstractmethod
    def rename_columns(self, col_names_map: dict[str, str]) -> "Table":
        pass

    @abstractmethod
    def shape(self, eager: bool = True) -> T_SHAPE:
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"shape={shape_str(self.shape(eager=False))}, "
            f"pkey_col_names={self.pkey_col_names}, "
            f"time_col_names={self.time_col_names}, "
            f"cols={self.columns})"
        )

    def query_graph(self, *, shapes=False) -> tuple[nx.MultiDiGraph, int]:
        """Returns the query graph and the version/id of the root node"""
        graph: nx.MultiDiGraph[int] = nx.MultiDiGraph()  # type: ignore
        self._populate_query_graph(graph, shapes=shapes)
        return graph, self.version

    def _populate_query_graph(
        self, graph: nx.MultiDiGraph, *, shapes: bool = False
    ) -> None:
        label = self._query_graph_label()
        label += f"\n{shape_str(self.shape(eager=shapes))}"
        graph.add_node(self.version, label=label)
        for i, child in enumerate(self.children):
            child._populate_query_graph(graph, shapes=shapes)
            match len(self.children):
                case 0 | 1:
                    label = ""
                case 2:
                    label = "left" if i == 0 else "right"
                case 3:
                    label = str(i + 1)
            graph.add_edge(child.version, self.version, label=label)

    def _query_graph_label(self) -> str:
        """Label describing this table's local operation for schema graph export"""
        return f"{self.name}"

    def view_with_new_children(self, new_children) -> "Table":
        """
        Return a view of this table but with all child tables replaced by the given one.
        """
        raise ValueError(
            f"_view_with_new_children operation unsupported for table {self.name} "
            f"of type {self.__class__.__name__}"
        )

    def is_cutoff_table(self) -> bool:
        return False


########################################################################################
# Utilities. ###########################################################################
########################################################################################


def shape_str(shape: T_SHAPE) -> str:
    assert len(shape) == 2
    if shape[0] is None:
        return f"[?, {shape[1]}]"
    else:
        return f"[{shape[0]}, {shape[1]}]"
