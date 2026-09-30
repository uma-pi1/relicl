from abc import ABC
from typing import Any, Literal

import duckdb
import numpy as np
import pandas as pd
from dask import dataframe as dd

from ..config import RelICLConfig
from ..timing import timed
from ..typing import T_DF, T_JOIN_HOW, T_SHAPE, TimingEvent
from .table import Table

########################################################################################
# JoinTable. ###########################################################################
########################################################################################


class JoinTable(Table, ABC):
    def __init__(
        self,
        left: Table,
        right: Table,
        left_on: list[str],
        right_on: list[str],
        how: T_JOIN_HOW,
        drop_left_keys: bool = False,
        drop_right_keys: bool = False,
    ) -> None:
        name = left.name + "*" + right.name
        pkey_col_names = left.pkey_col_names.union(right.pkey_col_names)
        time_col_names = left.time_col_names.union(right.time_col_names)
        super().__init__(name, pkey_col_names, time_col_names, [left, right])

        self.left_on = left_on
        self.right_on = right_on
        self.how = how
        self.drop_left_keys = drop_left_keys
        self.drop_right_keys = drop_right_keys

    @timed(event_type=TimingEvent.rewrite)
    def _op(self, left_df: T_DF, right_df: T_DF) -> T_DF:
        # This would be the pure Pandas implementation
        # return (
        #     self.children[0]
        #     .df_lazy()
        #     .merge(
        #         self.children[1].df_lazy(),
        #         left_on=self.left_on,
        #         right_on=self.right_on,
        #         how=self.how,
        #         suffixes=(None, None),
        #     )
        # )

        # Dask/pandas by default takes the cross product of all NA keys. We don't want
        # that, so we work around.
        left_na, left_notna = _split_frame_rows_na_notna(left_df, self.left_on)
        right_na, right_notna = _split_frame_rows_na_notna(right_df, self.right_on)

        results = [
            left_notna.merge(
                right_notna,
                left_on=self.left_on,
                right_on=self.right_on,
                how=self.how,
                suffixes=(None, None),
            )
        ]

        if self.how == "left" or self.how == "outer":
            results += [
                left_na.merge(
                    empty_frame_like(right_na),
                    left_on=self.left_on,
                    right_on=self.right_on,
                    how="left",
                    suffixes=(None, None),
                ),
            ]

        if self.how == "right" or self.how == "outer":
            results += [
                empty_frame_like(left_na).merge(
                    right_na,
                    left_on=self.left_on,
                    right_on=self.right_on,
                    how="right",
                    suffixes=(None, None),
                ),
            ]

        if len(results) == 1:
            result = results[0]
        else:
            if isinstance(left_df, pd.DataFrame):
                result = pd.concat(results)
            elif isinstance(left_df, dd.DataFrame):
                result = dd.concat(results)
            else:
                raise ValueError()

        if self.drop_left_keys:
            result = result.drop(columns=self.left_on)

        if self.drop_right_keys:
            result = result.drop(columns=self.right_on)

        return result

    def _query_graph_label(self) -> str:
        return (
            f"{self.how} join\non "
            f"{1 if isinstance(self.left_on, str) else len(self.left_on)} keys"
        )

    def view_with_new_children(self, new_children: list[Table]) -> Table:
        assert len(self.children) == len(new_children) == 2
        view = type(self)(
            left=new_children[0],
            right=new_children[1],
            left_on=self.left_on,
            right_on=self.right_on,
            how=self.how,
            drop_right_keys=self.drop_right_keys,
            drop_left_keys=self.drop_left_keys,
        )
        assert view.columns == self.columns
        return view


########################################################################################
# QueryTable. ##########################################################################
########################################################################################


class QueryTable(Table, ABC):
    def __init__(
        self, child: Table, query: str, local_dict: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            child.name, child.pkey_col_names, child.time_col_names, [child]
        )
        self.query_str = query
        self.local_dict = local_dict

    @timed(event_type=TimingEvent.rewrite)
    def _op(self, df: T_DF) -> T_DF:
        return df.query(self.query_str, local_dict=self.local_dict)

    def _query_graph_label(self) -> str:
        return f"σ({self.query_str})\n{self.local_dict}"


########################################################################################
# CutoffTable. #########################################################################
########################################################################################


class CutoffTable(Table, ABC):
    def __init__(
        self,
        child: Table,
        cutoff_time: pd.Timestamp | None,
        *,
        cutoff_op: str,
        keep_na: bool,
    ) -> None:
        super().__init__(
            child.name, child.pkey_col_names, child.time_col_names, [child]
        )
        self.cutoff_time = cutoff_time
        self.cutoff_op = cutoff_op
        self.keep_na = keep_na

    @timed(event_type=TimingEvent.preprocessing)
    def _op(self, df: T_DF) -> T_DF:
        query = ""
        if not pd.isna(self.cutoff_time):
            for time_col in self.time_col_names:
                # this assertion ensures that the NA filtering below works as intended
                # noinspection PyUnresolvedReferences
                assert np.issubdtype(df.dtypes[time_col], np.datetime64)
                query += " and " if len(query) > 0 else ""
                query += f"(`{time_col}` {self.cutoff_op} @cutoff_time"
                if self.keep_na:
                    query += f" or `{time_col}` in @nas"
                query += ")"

        result = df
        if len(query) > 0:
            result = result.query(
                query,
                local_dict={"cutoff_time": self.cutoff_time, "nas": [pd.NaT]},
            )
        return result

    def _query_graph_label(self) -> str:
        if len(self.time_col_names) == 0:
            return "cutoff\npass through (no time columns)"
        if pd.isna(self.cutoff_time):
            return "cutoff\npass through (no cutoff time)"
        return f"cutoff\n{self.cutoff_op} {self.cutoff_time}"

    def shape(self, eager=True):
        if len(self.time_col_names) == 0 or self.cutoff_time is None:
            # no cutoff; shape does not change
            return self.children[0].shape(eager)
        else:
            return super().shape(eager)

    def is_cutoff_table(self) -> bool:
        return True


########################################################################################
# CountFeatureTable. ###################################################################
########################################################################################


class CountFeatureTable(Table, ABC):
    """
    Adds a column counting, for each row of the "one" table, how many rows of the "many"
    table reference it.
    """

    def __init__(
        self, one_df: Table, many_df: Table, one_col_name: str, many_col_name: str
    ) -> None:
        super().__init__(
            one_df.name,
            one_df.pkey_col_names,
            one_df.time_col_names,
            [one_df, many_df],
        )
        self.one_col_name = one_col_name
        self.many_col_name = many_col_name

    @timed(event_type=TimingEvent.rewrite)
    def _op(self, one_df: T_DF, many_df: T_DF) -> T_DF:
        # Extract time columns.
        one_time_col_names = list(self.children[0].time_col_names)
        many_time_col_names = list(self.children[1].time_col_names)

        # Name of the new column.
        refcount_col_name = f"{self.one_col_name}__refcount__{self.many_col_name}"

        windows = RelICLConfig.instance().rewrite.count_feature_window_days or []

        # Both sides carry timestamps, so every row can be counted against its own: the
        # count only sees successor rows that precede the row it describes.
        if one_time_col_names and many_time_col_names:
            return self._add_per_row_counts(
                one_df,
                many_df,
                one_time_col_names,
                many_time_col_names,
                windows,
                refcount_col_name,
            )

        # no time filtering necessary, cheaper
        counts_table = (
            many_df.groupby(self.many_col_name)
            .size()
            .rename(refcount_col_name)
            .astype(pd.Int64Dtype())
            .to_frame()
        )
        new_one_df = one_df.merge(
            counts_table,
            left_on=self.one_col_name,
            right_on=self.many_col_name,
            how="left",
            suffixes=(None, None),
        )

        # fill N/A counts with 0 (because we know it's 0)
        new_one_df[refcount_col_name] = new_one_df[refcount_col_name].fillna(0)

        # Windows measured back from one reference for the whole table, since this table
        # has no per-row time to measure against. Cheap, no join. Windowing needs times on
        # the successor table: without them `max(axis=1)` below is an all-NaN float column
        # and the comparison raises.
        if windows and many_time_col_names:
            many_times = many_df[many_time_col_names].max(axis=1)

            # The one-side table does not have a time col, so windows cannot be based
            # upon that. Instead, use the current cutoff time. If that is not available,
            # use the max of the available times. This is a bad approximation if the
            # many-side table (or more specifically, its time col) are sparse.
            reference = _cutoff_time_below(self.children[1])
            if reference is None or pd.isna(reference):
                reference = many_times.max()

            # Compute counts per window (same approach as above).
            for window in windows:
                col_name = f"{refcount_col_name}__{window}d"
                recent = many_df[many_times > reference - pd.Timedelta(days=window)]
                counts_table = (
                    recent.groupby(self.many_col_name)
                    .size()
                    .rename(col_name)
                    .astype(pd.Int64Dtype())
                    .to_frame()
                )
                new_one_df = new_one_df.merge(
                    counts_table,
                    left_on=self.one_col_name,
                    right_on=self.many_col_name,
                    how="left",
                    suffixes=(None, None),
                )
                new_one_df[col_name] = new_one_df[col_name].fillna(0)

        return new_one_df

    def _add_per_row_counts(
        self,
        one_df: T_DF,
        many_df: T_DF,
        one_time_col_names: list[str],
        many_time_col_names: list[str],
        windows: list[int],
        refcount_col_name: str,
    ) -> T_DF:
        """Successor counts measured against each row's own timestamp."""

        # Only implemented for pandas backend for now; dask not supported.
        assert isinstance(one_df, pd.DataFrame) and isinstance(many_df, pd.DataFrame), (
            "per-row count features are implemented for the pandas backend only"
        )

        # Only the join key and the reference time matter. `pos` carries the original
        # row order.
        one_rows = pd.DataFrame(
            dict(
                pos=np.arange(len(one_df)),
                key=one_df[self.one_col_name].to_numpy(),
                t=one_df[one_time_col_names].max(axis=1).to_numpy(),
            )
        )
        many_rows = pd.DataFrame(
            dict(
                key=many_df[self.many_col_name].to_numpy(),
                t=many_df[many_time_col_names].max(axis=1).to_numpy(),
            )
        )

        # Every output column (window) is one aggregate over the same join, so a single
        # query produces all of them.
        #
        # For a task table, with `windows=[7, 28]`, this builds:
        #
        #     SELECT o.pos,
        #            count(m.t)                                      AS "..refcount..",
        #            count(m.t) FILTER (m.t > o.t - INTERVAL 7 DAY)  AS "..refcount..__7d",
        #            count(m.t) FILTER (m.t > o.t - INTERVAL 28 DAY) AS "..refcount..__28d"
        #     FROM one_rows o LEFT JOIN many_rows m
        #       ON o.key = m.key AND m.t <= o.t
        #     GROUP BY o.pos

        # aggregates: col_name -> aggregate expression
        aggregates = {refcount_col_name: "count(m.t)"}  # no window, always included
        for window in (int(w) for w in windows):
            aggregates[f"{refcount_col_name}__{window}d"] = (
                f"count(m.t) FILTER (m.t > o.t - INTERVAL {window} DAY)"  # window
            )

        # Parse aggregates to SQL string.
        select_list = ", ".join(
            f'{expr} AS "{col}"' for col, expr in aggregates.items()
        )

        # Build query.
        query = f"""
            SELECT o.pos, {select_list}
            FROM one_rows o LEFT JOIN many_rows m
              ON o.key = m.key AND m.t <= o.t
            GROUP BY o.pos
        """

        # Execute.
        with duckdb.connect() as con:
            con.register("one_rows", one_rows)
            con.register("many_rows", many_rows)
            counts = con.execute(query).df().set_index("pos")

        # GROUP BY may return rows in arbitrary order, hence the reindex by position.
        counts = counts.reindex(np.arange(len(one_df)))

        # Add the new columns to the original table.
        result = one_df.copy()
        for col in aggregates:
            result[col] = counts[col].to_numpy()

        # Convert to same data types as the other branch.
        return result.astype({col: pd.Int64Dtype() for col in aggregates})

    def view_with_new_children(self, new_children: list[Table]) -> Table:
        assert len(self.children) == len(new_children)
        view = type(self)(
            new_children[0],
            new_children[1],
            self.one_col_name,
            self.many_col_name,
        )
        assert view.columns == self.columns
        return view

    def _query_graph_label(self) -> str:
        return f"{self.name}:\nadd count of {self.many_col_name}"

    def shape(self, eager: bool = True) -> T_SHAPE:
        shape = self.children[0].shape(eager)
        shape = (shape[0], shape[1] + 1)
        return shape


class RenameColumnsTable(Table, ABC):
    def __init__(self, child: Table, col_names_map: dict[str, str]) -> None:
        super().__init__(
            child.name,
            child.pkey_col_names,
            child.time_col_names,
            [child],
        )
        self._update_stored_col_names(col_names_map)
        self.col_names_map = col_names_map

    @timed(event_type=TimingEvent.rewrite)
    def _op(self, df: T_DF) -> T_DF:
        return df.rename(columns=self.col_names_map)

    def view_with_new_children(self, new_children: list[Table]) -> Table:
        assert len(self.children) == len(new_children) == 1
        view = type(self)(new_children[0], self.col_names_map)
        return view

    def _query_graph_label(self) -> str:
        return "rename columns"


########################################################################################
# Utilities. ###########################################################################
########################################################################################


def _cutoff_time_below(table: Table) -> pd.Timestamp | None:
    """The evaluation cutoff of the nearest `CutoffTable` below `table`, if any."""

    # `from_relbench` puts a CutoffTable above every base table and `db.cutoff` stamps the
    # evaluation timestamp into it, so the cutoff is reachable by descending. Only the
    # first child is followed: for the nodes that appear under a count feature that is the
    # lineage of the table itself. Returns None when no cutoff table is found, and the
    # cutoff is None on the uncut task, so callers need a fallback either way.
    while True:
        if table.is_cutoff_table():
            assert isinstance(table, CutoffTable)
            return table.cutoff_time
        if not table.children:
            return None
        table = table.children[0]


def _split_frame_rows_na_notna(
    df: T_DF,
    col_names: str | list[str],
    combine: Literal["and", "or"] = "and",
) -> tuple[T_DF, T_DF]:
    """
    Splits a dask dataframe into NA-rows (first result) and not-NA-rows (second
    result).

    If "combine" is "and" ("or"), NA-rows are rows in which all (any) specified columns
    are NA.
    """
    if isinstance(col_names, str):
        col_names = [col_names]

    if combine == "and":
        isna = df[col_names].isna().all(axis=1)
    elif combine == "or":
        isna = df[col_names].isna().any(axis=1)
    else:
        raise ValueError(f"unknown combine: {combine}")

    return df[isna], df[~isna]


def empty_frame_like(df: T_DF) -> T_DF:
    """
    Return an empty Dataframe with the same columns and column types as the
    given dataframe.
    """
    # noinspection PyTypeChecker,PyUnresolvedReferences
    empty_pd = pd.DataFrame({col: pd.Series(dtype=df[col].dtype) for col in df.columns})

    if isinstance(df, dd.DataFrame):
        return dd.from_pandas(empty_pd, npartitions=df.npartitions)

    return empty_pd
