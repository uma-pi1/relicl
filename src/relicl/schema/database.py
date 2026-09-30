from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Mapping

import dask.dataframe as dd
import networkx as nx
import pandas as pd
from ordered_set import OrderedSet
from relbench.base import Table as RelbenchTable

from relicl.config import RelICLConfig
from relicl.schema import Table
from relicl.schema.table_dask import DaskTable

########################################################################################
# Refs. ################################################################################
########################################################################################


class RefType(StrEnum):
    many_to_one = auto()
    one_to_many = auto()
    many_to_many = auto()

    @property
    def is_from_one(self) -> bool:
        return self == RefType.one_to_many

    @property
    def is_from_many(self) -> bool:
        return self in [RefType.many_to_one, RefType.many_to_many]

    @property
    def is_to_one(self) -> bool:
        return self == RefType.many_to_one

    @property
    def is_to_many(self) -> bool:
        return self in [RefType.one_to_many, RefType.many_to_many]

    def inverse(self) -> "RefType":
        match self:
            case RefType.many_to_one:
                return RefType.one_to_many
            case RefType.one_to_many:
                return RefType.many_to_one
            case RefType.many_to_many:
                return RefType.many_to_many


@dataclass(frozen=True)
class Ref:
    """A reference between tables"""

    from_col_name: str
    to_col_name: str
    type: RefType

    def inverse(self) -> "Ref":
        return Ref(self.to_col_name, self.from_col_name, self.type.inverse())

    @property
    def is_from_one(self) -> bool:
        return self.type.is_from_one

    @property
    def is_from_many(self) -> bool:
        return self.type.is_from_many

    @property
    def is_to_one(self) -> bool:
        return self.type.is_to_one

    @property
    def is_to_many(self) -> bool:
        return self.type.is_to_many


########################################################################################
# Database. ############################################################################
########################################################################################


class Database:
    """Consists of a set of tables and a schema."""

    def __init__(
        self,
        tables: dict[str, Table] | None = None,
        schema: nx.MultiDiGraph | None = None,
    ) -> None:
        self.tables: dict[str, Table] = {} if tables is None else tables

        # noinspection PyTypeChecker
        self.schema: nx.MultiDiGraph[str] = (
            nx.MultiDiGraph() if schema is None else schema
        )

    def add_table(self, table: Table) -> None:
        self.tables[table.name] = table
        self.schema.add_node(table.name)

    def drop_table(self, name) -> None:
        del self.tables[name]
        self.schema.remove_node(name)

    def add_many_to_one(
        self,
        many_table_name: str,
        many_col_name: str,
        one_table_name: str,
        one_col_name: str,
    ) -> None:
        self.schema.add_edge(
            many_table_name,
            one_table_name,
            ref=Ref(many_col_name, one_col_name, RefType.many_to_one),
        )

    def add_many_to_many(
        self,
        table1_name: str,
        col1_name: str,
        table2_name: str,
        col2_name: str,
    ) -> None:
        self.schema.add_edge(
            table1_name,
            table2_name,
            ref=Ref(col1_name, col2_name, RefType.many_to_many),
        )

    def get_fkey_refs(self, table_name: str) -> list[Ref]:
        refs: list[Ref] = []
        for _, __, data in self.schema.out_edges(table_name, data=True):
            ref = data["ref"]
            refs.append(ref)
        return refs

    def cutoff(self, timestamp: pd.Timestamp | None) -> "Database":
        """Returns a view of this database up to the given timestamp.

        This is done by replacing the timestamp in all CutoffTables used to define the
        tables of this database. This method throws an AssertionError when there is a
        base table that does not have a cutoff table on top.

        """
        cutoff_db = Database()
        cutoff_db.schema = self.schema
        new_tables: dict[int, Table] = {}
        for table_name, table in self.tables.items():
            cutoff_db.tables[table_name] = _cutoff_view_of_table(
                table,
                None,
                cutoff_time=timestamp,
                cutoff_op="<=",
                new_tables=new_tables,
            )
        return cutoff_db

    def replace_table(self, old_table: Table, new_table: Table) -> None:
        """Replaces all occurrences of old_table by new_table.

        If old_table occurs as a schema table, replace it there. If old_table occurs as
        somewhere in the computation of a derived table, replace it there, too.

        """

        assert old_table.name == new_table.name
        for table_name, table in self.tables.items():
            if table is old_table:
                assert table_name == old_table.name
                self.tables[table_name] = new_table
            else:
                parents = [table]
                while len(parents) > 0:
                    parent = parents.pop()
                    for i, child in enumerate(parent.children):
                        if child is old_table:
                            parent.children[i] = new_table
                        else:
                            parents.append(child)

    def rename_table(self, old_table_name: str, new_table_name: str) -> None:
        if old_table_name == new_table_name:
            return

        assert new_table_name not in self.tables
        table = self.tables.pop(old_table_name)
        table.name = new_table_name
        self.tables[new_table_name] = table
        nx.relabel_nodes(self.schema, {old_table_name: new_table_name}, copy=False)

    def assert_is_valid(self) -> None:
        """
        Checks whether all tables/columns are compatible with the schema.
        """
        assert set(self.schema.nodes.keys()) == set(self.tables.keys())
        for u, v, _, data in self.schema.edges(data=True, keys=True):
            ref = data["ref"]
            assert ref.from_col_name in self.tables[u].columns, (
                f"{ref.from_col_name} not in {self.tables[u].name}{self.tables[u].columns}"
            )
            assert ref.to_col_name in self.tables[v].columns, (
                f"{ref.to_col_name} not in {self.tables[v].name}{self.tables[v].columns}"
            )

    def normalize_column_names(self) -> None:
        """Modifies this database such that all column names are unique."""

        for table_name, table in self.tables.items():
            col_names_map = {
                col_name: normalized_column_name(table_name, col_name)
                for col_name in table.columns
                if not is_normalized_column_name(table_name, col_name)
            }
            if len(col_names_map) > 0:
                self.rename_columns_of_table(table_name, col_names_map)

        self.assert_is_valid()

    def rename_columns_of_table(
        self, table_name: str, col_names_map: dict[str, str]
    ) -> None:
        self.tables[table_name] = self.tables[table_name].rename_columns(col_names_map)
        for u, v, data in self.schema.edges(data=True):
            ref: Ref = data["ref"]
            from_col_name = ref.from_col_name
            to_col_name = ref.to_col_name
            changed = False
            if u == table_name:
                from_col_name = col_names_map.get(from_col_name, from_col_name)
                changed = True
            if v == table_name:
                to_col_name = col_names_map.get(to_col_name, to_col_name)
                changed = True
            if changed:
                data["ref"] = Ref(from_col_name, to_col_name, ref.type)

    @classmethod
    def from_relbench(cls, relbench_table_dict: dict[str, RelbenchTable]) -> "Database":
        db = cls()
        for table_name, t in relbench_table_dict.items():
            _add_relbench_table_to_database(db, table_name, t, relbench_table_dict)

        db.assert_is_valid()
        return db

    def compute(self) -> None:
        """Precompute data frames for all tables of this database."""

        lazy_dfs: list[dd.DataFrame] = []
        tables: list[Table] = []
        for table in self.tables.values():
            if not table.is_cached():
                if isinstance(table, DaskTable):
                    # we collect these and compute them at once lazily
                    lazy_dfs.append(table.df_lazy())
                else:
                    # precompute right away
                    # noinspection PyStatementEffect
                    table.df

        if len(lazy_dfs) > 0:
            # compute all Dask tables at once
            dfs = dd.compute(*lazy_dfs)
            for i in range(len(tables)):
                df = dfs[i]
                table = tables[i]
                table._cached_df = df
                table._cached_signature = table.signature()


########################################################################################
# Utilities. ###########################################################################
########################################################################################


def _add_relbench_table_to_database(
    db: Database,
    table_name: str,
    relbench_table: RelbenchTable,
    relbench_table_dict: Mapping[str, RelbenchTable],
) -> None:
    """Adds a RelBench table to a RelICL database"""
    config = RelICLConfig.instance()

    # Add the table.
    assert table_name not in db.tables
    pkey_col_names: OrderedSet[str] = OrderedSet()
    if relbench_table.pkey_col is not None:
        pkey_col_names.add(relbench_table.pkey_col)

    time_col_names: OrderedSet[str] = OrderedSet()
    if relbench_table.time_col is not None:
        time_col_names.add(relbench_table.time_col)

    table = Table.create(
        table_name,
        pkey_col_names,
        time_col_names,
        relbench_table.df,
        config.backend,
    )
    db.add_table(table)

    # Add foreign key relationships.
    for fkey_col_name, pkey_table_name in relbench_table.fkey_col_to_pkey_table.items():
        pkey_col_name = relbench_table_dict[pkey_table_name].pkey_col
        assert pkey_col_name is not None
        db.add_many_to_one(
            many_table_name=table_name,
            many_col_name=fkey_col_name,
            one_table_name=pkey_table_name,
            one_col_name=pkey_col_name,
        )


def _cutoff_view_of_table(
    table: Table,
    parent: Table | None,
    *,
    cutoff_time: pd.Timestamp | None,
    cutoff_op: str,
    new_tables: dict[int, Table],
) -> Table:
    # we have seen this table already
    if id(table) in new_tables:
        return new_tables[id(table)]

    if table.is_cutoff_table():
        if len(table.time_col_names) == 0 and len(table.children[0].children) == 0:
            # we don't need a new cutoff table, as no filtering is performed
            # and the table below is a base table
            return table
        else:
            # otherwise we create a new cutoff table
            new_child = _cutoff_view_of_table(
                table.children[0],
                table,
                cutoff_time=cutoff_time,
                cutoff_op=cutoff_op,
                new_tables=new_tables,
            )
            new_table = new_child.filter_by_time(cutoff_time, cutoff_op=cutoff_op)
            new_tables[id(table)] = new_table
            return new_table
    elif table.children is None or len(table.children) == 0:
        # ensure that all base tables are correctly filtered
        assert isinstance(parent, Table)
        assert parent.is_cutoff_table()
        assert table.time_col_names == parent.time_col_names
        return table
    else:
        # traverse the children
        new_children = [
            _cutoff_view_of_table(
                child,
                table,
                cutoff_time=cutoff_time,
                cutoff_op=cutoff_op,
                new_tables=new_tables,
            )
            for child in table.children
        ]

        # see if we need to change something
        assert len(table.children) == len(new_children)
        new_table = table  # assume nothing changed
        for i in range(len(table.children)):
            if id(table.children[i]) != id(new_children[i]):
                # something changed
                new_table = table.view_with_new_children(new_children)
                break

        # all done
        new_tables[id(table)] = new_table
        return new_table


def normalized_column_name(table_name: str, col_name: str) -> str:
    """Returns a normalized columns name of the form "table__column". This method is
    idempotent."""
    if col_name.startswith(f"{table_name}__"):
        return col_name
    else:
        return f"{table_name}__{col_name}"


def is_normalized_column_name(table_name: str, col_name: str) -> bool:
    prefix = f"{table_name}__"
    return col_name.startswith(prefix)


def denormalized_column_name(table_name: str, col_name: str) -> str:
    prefix = f"{table_name}__"
    if col_name.startswith(prefix):
        return col_name[len(prefix) :]
    else:
        return col_name
