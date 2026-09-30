import logging

import networkx as nx

from relicl.config import RelICLConfig
from relicl.rewrite.utils import (
    count_on_many_side,
    count_on_one_side,
    eliminate_unreachable_tables,
    remove_edge_by_pulling_fk_rows,
)
from relicl.schema import RDLTask
from relicl.schema.database import Ref, denormalized_column_name, normalized_column_name
from relicl.timing import timed
from relicl.typing import SaveStepFn, TimingEvent

logger = logging.getLogger(__name__)


@timed(event_type=TimingEvent.rewrite)
def rewrite_eliminate_junction_tables(rdl_task: RDLTask, save_step: SaveStepFn) -> None:
    """
    Repeatedly eliminates tables that have no incoming but multiple outgoing foreign
    keys.
    """

    while action := _eliminate_a_junction_table(rdl_task, save_step):
        save_step(f"eliminate_junction_tables: {action}")
        eliminate_unreachable_tables(rdl_task, save_step)


def _eliminate_a_junction_table(rdl_task: RDLTask, save_step: SaveStepFn) -> str | None:
    config = RelICLConfig.instance()
    assert config.rewrite.eliminate_junction_tables_depth > 0
    schema: nx.MultiDiGraph = rdl_task.db.schema
    for s in schema.nodes:
        # Is "s" a junction table?
        if (
            count_on_one_side(schema, s) == 0
            and count_on_many_side(schema, s) > 1
            and not (
                schema.has_edge(rdl_task.task_table_name, s)
                or schema.has_edge(s, rdl_task.task_table_name)
            )
        ):
            logger.debug(f"Identified junction table {s}")
            return _eliminate_junction_table(
                s, rdl_task, save_step, n=config.rewrite.eliminate_junction_tables_depth
            )

    return None


def _eliminate_junction_table(
    s: str, rdl_task: RDLTask, save_step: SaveStepFn, n: int
) -> str | None:
    schema: nx.MultiDiGraph = rdl_task.db.schema
    edges = list(schema.in_edges(s, keys=True, data=True)) + list(
        schema.out_edges(s, keys=True, data=True)
    )

    # drop the junction table
    schema.remove_node(s)
    old_table = rdl_task.db.tables[s]
    del rdl_task.db.tables[s]

    if n == 0:
        return f"Eliminated junction table {s}"

    new_tables = []
    col_names_maps = []
    for i in range(len(edges)):
        # add a copy of the junction table
        new_junction_table = old_table.rename(unique_new_name(schema, old_table.name))
        new_table = new_junction_table
        col_names_map = {
            col_name: normalized_column_name(
                new_table.name,
                denormalized_column_name(old_table.name, col_name),
            )
            for col_name in new_table.columns
        }
        new_table = new_table.rename_columns(col_names_map)
        rdl_task.db.tables[new_table.name] = new_table
        schema.add_node(new_table.name)
        logger.debug(f"Created table {new_table.name}")

        # add and join in all but the current edge
        for j, edge in enumerate(edges):
            u, v, k, d = edge
            ref: Ref = d["ref"]
            from_col_name, to_col_name = ref.from_col_name, ref.to_col_name
            assert u != v
            if u == s:
                u = new_table.name
                from_col_name = col_names_map[from_col_name]
            if v == s:
                v = new_table.name
                to_col_name = col_names_map[to_col_name]
            if i != j:
                schema.add_edge(u, v, k, **d)
                left = u == new_table.name
                new_table = remove_edge_by_pulling_fk_rows(
                    rdl_task,
                    u,
                    v,
                    k,
                    Ref(from_col_name, to_col_name, ref.type),
                    pull_t_into_s=left,
                    rewire_edges=False,
                    drop_pulled_keys=True,
                )

        # add the current edge
        u, v, k, d = edges[i]
        dd = dict(**d)
        ref = dd["ref"]
        from_col_name, to_col_name = ref.from_col_name, ref.to_col_name
        assert u != v
        if u == s:
            u = new_table.name
            from_col_name = col_names_map[from_col_name]
        if v == s:
            v = new_table.name
            to_col_name = col_names_map[to_col_name]
        dd["ref"] = Ref(from_col_name, to_col_name, ref.type)
        schema.add_edge(u, v, **dd)

        new_tables.append(new_table)
        col_names_maps.append(col_names_map)

    # now add the junction table back in
    assert len(edges) == 2  # more edges may need more thought
    rdl_task.db.add_table(old_table)
    for i in range(len(edges)):
        u, v, k, d = edges[i]
        dd = dict(**d)
        ref = d["ref"]
        from_col_name, to_col_name = ref.from_col_name, ref.to_col_name
        new_table = new_tables[i]
        col_names_map = col_names_maps[i]
        if u == old_table.name:
            v = new_table.name
            to_col_name = col_names_map[from_col_name]
        else:
            assert v == old_table.name
            u = new_table.name
            from_col_name = col_names_map[to_col_name]
        dd["ref"] = Ref(from_col_name, to_col_name, ref.type)
        schema.add_edge(u, v, **dd)
        rdl_task.db.assert_is_valid()

    save_step(f"Unrolled junction table {s}, {n - 1} times remaining")
    return _eliminate_junction_table(s, rdl_task, save_step, n - 1)


def unique_new_name(schema: nx.MultiDiGraph, table_name: str):
    num = 0
    while True:
        num = num + 1
        if not any([f"{table_name}#{num}" in s for s in schema.nodes]):
            return f"{table_name}#{num}"
