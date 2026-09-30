import networkx as nx

from relicl.rewrite.utils import (
    count_on_many_side,
    eliminate_unreachable_tables,
    remove_edge_by_pulling_fk_rows,
)
from relicl.schema import RDLTask
from relicl.timing import timed
from relicl.typing import SaveStepFn, TimingEvent


@timed(event_type=TimingEvent.rewrite)
def rewrite_pull_fk_rows(rdl_task: RDLTask, save_step: SaveStepFn) -> None:
    """Repeatedly eliminates tables that have no outgoing foreign keys"""

    # eliminate nodes with only one-side (no fks)
    while action := _eliminate_a_table_without_fks(rdl_task):
        save_step(f"pull_fk_rows: {action}")


def _eliminate_a_table_without_fks(rdl_task: RDLTask) -> str | None:
    schema: nx.MultiDiGraph = rdl_task.db.schema

    for s in schema.nodes:
        # does s have any fks?
        if count_on_many_side(schema, s) == 0 and not (
            schema.has_edge(rdl_task.task_table_name, s)
            or schema.has_edge(s, rdl_task.task_table_name)
        ):
            # no -> drop it
            for u, _, k, d in list(schema.in_edges(s, keys=True, data=True)):
                remove_edge_by_pulling_fk_rows(rdl_task, u, s, k, d["ref"], True)
            for _, v, k, d in list(schema.out_edges(s, keys=True, data=True)):
                remove_edge_by_pulling_fk_rows(rdl_task, s, v, k, d["ref"], False)

            eliminate_unreachable_tables(rdl_task)
            return f"Distributed {s} into neighbors"

    return None
