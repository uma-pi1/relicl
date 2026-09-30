import networkx as nx

from relicl.rewrite.utils import (
    count_on_many_side,
    count_on_one_side,
    eliminate_unreachable_tables,
    remove_edge_by_pulling_fk_rows,
)
from relicl.schema import RDLTask
from relicl.timing import timed
from relicl.typing import SaveStepFn, TimingEvent


@timed(event_type=TimingEvent.rewrite)
def rewrite_contract_edge_from_task_table(
    rdl_task: RDLTask, save_step: SaveStepFn
) -> None:
    """Contract the edge from the task table to the entity table"""

    schema: nx.MultiDiGraph = rdl_task.db.schema
    s = rdl_task.task_table_name

    # first drop the edge
    assert count_on_one_side(schema, s) == 0 and count_on_many_side(schema, s) == 1
    # there is just one edge in the loops below
    for u, _, k, d in list(schema.in_edges(s, keys=True, data=True)):
        remove_edge_by_pulling_fk_rows(
            rdl_task, u, s, k, d["ref"], False, rewire_edges=True
        )
    for _, v, k, d in list(schema.out_edges(s, keys=True, data=True)):
        remove_edge_by_pulling_fk_rows(
            rdl_task, s, v, k, d["ref"], True, rewire_edges=True
        )

    # end then the entity table (which is now unreachable due to rewiring)
    eliminate_unreachable_tables(rdl_task)

    save_step("contract_edge_from_task_table")
