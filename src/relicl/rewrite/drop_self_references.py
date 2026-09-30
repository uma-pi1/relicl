import logging

import networkx as nx

from relicl.schema import RDLTask, Ref
from relicl.timing import timed
from relicl.typing import SaveStepFn, TimingEvent

logger = logging.getLogger(__name__)


@timed(event_type=TimingEvent.rewrite)
def rewrite_drop_self_references(rdl_task: RDLTask, save_step: SaveStepFn) -> None:
    """Drop edges from a table to itself, e.g., rel-stack's `posts.ParentId`."""

    schema: nx.MultiDiGraph = rdl_task.db.schema

    dropped: list[str] = []
    for u, v, k, data in list(schema.edges(keys=True, data=True)):
        if u != v:
            continue

        ref: Ref = data["ref"]
        schema.remove_edge(u, v, k)
        dropped.append(f"{u}.{ref.from_col_name}->{v}.{ref.to_col_name}")

    if dropped:
        logger.warning(f"Dropped self-references: {dropped}.")
        save_step("drop_self_references")
