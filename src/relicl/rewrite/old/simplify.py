import logging

import networkx as nx

from relicl.config import RelICLConfig
from relicl.schema import RDLTask
from relicl.utils import store_schema_graph_to_disk

from ...timing import timed
from ...typing import TimingEvent
from .empirical_keys import EmpiricalKeysJoiner
from .relationships import RelationshipJoiner

logger = logging.getLogger(__name__)


@timed(event_type=TimingEvent.rewrite)
def simplify_task(rdl_task: RDLTask) -> None:
    config = RelICLConfig.instance()
    # Plot schema graph.
    if config.output.graphs:
        store_schema_graph_to_disk(
            rdl_task.db.schema,
            "phase-0.png",
            tables=rdl_task.db.tables,
            shapes=config.output.all_shapes,
            highlight_nodes=[rdl_task.task_table_name],
        )

    # Schema simplification. ###########################################################

    # # Phase I: Relationship-based joins.
    relationship_joiner = RelationshipJoiner(rdl_task.db, rdl_task.task_table_name)
    rdl_task.db, rdl_task.task_table_name = relationship_joiner.join_tables()
    if config.output.graphs:
        store_schema_graph_to_disk(
            rdl_task.db.schema,
            "phase-1.png",
            tables=rdl_task.db.tables,
            shapes=config.output.all_shapes,
            highlight_nodes=[rdl_task.task_table_name],
        )

    # # Phase II: Empirical key-based joins.
    empirical_keys_joiner = EmpiricalKeysJoiner(rdl_task.db, rdl_task.task_table_name)
    rdl_task.db, rdl_task.task_table_name = empirical_keys_joiner.join_tables()

    # Remove self-loops (they are not required for inference).
    rdl_task.db.schema.remove_edges_from(nx.selfloop_edges(rdl_task.db.schema))
    if config.output.graphs:
        store_schema_graph_to_disk(
            rdl_task.db.schema,
            "phase-2.png",
            tables=rdl_task.db.tables,
            shapes=config.output.all_shapes,
            highlight_nodes=[rdl_task.task_table_name],
        )

    # Log results.
    logger.debug("=" * 88)
    logger.debug("Resulting tables:")
    logger.debug("=" * 88)
    for table in rdl_task.db.tables.values():
        logger.debug(table)
