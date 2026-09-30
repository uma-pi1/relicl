import logging
from abc import ABC, abstractmethod

import networkx as nx

from relicl.config import TASK_TABLE_NAME
from relicl.schema import Database, Table
from relicl.typing import T_JOIN_HOW

logger = logging.getLogger(__name__)


########################################################################################
# Class: Joiner ########################################################################
########################################################################################


class Joiner(ABC):
    _JOIN_HOW: T_JOIN_HOW = "outer"

    def __init__(self, db: Database, starting_node: str | None = None) -> None:
        self._db = db
        self._join_graph = self._build_join_graph()
        self._starting_node = starting_node

    # Abstract methods. ################################################################

    @abstractmethod
    def _build_join_graph(self) -> nx.DiGraph:
        """
        The join graph contains information on which tables/nodes can be joined. The
        graph's edges need to contain the attributes `left_on` and `right_on`.
        """

    # Public methods. ##################################################################

    def join_tables(self) -> tuple[Database, str]:
        # Initialize return variables.
        simplified_tables: dict[str, Table] = {}
        simplified_schema: nx.MultiDiGraph[str] = self._db.schema.copy()

        i = 0  # Required to include starting node.
        while True:
            # Find the current longest path (relationship joiner: start with
            # `task_table`).
            longest_path = (
                _find_longest_path_from(self._join_graph, self._starting_node)
                if self._starting_node and i == 0
                else _find_longest_path(self._join_graph)
            )

            # If there is no path (i.e., empty graph), break the loop.
            if len(longest_path) == 0:
                break

            # Join tables along the longest path (using edge attributes).
            simplified_table = _join_tables_along_path(
                self._join_graph, longest_path, self._db.tables, how=self._JOIN_HOW
            )
            simplified_tables[simplified_table.name] = simplified_table

            # Update the schema graph.
            simplified_schema = _contract_path(
                simplified_schema, longest_path, simplified_table.name
            )

            # Update the join graph (by deleting the used nodes).
            # Means: A table can be joined at most once.
            for table_name in longest_path:
                self._join_graph.remove_node(table_name)

            # Increment iteration counter.
            i += 1

        # Create simplified database.
        simplified_db = Database(simplified_tables, simplified_schema)

        task_table_names = [
            table_name
            for table_name, table in simplified_db.tables.items()
            if TASK_TABLE_NAME in table.name  # hacky since name-based
        ]
        assert len(task_table_names) == 1
        return simplified_db, task_table_names[0]


########################################################################################
# Helpers. #############################################################################
########################################################################################


def _contract_path(
    schema: nx.MultiDiGraph,
    path: list[str],
    new_name: str,
) -> nx.MultiDiGraph:
    # Iteratively contract all nodes in the path into the first.
    first = path[0]
    for node in path[1:]:
        schema = nx.contracted_nodes(schema, first, node, self_loops=True, copy=False)

    # Rename the first node to the joined table's name.
    return nx.relabel_nodes(schema, {first: new_name}, copy=False)


# Joining. #############################################################################


def _join_tables_along_path(
    graph: nx.DiGraph,
    path: list[str],
    table_dict: dict[str, Table],
    how: T_JOIN_HOW,
) -> Table:
    # Get edges along the path.
    edges: dict[tuple[str, str], dict] = {}
    for i in range(1, len(path)):
        source = path[i - 1]
        target = path[i]
        edge_data = graph.get_edge_data(source, target)
        edges[(source, target)] = edge_data

    # Get initial table and metadata.
    joined_table = table_dict[path[0]]

    # Join along edges.
    for (source, target), edge_data in edges.items():
        # Extract join cols.
        left_on = edge_data["left_on"]
        right_on = edge_data["right_on"]

        # Join.
        logger.debug(f"Joining {source} -> {target} on cols {left_on} = {right_on} ...")
        joined_table = joined_table.join(
            other=table_dict[target],
            left_on=left_on,
            right_on=right_on,
            how=how,
        )

    # Return result.
    return joined_table


# Finding paths. #######################################################################


def _find_longest_path_from(graph: nx.DiGraph, starting_node: str) -> list[str]:
    """
    Find the longest path from a given starting node to any end node.
    """
    longest_path: list[str] = []
    for end_node in graph.nodes():
        for path in nx.all_simple_paths(graph, source=starting_node, target=end_node):
            if len(path) > len(longest_path):
                longest_path = path

    return longest_path


def _find_longest_path(graph: nx.DiGraph) -> list[str]:
    """
    Find the longest path from _any_ starting node to any end node.
    """
    longest_path: list[str] = []
    for starting_node in graph.nodes():
        path_from_node = _find_longest_path_from(graph, starting_node)
        if len(path_from_node) > len(longest_path):
            longest_path = path_from_node

    return longest_path
