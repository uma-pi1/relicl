import networkx as nx

from relicl.schema import Database

from .joiner import Joiner

########################################################################################
# Class: RelationshipJoiner ############################################################
########################################################################################


class RelationshipJoiner(Joiner):
    def __init__(self, db: Database, task_table_name: str) -> None:
        super().__init__(db, starting_node=task_table_name)

    def _build_join_graph(self) -> nx.DiGraph:
        graph: nx.DiGraph = nx.DiGraph()
        graph.add_nodes_from(self._db.schema.nodes)
        for u, v, data in self._db.schema.edges(data=True):
            ref = data["ref"]
            graph.add_edge(
                u, v, left_on=ref.from_col_name, right_on=ref.to_col_name, ref=ref
            )
        return graph
