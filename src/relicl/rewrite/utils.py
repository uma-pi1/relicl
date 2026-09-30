import logging
import os

import networkx as nx
from hydra.core.hydra_config import HydraConfig
from matplotlib.backends.backend_pdf import PdfPages

from relicl.config import RelICLConfig
from relicl.schema import RDLTask, Ref, Table
from relicl.typing import SaveStepFn
from relicl.utils import store_rdl_task_graph_to_disk

logger = logging.getLogger(__name__)


def eliminate_unreachable_tables(
    rdl_task: RDLTask, save_step: SaveStepFn | None = None
) -> bool:
    """Eliminate all tables that are not reachable from the task table."""

    removed_tables: list[str] = []
    schema: nx.MultiDiGraph = rdl_task.db.schema
    reachable = set(
        nx.dfs_preorder_nodes(
            schema.to_undirected(as_view=True), rdl_task.task_table_name
        )
    )
    for u in list(schema.nodes):
        if u not in reachable:
            schema.remove_node(u)
            del rdl_task.db.tables[u]
            logger.debug(f"Eliminated unreachable table {u}")
            removed_tables.append(u)

    if save_step is not None and len(removed_tables) > 0:
        save_step(f"Eliminated unreachable tables: {removed_tables}")

    return len(removed_tables) > 0


def count_on_one_side(schema: nx.MultiDiGraph, n: str) -> int:
    """
    Counts how often this table occurs on the one-side of a one-to-many or one-to-one
    relationship.
    """

    count = 0
    for u, _, data in schema.in_edges(n, data=True):
        ref: Ref = data["ref"]
        if ref.is_to_one:
            count += 1
    for u, _, data in schema.out_edges(n, data=True):
        ref = data["ref"]
        if ref.is_from_one:
            count += 1

    return count


def count_on_many_side(schema: nx.MultiDiGraph, node: str) -> int:
    """
    Counts how often this table occurs on the many-side of a one-to-many or many-to-many
    relationship.
    """

    count = 0
    for u, _, data in schema.in_edges(node, data=True):
        ref: Ref = data["ref"]
        if ref.is_to_many:
            count += 1
    for u, _, data in schema.out_edges(node, data=True):
        ref = data["ref"]
        if ref.is_from_many:
            count += 1

    return count


def remove_edge_by_pulling_fk_rows(
    rdl_task: RDLTask,
    s: str,
    t: str,
    k: int,
    ref: Ref,
    pull_t_into_s: bool = True,
    rewire_edges: bool = False,
    drop_pulled_keys: bool = False,
) -> Table:
    """Removes the edge (s, t, k) of type ref by joining t into s.

    t needs to be on the one-side of relationship ref. If rewrite_edges is set, all the
    other edges incident to t are rewired to s.

    If pull_t_into_s is False, then the roles of s and t reverse.

    returns the newly created table
    """

    assert s != t

    # remove the edge
    schema: nx.MultiDiGraph = rdl_task.db.schema
    schema.remove_edge(s, t, k)

    # normalize order
    if not pull_t_into_s:
        s, t = t, s
        ref = ref.inverse()
    assert ref.is_to_one

    # create the new table
    db = rdl_task.db
    new_table = db.tables[s].join(
        db.tables[t],
        left_on=[ref.from_col_name],
        right_on=[ref.to_col_name],
        how="left",
        drop_right_keys=drop_pulled_keys,
    )

    # update the table in the schema
    db.tables[s] = new_table
    db.rename_table(s, new_table.name)
    if rdl_task.task_table_name == s:
        rdl_task.task_table_name = new_table.name

    # rewire incident edges to table
    # TODO some types many need to be changed to many-to-many
    if rewire_edges:
        for u, _, k, d in list(schema.in_edges(t, keys=True, data=True)):
            assert u != t
            schema.remove_edge(u, t, k)
            schema.add_edge(u, new_table.name, **d)
        for _, v, k, d in list(schema.out_edges(t, keys=True, data=True)):
            assert v != t
            schema.remove_edge(t, v, k)
            schema.add_edge(new_table.name, v, **d)

    logger.debug(
        f"Merged {t} into {s}" + (" and rewired {t}'s edges" if rewire_edges else "")
    )

    return new_table


def save_step_fn(rdl_task: RDLTask, filename: str | None = None) -> SaveStepFn:
    """Returns a function to save a rewrite step into a graph"""
    config = RelICLConfig.instance()
    if config.output.graphs:
        if filename is None:
            outdir = os.path.join(HydraConfig.get().runtime.output_dir, "graphs")
            filename = os.path.join(outdir, "schema-rewrite-steps.pdf")
        pdf = PdfPages(filename)  # type: ignore

        def f(title: str = "", close: bool = False) -> None:
            if not close:
                store_rdl_task_graph_to_disk(
                    rdl_task, pdf, title=title, shapes=config.output.all_shapes
                )
            else:
                pdf.close()

        return f  # type: ignore[return-value]
    else:
        return SaveStepFn.default()
