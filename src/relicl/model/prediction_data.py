import copy
import logging
import os

import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig

from relicl.config import TGNNConfig, WithConfig
from relicl.effective_config import EffectiveConfig
from relicl.model.embeddings import ContextRowEmbeddings, RowEmbeddings
from relicl.model.extractor import Extractor
from relicl.model.predictor import PredictionTask
from relicl.model.row_selection import (
    AllRowSelector,
    RelevantRowSelector,
    SamplingRelevantRowSelector,
)
from relicl.model.vectorizer import Vectorizer
from relicl.progress import tqdm
from relicl.schema import Ref, RefType
from relicl.timing import timed
from relicl.typing import FusionType, PoolingMethod, TimingEvent
from relicl.utils import store_schema_graph_to_disk

logger = logging.getLogger(__name__)

# PredictionData. ######################################################################


class PredictionData(WithConfig):
    """Holds all information gathered when processing a prediction task."""

    def __init__(self, task: PredictionTask) -> None:
        super().__init__()

        # Task and task information.
        self.task = task
        self.task_table_name = task.train_table.name

        # Information about relevant rows for each table
        self.relevant_rows_by_table: dict[str, RelevantRowSelector] = {}
        self.row_embeddings: dict[
            str, RowEmbeddings
        ] = {}  # table name -> RowEmbeddings

        # Inference DAG.
        try:
            _inference_dag = _create_inference_dag(
                schema_graph=task.db.schema,
                source=self.task_table_name,
                shortest_path_pooling=self.config.fusion.shortest_path_pooling,
            )
        except nx.HasACycle:
            logger.warning(
                "Found cycle in inference graph. Enabling shortest path pooling."
            )
            EffectiveConfig.count("shortest_path_pooling_forced")
            _inference_dag = _create_inference_dag(
                schema_graph=task.db.schema,
                source=self.task_table_name,
                shortest_path_pooling=True,
            )

        self.inference_dag = _inference_dag

        if self.config.output.graphs:
            outdir = os.path.join(
                HydraConfig.get().runtime.output_dir, "graphs", "inference-dag"
            )
            os.makedirs(outdir, exist_ok=True)
            store_schema_graph_to_disk(
                self.inference_dag,
                os.path.join(outdir, f"{task.label}.pdf"),
                tables=task.db.tables,
                shapes=self.config.output.all_shapes,
                highlight_nodes=[task.train_table.name],
            )

    @timed(event_type=TimingEvent.preprocessing)
    def compute_relevant_rows(self) -> None:
        # Make sure no relevant rows have been computed.
        assert len(self.relevant_rows_by_table) == 0

        # Mark all task table rows as relevant, all context table rows as irrelevant.
        for table_name, table in self.task.db.tables.items():
            if table_name == self.task_table_name:
                self.relevant_rows_by_table[table_name] = AllRowSelector(
                    self.task_table_name,
                    pd.concat([self.task.train_table.df, self.task.test_table.df]),
                )
            else:
                self.relevant_rows_by_table[table_name] = SamplingRelevantRowSelector(
                    table_name, table.df
                )

        # Process from task table outwards.
        for s in nx.topological_sort(self.inference_dag):
            # Get the relevant rows.
            relevant_rows_s = self.relevant_rows_by_table[s]

            # Propagate required rows from s to t for all successors t.
            for _, t, key, data in self.inference_dag.out_edges(
                s, keys=True, data=True
            ):
                assert isinstance(t, str)
                ref = data["ref"]

                # Extract tables and keys.
                s_col_name, t_col_name = ref.from_col_name, ref.to_col_name
                s_keys = relevant_rows_s.df[s_col_name]
                self.relevant_rows_by_table[t].mark_keys_relevant(t_col_name, s_keys)

        # Drop context tables that no task row reaches (for instance: a foreign key may
        # never hit a task entity on rel-avito/ad-ctr, since `VisitStream` and
        # `PhoneRequestsStream` share no `AdID` with the ads the task asks about). Their
        # own successors are then empty for the same reason, and are dropped in turn
        # because the topological order visits them after their predecessor.
        for table_name in list(nx.topological_sort(self.inference_dag)):
            if table_name == self.task_table_name:
                continue

            if self.relevant_rows_by_table[table_name].df.empty:
                # On a dry run every context table is empty by construction, no need to
                # log.
                if not self.config.run.dry:
                    logger.warning(
                        f"No row of table '{table_name}' is relevant to the task table. "
                        "Dropping it from the inference graph."
                    )
                EffectiveConfig.add("context_tables_dropped", table_name)
                self.inference_dag.remove_node(table_name)

        EffectiveConfig.maximum(
            "context_tables_used", self.inference_dag.number_of_nodes() - 1
        )

    @timed()
    def extract_all_context_embeddings(self, extractor: Extractor) -> None:
        """Extract embeddings from all tables except the task table."""

        # Process towards task table.
        it = tqdm(
            list(reversed(list(nx.topological_sort(self.inference_dag)))),
            desc="Extract",
            position=2,
            leave=False,
        )
        for context_table_name in it:
            it.set_description(f"Table {context_table_name}")

            if context_table_name == self.task_table_name:
                continue

            # collect the row embeddings from successor tables via the respective keys
            context_embs_list = self.get_context_embeddings(
                context_table_name, extractor
            )

            # and the data frame
            df_relevant = self.relevant_rows_by_table[context_table_name].df
            df_relevant_vec = Vectorizer.new_instance().fit_transform(df_relevant)
            embs = extractor.extract(
                df_relevant_vec, context_embs_list, self.task.task_type
            )

            # if we use early fusion, apply dimensionality reduction
            if self.config.fusion.type is FusionType.early:
                embs.reduce_dimensionality()

            # save the embeddings
            self.row_embeddings[context_table_name] = embs

    @timed()
    def get_context_embeddings(
        self, context_table_name: str, extractor: Extractor
    ) -> list[ContextRowEmbeddings]:
        """
        Collect the row embeddings from successor tables via the respective keys. The
        extractor is needed for normalization (if enabled).
        """
        s, relevant_rows_s = (
            context_table_name,
            self.relevant_rows_by_table[context_table_name],
        )

        context_embs_list: list[ContextRowEmbeddings] = []
        for _, t, data in self.inference_dag.out_edges(context_table_name, data=True):
            # Extract reference.
            ref: Ref = data["ref"]
            s_col_name, t_col_name = ref.from_col_name, ref.to_col_name

            # Get relevant rows object.
            relevant_rows_t = self.relevant_rows_by_table[t]

            propagated = _propagate(
                # Haystack: comes from the successor t.
                self.row_embeddings[t].embs,
                embeddings_key_col=relevant_rows_t.df[t_col_name],
                embeddings_time_cols=relevant_rows_t.df[
                    list(self.task.db.tables[t].time_col_names)
                ],
                # Needles: come from the current table s.
                needles_key_col=relevant_rows_s.df[s_col_name],
                needles_time_cols=relevant_rows_s.df[
                    list(self.task.db.tables[s].time_col_names)
                ],
                # Pooling.
                pooling_method=self.config.fusion.key_pooling,
                allow_context_from_future=self.config.fusion.allow_context_from_future,
                # ContextRowEmbeddings label.
                label=f"__ref_{s_col_name}_to_{t_col_name}",
            )

            # Normalize (if configured).
            if self.config.fusion.type is FusionType.late:
                late_config = self.config.fusion.late
                assert late_config is not None, "fusion.late is unset"

                if late_config.key_normalize:
                    propagated.embs = extractor.normalize(propagated.embs)

            # propagate and store
            context_embs_list.append(propagated)

        return context_embs_list


class PredictionDataGNN(WithConfig):
    def __init__(self, task: PredictionTask) -> None:
        super().__init__()

        # Task and task information.
        self.task = task
        self.task_table_name = task.train_table.name

        tabular_gnn = self.config.fusion.tabular_gnn
        assert tabular_gnn is not None, "fusion.tabular_gnn is unset"
        self.tabular_gnn: TGNNConfig = tabular_gnn

        self.row_embs: dict[str, RowEmbeddings] = {}

    def update_row_embeddings(
        self, extractor: Extractor, remaining_layers: int | None = None
    ) -> None:
        is_first_layer = len(self.row_embs) == 0

        updated_row_embs: dict[str, RowEmbeddings] = {}

        it = tqdm(self.task.db.tables.items(), desc="Extract", position=2, leave=False)
        for table_name, table in it:
            it.set_description(f"Table {table_name}")

            if (
                not self.tabular_gnn.include_task_table
                and table_name == self.task_table_name
            ):
                # do not propagate row embeddings through the task table then
                continue

            if remaining_layers is not None and (
                nx.shortest_path_length(
                    self.task.db.schema.to_undirected(as_view=True),
                    self.task_table_name,
                    table_name,
                )
                >= remaining_layers + 2
                or (remaining_layers == 0 and table_name == self.task_table_name)
            ):
                # not needed in further iterations
                continue

            if is_first_layer:
                logger.debug(f"Initializing row embeddings for table {table_name}")
                df_vec = self.task.df_vec(table_name)
                embs = extractor.extract(df_vec, [], self.task.task_type)
                embs.reduce_dimensionality()
                updated_row_embs[table_name] = embs
            else:
                logger.debug(f"Updating row embeddings for table {table_name}")
                context_embs_list = self.get_context_embeddings(table_name, extractor)
                logger.debug(
                    f"Obtained context embeddings from {len(context_embs_list)} tables"
                )
                if len(context_embs_list) > 0:
                    df_vec = self.task.df_vec(table_name)
                    embs = extractor.extract(
                        df_vec, context_embs_list, self.task.task_type
                    )
                    embs.reduce_dimensionality()
                    updated_row_embs[table_name] = embs
                else:
                    # no active neighbors -> shortcut
                    updated_row_embs[table_name] = self.row_embs[table_name]

        self.row_embs = updated_row_embs

    def get_context_embeddings(
        self, context_table_name: str, extractor: Extractor
    ) -> list[ContextRowEmbeddings]:
        schema = self.task.db.schema
        edges = list(schema.in_edges(context_table_name, data=True)) + list(
            schema.out_edges(context_table_name, data=True)
        )

        context_embs_list: list[ContextRowEmbeddings] = []
        s_df = self.task.df(context_table_name)
        for edge in edges:
            # normalize the edge so that s is the context_table (from)
            s, t, data = edge
            ref: Ref = data["ref"]
            assert s != t
            if t == context_table_name:
                s, t = t, s
                ref = ref.inverse()
            assert s == context_table_name

            if not self.tabular_gnn.include_task_table and t == self.task_table_name:
                # we do not integrate row embeddings for the task table then
                continue

            if self.tabular_gnn.only_towards_task_table:
                if nx.shortest_path_length(
                    self.task.db.schema.to_undirected(as_view=True),
                    self.task_table_name,
                    context_table_name,
                ) > nx.shortest_path_length(
                    self.task.db.schema.to_undirected(as_view=True),
                    self.task_table_name,
                    t,
                ):
                    # the context table is further away than the table from which to pool
                    # -> skip
                    continue

            t_df = self.task.df(t)
            s_col_name, t_col_name = ref.from_col_name, ref.to_col_name

            # get corresponding context embeddings
            propagated = _propagate(
                # Haystack: comes from the successor t.
                self.row_embs[t].embs,
                embeddings_key_col=t_df[t_col_name],
                embeddings_time_cols=t_df[list(self.task.db.tables[t].time_col_names)],
                # Needles: come from the current table s.
                needles_key_col=s_df[s_col_name],
                needles_time_cols=s_df[list(self.task.db.tables[s].time_col_names)],
                # Pooling.
                pooling_method=self.config.fusion.key_pooling,
                allow_context_from_future=self.config.fusion.allow_context_from_future,
                # ContextRowEmbeddings label.
                label=f"__ref_{s_col_name}_to_{t_col_name}",
            )

            # Normalize (if configured).
            if self.config.fusion.type is FusionType.late:
                late_config = self.config.fusion.late
                assert late_config is not None, "fusion.late is unset"

                if late_config.key_normalize:
                    propagated.embs = extractor.normalize(propagated.embs)

            # propagate and store
            context_embs_list.append(propagated)

        return context_embs_list


# Utilities. ###########################################################################


@timed(event_type=TimingEvent.fusion)
def _propagate(
    # Haystack.
    embeddings: npt.NDArray,
    embeddings_key_col: pd.Series,
    embeddings_time_cols: pd.DataFrame,
    # Needles.
    needles_key_col: pd.Series,
    needles_time_cols: pd.DataFrame,
    # Pooling.
    pooling_method: PoolingMethod,
    allow_context_from_future: bool,
    # ContextRowEmbeddings label.
    label: str,
) -> ContextRowEmbeddings:
    """
    Produces one pooled embedding for each needle, ensuring that the pooled
    embeddings belong to a point in time no later than the respective needle.
    """

    # Extract dimensions.
    n_estimators, n_rows, embed_dim = embeddings.shape
    n_needles = len(needles_key_col)

    # Prevent indexing issues in the `loc` method.
    assert embeddings_key_col.dtype == needles_key_col.dtype, (
        f"Key column dtype mismatch: haystack has {embeddings_key_col.dtype}, "
        f"needles have {needles_key_col.dtype}"
    )

    # Determine embedding indices through a join. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Want: Mapping
    #
    #   needle_idx -> { haystack_idx }
    #
    # where the needle time >= haystack time; see below.

    haystack_df = pd.DataFrame(
        dict(
            # Numpy conversion prevents any pandas index issues.
            haystack_key=embeddings_key_col.to_numpy(),
            haystack_idx=np.arange(n_rows),
        )
    )

    needles_df = pd.DataFrame(
        dict(
            # Numpy conversion prevents any pandas index issues.
            needle_key=needles_key_col.to_numpy(),
            needle_idx=np.arange(len(needles_key_col)),
        )
    )

    # Set up temporal filtering if it is used.
    if not allow_context_from_future:
        # Haystack: Take the latest date to avoid leakage.
        embeddings_time_col = pd.to_datetime(embeddings_time_cols.max(axis=1))

        # Haystack: Rows without dates can be used regardless of the needles' times.
        embeddings_time_col = embeddings_time_col.fillna(pd.Timestamp.min)

        # Needles: Take the latest date to avoid leakage.
        needles_time_col = pd.to_datetime(needles_time_cols.max(axis=1))

        # Needles: Rows without dates can use any haystack item.
        needles_time_col = needles_time_col.fillna(pd.Timestamp.max)

        # add the columns
        haystack_df["haystack_time"] = embeddings_time_col.to_numpy()
        needles_df["needle_time"] = needles_time_col.to_numpy()

    # now do the join
    pairs = needles_df.merge(
        haystack_df,
        left_on="needle_key",
        right_on="haystack_key",
    )

    # and apply the temporal filtering, if used
    if not allow_context_from_future:
        pairs = pairs[pairs["haystack_time"] <= pairs["needle_time"]]

    # Pool haystack embeddings (ignoring NaNs). ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings_t = torch.tensor(embeddings, device=device)  # [n_est, n_rows, dim]

    # Output: NaN for needles with no matches, filled in below.
    pooled = torch.full(
        (n_estimators, n_needles, embed_dim),
        float("nan"),
        device=device,
        dtype=embeddings_t.dtype,
    )
    counts_t = torch.zeros(n_needles, device=device, dtype=torch.int32)

    if len(pairs) > 0:
        emb_indices = torch.tensor(pairs["haystack_idx"].to_numpy(), device=device)
        needle_indices = torch.tensor(pairs["needle_idx"].to_numpy(), device=device)

        # Gather all relevant embeddings.
        gathered = embeddings_t[:, emb_indices, :]  # [n_est, total_matches, dim]

        # Build index tensor for scatter_reduce: shape must match gathered.
        index = needle_indices[None, :, None].expand(n_estimators, -1, embed_dim)

        match pooling_method:
            case PoolingMethod.sum:
                src = gathered.nan_to_num(0.0)
                pooled.scatter_reduce_(1, index, src, reduce="sum", include_self=False)

            case PoolingMethod.max:
                src = gathered.nan_to_num(float("-inf"))
                pooled.scatter_reduce_(1, index, src, reduce="amax", include_self=False)

            case PoolingMethod.mean:
                # Sum non-NaN values, divide by count of non-NaN values per position.
                src = gathered.nan_to_num(0.0)
                pooled.scatter_reduce_(1, index, src, reduce="sum", include_self=False)
                valid_counts = torch.zeros_like(pooled)
                valid_counts.scatter_reduce_(
                    1,
                    index,
                    (~gathered.isnan()).to(embeddings_t.dtype),
                    reduce="sum",
                    include_self=False,
                )
                # Clamp prevents div-by-zero; the explicit NaN assignment restores
                # slots where every value in the group was NaN.
                pooled = pooled / valid_counts.clamp(min=1)
                pooled[valid_counts == 0] = float("nan")

        # Counts: number of haystack matches per needle.
        counts_t.scatter_add_(
            dim=0,
            index=needle_indices,
            src=torch.ones(len(needle_indices), device=device, dtype=torch.int32),
        )

    return ContextRowEmbeddings(
        embs=pooled.cpu().numpy(),
        counts=counts_t.cpu().numpy(),
        label=label,
    )


@timed()
def _create_inference_dag(
    schema_graph: nx.MultiDiGraph, source, shortest_path_pooling=False
):
    """Create a DAG along which inference takes place."""

    schema_graph = copy.deepcopy(schema_graph)
    inference_dag: nx.MultiDiGraph = nx.MultiDiGraph()

    # order vertices by distance from source
    layers = list(nx.bfs_layers(schema_graph.to_undirected(as_view=True), source))

    # build inference graph
    for layer in layers:
        for n in layer:
            inference_dag.add_node(n)

            for s, _, key, data in list(schema_graph.in_edges(n, data=True, keys=True)):
                ref = data["ref"]
                assert ref.type == RefType.many_to_one
                # For reference between tables in the same layer, we want to
                # propagate embeddings along the foreign key direction; thus ignore
                # these edges here (will be processed later when processing s)
                if s not in layer:
                    inference_dag.add_edge(n, s, key, ref=ref.inverse())
                    schema_graph.remove_edge(s, n, key)

            for _, t, key, data in list(
                schema_graph.out_edges(n, data=True, keys=True)
            ):
                ref = data["ref"]
                assert ref.type == RefType.many_to_one
                if not (shortest_path_pooling and t in layer):
                    inference_dag.add_edge(n, t, key, ref=ref)
                else:
                    logger.debug(
                        f"Ignoring edge {_}.{ref.from_col_name}->{t}.{ref.to_col_name} due to shortest path pooling"
                    )
                schema_graph.remove_edge(n, t, key)

    try:
        nx.find_cycle(inference_dag)
        raise nx.HasACycle()
    except nx.NetworkXNoCycle:
        pass

    return inference_dag
