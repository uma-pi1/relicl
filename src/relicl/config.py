import os.path as osp
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from relbench.base import TaskType

from relicl.typing import (
    BackendType,
    DataSourceType,
    DimReductionMethod,
    FusionType,
    InferenceType,
    ModelType,
    PoolingMethod,
    RegressionOutput,
    SimplifyMethod,
    TargetTransform,
    VectorizerMethod,
)

########################################################################################
# Global constants.
CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = Path(
    osp.abspath(
        next(
            parent
            for parent in CURRENT_FILE.parents
            if (parent / "pyproject.toml").exists()
        )
    )
)
RESULTS_FILENAME = "_RESULTS.yaml"

# Each ensemble member records in this file which Hydra overrides created it.
MEMBER_OVERRIDES_FILENAME = "_ENSEMBLE_MEMBER.yaml"

# For relbench.
TASK_TABLE_NAME = "task_table"
TASK_TABLE_ID_COL_NAME = "__task_id"


########################################################################################
# Task
@dataclass
class RelBenchConfig:
    db: str = "rel-f1"
    task: str = "driver-top3"


########################################################################################
# Method
@dataclass
class MethodConfig:
    model: ModelType = ModelType.tabicl
    """Name of the backbone tabular model."""

    test_batch_size: int = 10_000
    """Batch size for inference on the test table."""

    seed: int = 0
    """PRNG seed"""

    target_transform: TargetTransform = TargetTransform.none
    """
    Regression only: transform applied to the target before the backbone is fitted on
    it, inverted again on the predictions.
    """

    use_max_reproducibility: bool = False
    """
    Ensures exact reproducibility of results by using deterministic CUDA algorithms
    and requiring relevant environment variables to be set (CUBLAS_WORKSPACE_CONFIG 
    and PYTHONHASHSEED). This may increase run time.
    """


# Pretrained TabICL checkpoint per task type.
TABICL_CHECKPOINTS = {
    TaskType.BINARY_CLASSIFICATION: "tabicl-classifier-v2-20260212.ckpt",
    TaskType.REGRESSION: "tabicl-regressor-v2-20260212.ckpt",
}


@dataclass
class TabICLConfig:
    n_estimators: int = 8  # TabICL's default.
    verbose: bool = False

    checkpoint_version: str | None = None
    """
    Pretrained checkpoint. Left unset it is derived from the task type via
    `TABICL_CHECKPOINTS`; set it only to pin a checkpoint that is not the default.
    """

    model_path: str | None = None
    """
    Checkpoint to load instead of the pretrained one. Only applied to the predictor.
    """

    outlier_threshold: float = 4.0
    """
    Parameter for TabICL: features are clipped where `|z| > outlier_threshold`. 
    """

    norm_methods: list[str] | None = None
    """
    Parameter for TabICL: normalizations the ensemble members draw from (`none`,
    `power`, `quantile`, `quantile_rtdl`, `robust`). TabICL default: `["none", "power"]`.
    """

    feat_shuffle_method: str = "latin"
    """
    Parameter for TabICL: feature permutation strategy across ensemble members (`none`,
    `shift`, `random`, `latin`).
    """

    regression_output: RegressionOutput = RegressionOutput.median
    """Parameter for TabICLRegressor: Point estimate for regression."""


def resolve_tabicl_checkpoint(task_type: TaskType) -> str:
    """The checkpoint TabICL will actually load for `task_type`."""
    tabicl_config = RelICLConfig.instance().tabicl
    assert tabicl_config is not None, "tabicl is unset; method.model is not tabicl"

    configured = tabicl_config.checkpoint_version
    if configured is not None:
        return configured

    # Sanity check: only the two supported task types have a checkpoint.
    assert task_type in TABICL_CHECKPOINTS, f"No TabICL checkpoint for {task_type}"
    return TABICL_CHECKPOINTS[task_type]


@dataclass
class TabPFNConfig:
    n_estimators: int = 4  # default: 8

    data_source: DataSourceType = DataSourceType.test
    """Data source of extracted embeddings"""


@dataclass
class TabFMConfig:
    n_estimators: int = 4  # default: 32
    verbose: bool = False


########################################################################################
# Fusion


@dataclass
class LateFusionConfig:
    key_normalize: bool = True
    """Normalize pooled key embeddings"""

    successor_pooling: PoolingMethod = PoolingMethod.mean
    """Pooling of embeddings from the successor context tables"""

    successor_normalize: bool = True
    """Normalize the pooled successor embeddings"""

    combine_pooling: PoolingMethod = PoolingMethod.mean
    """How context row embeddings and pooled successor embeddings are combined"""

    combine_normalize: bool = True
    """Normalize the combined embeddings"""

    combine_successor_weight: float = 1.0
    """Weight of pooled successor table embeddings in combined pooling"""

    task_pooling: PoolingMethod = PoolingMethod.mean
    """Final pooling as late fusion in the task table"""

    task_normalize: bool = True
    """Normalize the context embeddings for the task table"""

    task_successor_weight: float = 1.0
    """Weight of successor tables for the task table"""


@dataclass
class EarlyFusionConfig:
    reduce_dim: bool = True
    """Whether to reduce the dimensionality of the embeddings."""

    reduce_per_estimator: bool = False
    """
    When set, dimensionality reduction is applied for each estimator separately. The
    final dimensionality is then `n_estimators * reduce_dim`. If unset, it is applied 
    across all estimators and the final dimensionality is `reduce_dim`.
    """

    reduction_method: DimReductionMethod = DimReductionMethod.gaussian_random_projection
    """Method to reduce the dimensionality of the embeddings."""

    reduction_dim: int = 32
    """Dimensionality of the embeddings after reduction."""


@dataclass
class TGNNConfig:
    n_layers: int = 3
    include_task_table: bool = True
    only_towards_task_table: bool = True


@dataclass
class FusionConfig:
    inference: InferenceType = InferenceType.one_pass
    """Whether context propagation is a single pass or multi-layer."""

    tabular_gnn: TGNNConfig | None = None
    """
    Parameters of multi-layer propagation, `None` unless `inference` is `tabular_gnn`.
    """

    key_pooling: PoolingMethod = PoolingMethod.mean
    """Method to pool row embeddings of all rows for a single key"""

    shortest_path_pooling: bool = False
    """Only pool along the shortest paths towards the task table. Ignores any other
    connection between the tables."""

    type: FusionType = FusionType.early
    """Whether to use early or late fusion."""

    late: LateFusionConfig | None = None
    """
    Parameters of late fusion, `None` unless `type` is `late`. Both variants are set by
    the `fusion` config group (`config/fusion/`), which fills in the selected one and
    nulls the other, so that a run's config only carries the parameters in effect.
    """

    early: EarlyFusionConfig | None = None
    """Parameters of early fusion, `None` unless `type` is `early`; see `late`."""

    allow_context_from_future: bool = False
    """
    Whether to use context embeddings from the future (but never beyond the train/test
    cutoff).
    """


########################################################################################
# Rewrite and sampling


@dataclass
class RewriteConfig:
    count_features: bool = True
    """For each key, include counts of successor rows as additional features"""

    count_features_after_simplify: bool = False
    """Add successor counts after schema simplification instead of before."""

    count_feature_window_days: list[int] | None = None
    """Additional successor counts restricted to the last N days, one column per entry."""

    simplify: SimplifyMethod | None = SimplifyMethod.default

    eliminate_junction_tables_depth: int = 1
    """When eliminating junction tables, how many "hops" should be captured afterwards."""


@dataclass
class TrainTableSampleConfig:
    sample_size: int = 50_000
    """
    The number of rows sampled from the train table. A soft cap: under
    `restrict_to_test_keys` it is raised to fit one row per test key rather than leaving
    some test entities without their own history in the context window.
    """

    # TODO shouldn't this be false by default?
    restrict_to_test_keys: bool = True
    """
    Whether to restrict the train table to only include rows with keys present in the
    test table.
    """

    anchor_budget_fraction: float = 1.0
    """
    Share of `sample_size` that the per-test-key rows ("anchors") may claim. The depth
    per key is derived from it as `sample_size * fraction / n_test_keys`, so it adapts
    to however many distinct keys a test batch holds instead of being a fixed count.
    It is rounded up to at least 1, so with more test keys than budget the anchors can
    claim more than this share and grow `sample_size` (see there). 0 disables anchoring.
    """

    only_fill_from_test_keys: bool = False
    """
    If True, only fill up the training set from rows with matching keys. If False,
    fill up from all rows.
    """

    recency_half_life_steps: float | None = None
    """
    Half-life of the recency weight used when scoring training rows, in task timesteps:
    a row this many steps older than the newest one is half as likely to be sampled.
    Left unset every row is equally likely (i.e., uniform sampling).
    """


@dataclass
class SamplingConfig:
    context_tables_sample_size: int = 50_000
    """The maximum number of rows sampled from context tables."""

    train: TrainTableSampleConfig = field(default_factory=TrainTableSampleConfig)


@dataclass
class TextEncoderConfig:
    enable: bool = False
    """
    Embed free-text columns with a sentence encoder (TabICL considers it as categories
    otherwise).
    """

    # Embedding computation. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    """Sentence-transformers model from HuggingFace."""

    batch_size: int = 256

    # Dimensionality reduction. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    reduction_dim: int = 16
    """Resulting dimension of embedded text columns after dimensionality reduction."""

    reduction_method: DimReductionMethod = DimReductionMethod.pca
    """Which dimensionality reduction method to use."""

    # Heuristic to distinguish between strings and text. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    min_unique: int = 20
    """Distinct values a string column needs before it counts as prose."""

    heuristic_sample_size: int = 10_000
    """Rows the prose heuristic inspects."""

    min_mean_words: float = 4.0
    """Mean word count, over the distinct values, before a column counts as prose."""


@dataclass
class VectorizerConfig:
    type: VectorizerMethod = VectorizerMethod.default
    n_jobs: int | None = -1
    """Number of jobs to run in parallel. `None` means 1, and -1 means using all processors."""

    array_col_width: int = 3
    """Array columns are exploded into this many columns."""

    relative_time: bool = False
    """
    Encode time columns as an age in days relative to the newest timestamp in the frame
    (the evaluation cutoff for the task table). Calendar components are emitted either 
    way.
    """

    text: TextEncoderConfig = field(default_factory=TextEncoderConfig)


########################################################################################
# General
@dataclass
class TraceConfig:
    enable: bool = True
    """Whether to enable tracing/logging (trace.yaml)"""

    echo: bool = False
    disabled_events_regex: str = ".*_debug"
    enabled_events_regex: str = ""
    timings: bool = False


@dataclass
class OutputConfig:
    trace: TraceConfig = field(default_factory=TraceConfig)

    predictions: bool = True
    """Whether to output predictions (e.g., in predictions-test.parquet)"""

    progress: bool = True
    """Whether to show tqdm progress bars; see `relicl.progress`."""

    graphs: bool = True
    """Whether to export various graphs to disk."""

    all_shapes: bool = False
    """Whether exported test/schema/query graphs are annotated with their shape.
    Expensive (and not optimized), but useful to gain insight."""

    dump_final_table: bool = False
    """Whether to dump the final feature table handed to the tabular backbone; see
    `relicl.dump`. Only affects validation timesteps under `fusion.type=early`."""


@dataclass
class RunConfig:
    val: bool = False
    """Whether to run on validation data."""

    test: bool = True
    """Whether to run on test data."""

    dry: bool = False
    """If set, performs a dry run using empty tables"""


@dataclass
class EnsembleConfig:
    enable: bool = True

    members: list[list[str]] = field(
        default_factory=lambda: [
            [],  # don't change anything
        ]
    )
    """
    One entry per member, each a list of Hydra overrides (e.g. `[fusion=late,
    method.seed=5]`). 
    """

    average_logits: bool = True


@dataclass
class RelICLConfig:
    backend: BackendType = BackendType.pandas
    relbench: RelBenchConfig = field(default_factory=RelBenchConfig)
    run: RunConfig = field(default_factory=RunConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    # Parameters of the backbone selected by `method.model`.
    tabicl: TabICLConfig | None = None
    tabpfn: TabPFNConfig | None = None
    tabfm: TabFMConfig | None = None
    fusion: FusionConfig = field(default_factory=FusionConfig)
    rewrite: RewriteConfig = field(default_factory=RewriteConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    vectorizer: VectorizerConfig = field(default_factory=VectorizerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)

    @classmethod
    def has_instance(cls) -> bool:
        global _global_config
        return _global_config is not None

    @classmethod
    def instance(cls) -> "RelICLConfig":
        global _global_config
        if _global_config is None:
            raise Exception("RelICLConfig not initialized.")
        return _global_config

    @classmethod
    def set(cls, config: "RelICLConfig") -> None:
        global _global_config
        if _global_config is None:
            _global_config = config
        else:
            raise Exception("Global RelICLConfig already set.")


_global_config: RelICLConfig | None = None


class WithConfig:
    config: RelICLConfig

    def __init__(self):
        self.config = RelICLConfig.instance()


########################################################################################
# Flatten utils. #######################################################################
########################################################################################


def _flatten(config: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in config.keys():
        if isinstance(config[key], Mapping):
            result.update(_flatten(config[key], prefix + key + "."))
        else:
            result[prefix + key] = config[key]

    return result


# Stands in for a key that one of the compared configs does not have at all, as opposed
# to having it set to a different value.
NOT_EXISTENT = "<not existent>"


def flatten_and_diff(configs: Iterable[Mapping[str, Any]]) -> Mapping[str, list[Any]]:
    keys: dict[str, None] = {}
    flattened_configs: list[dict[str, Any]] = []
    for c in configs:
        config = _flatten(c)
        flattened_configs.append(config)
        keys.update(dict.fromkeys(config.keys()))

    diff: dict[str, list[Any]] = {}
    for k in keys.keys():
        is_relevant = False
        values: list[Any] = []
        for i, config in enumerate(flattened_configs):
            if k not in config:
                is_relevant = True
                values.append(NOT_EXISTENT)
                continue
            else:
                value = flattened_configs[i].get(k)
                values.append(value)

            if not is_relevant and value != flattened_configs[0][k]:
                is_relevant = True

        if is_relevant:
            diff[k] = values

    return diff
