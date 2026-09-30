import io
import logging
import os
import os.path as osp
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable

import hydra
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from relbench.base import TaskType
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from tabicl import TabICLClassifier  # type: ignore[attr-defined]

from relicl import rng
from relicl.config import (
    REPO_ROOT,
    DimReductionMethod,
    EarlyFusionConfig,
    FusionType,
    RelICLConfig,
)
from relicl.model import PredictionTask
from relicl.model.relicl import RelICLModel
from relicl.rng import seed
from relicl.schema import Database, Table
from relicl.trace import Trace
from relicl.utils import compute_binary_classification_metrics

cs = ConfigStore.instance()
cs.store(name="base_config", node=RelICLConfig)

########################################################################################
# Helpers. #############################################################################
########################################################################################

# Task loading. ########################################################################


@dataclass
class _TaskSpec:
    """Everything needed to fetch and binarize one OpenML task."""

    openml_id: int
    positive_class: str | None  # None → use a custom transform
    description: str
    custom_target: Callable[[pd.Series], pd.Series] | None = None


def _top_class_vs_rest(series: pd.Series) -> pd.Series:
    """Binaries a multiclass target: most-frequent class = 1, rest = 0."""
    top = series.value_counts().idxmax()
    return (series == top).astype(int)


_REGISTRY: dict[str, _TaskSpec] = {
    # -----------------------------------------------------------------------
    # Classics / benchmarks
    # -----------------------------------------------------------------------
    "credit-g": _TaskSpec(
        openml_id=31,
        positive_class="good",
        description="German credit risk (1k rows, 20 features). Your current baseline.",
    ),
    "adult": _TaskSpec(
        openml_id=1590,
        positive_class=">50K",
        description="Census income >50K (48k rows, 14 features). Moderate difficulty.",
    ),
    "bank-marketing": _TaskSpec(
        openml_id=1461,
        positive_class="2",  # OpenML encodes 'yes' as '2'
        description="Term-deposit subscription (45k rows, 16 features). Class imbalance ~11%.",
    ),
    # -----------------------------------------------------------------------
    # Moderate–hard
    # -----------------------------------------------------------------------
    "nomao": _TaskSpec(
        openml_id=1486,
        positive_class="2",
        description="Location-duplicate detection (34k rows, 118 features). Dense, noisy.",
    ),
    "eye-movements": _TaskSpec(
        openml_id=1044,
        positive_class=None,
        custom_target=_top_class_vs_rest,
        description="Eye-tracking fixation type (10k rows, 27 features). Noisy boundaries.",
    ),
    "bioresponse": _TaskSpec(
        openml_id=4134,
        positive_class="1",
        description="Molecular biological activity (3.8k rows, 1776 features). Very high-dim.",
    ),
    "phoneme": _TaskSpec(
        openml_id=1489,
        positive_class="2",
        description="Phoneme classification (5.4k rows, 5 features). Non-linear boundary.",
    ),
    # -----------------------------------------------------------------------
    # Hard / low ceiling
    # -----------------------------------------------------------------------
    "miniboone": _TaskSpec(
        openml_id=41150,
        positive_class="1",
        description="Particle physics signal vs background (130k rows, 50 features). ~97% ceiling.",
    ),
    "jannis": _TaskSpec(
        openml_id=168911,
        positive_class=None,
        custom_target=_top_class_vs_rest,
        description="NeurIPS AutoML benchmark (83k rows, 54 features). Intentionally hard.",
    ),
    "philippine": _TaskSpec(
        openml_id=41145,
        positive_class=None,
        custom_target=_top_class_vs_rest,
        description="NeurIPS AutoML benchmark (5.8k rows, 308 features). High-dim, hard.",
    ),
    "kdd-churn": _TaskSpec(
        openml_id=1111,
        positive_class="True",
        description="KDD Cup 09 churn (50k rows, 230 features). Extreme class imbalance ~7%.",
    ),
    # -----------------------------------------------------------------------
    # Genuinely hard / low AUROC ceiling (~70-75%)
    # -----------------------------------------------------------------------
    "compass": _TaskSpec(
        openml_id=42193,
        positive_class="1",
        description="COMPAS recidivism (7k rows, 12 features). ~72% ceiling; ethics benchmark.",
    ),
    "road-safety": _TaskSpec(
        openml_id=41926,
        positive_class="1",
        description="UK road accident fatality (111k rows, 32 features). Low ~75% ceiling.",
    ),
    "kr-vs-kp": _TaskSpec(
        openml_id=3,
        positive_class="won",
        description="Chess endgame (3.2k rows, 36 features). All categorical, ~100% ceiling.",
    ),
    "qsar-biodeg": _TaskSpec(
        openml_id=1494,
        positive_class="2",
        description="Ready biodegradability (1.1k rows, 41 features). Small, mid-dimensional.",
    ),
    "spambase": _TaskSpec(
        openml_id=44,
        positive_class="1",
        description="Spam detection (4.6k rows, 57 features). Mid-dimensional, numeric.",
    ),
    "jasmine": _TaskSpec(
        openml_id=41143,
        positive_class="1",
        description="NeurIPS AutoML benchmark (3k rows, 144 features). Mostly categorical.",
    ),
    "churn": _TaskSpec(
        openml_id=40701,
        positive_class="1",
        description="Telecom churn (5k rows, 20 features). Imbalanced at 14% positive.",
    ),
}


# ---------------------------------------------------------------------------
# Public enum + loader
# ---------------------------------------------------------------------------


def load_task(
    task: str,
) -> tuple[pd.DataFrame, pd.Series, str]:
    """
    Fetch an OpenML binary-classification task.

    Returns
    -------
    data : pd.DataFrame
        Feature matrix (raw, as returned by OpenML — may contain categoricals).
    target_col : pd.Series
        Binary integer target (1 = positive class).
    target_col_name : str
        Name of the target column.
    """
    if task not in _REGISTRY:
        raise ValueError(f"Unknown task: {task}")

    spec = _REGISTRY[task]
    data, target_col = fetch_openml(
        data_id=spec.openml_id,
        as_frame=True,
        return_X_y=True,
    )

    if spec.custom_target is not None:
        target_col = spec.custom_target(target_col)
    else:
        # Normalize string comparison: strip whitespace, case-insensitive
        positive = spec.positive_class.strip().lower()  # type: ignore[union-attr]
        target_col = (
            target_col.astype(str).str.strip().str.lower() == positive
        ).astype(int)

    target_col_name = str(target_col.name)
    return data, target_col, target_col_name


# Output. ##############################################################################


@contextmanager
def suppressed_output():
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = io.StringIO()
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        logging.disable(logging.NOTSET)


########################################################################################
# Main function. #######################################################################
########################################################################################


@hydra.main(
    version_base="1.3", config_path=str(REPO_ROOT / "config"), config_name="relicl"
)
def main(cfg: RelICLConfig) -> None:
    RelICLConfig.set(cfg)
    out_dir = osp.join(str(REPO_ROOT), "sandbox-outputs")
    os.makedirs(out_dir, exist_ok=True)

    results: list[dict[str, Any]] = []

    # Define HP grid. ##############################################################

    grid = dict(
        # reduction_dim=[1, 2, 4, 8, 16, 32, 64],
        reduction_dim=[16, 32, 64],
        reduction_method=[
            DimReductionMethod.gaussian_random_projection,
            # DimReductionMethod.sign_random_projection,
            # DimReductionMethod.sparse_random_projection,
            DimReductionMethod.pca,
            # DimReductionMethod.pca_whiten,
            # DimReductionMethod.svd,
            # DimReductionMethod.svd_whiten,
            # DimReductionMethod.random_k,
            DimReductionMethod.first_k,
        ],
    )

    # Define experiment structure. ####################################################

    n_inter_tables_options = [0, 1, 2, 3, 5, 10]
    data_split_options = [True, False]
    seeds = [0, 10, 50, 100]

    datasets = ["bioresponse", "phoneme", "philippine", "compass", "credit-g"]
    # datasets = ["credit-g"]
    assert all([d in _REGISTRY for d in datasets])

    for split_seed in seeds:
        for dataset in datasets:
            # Load data. ###################################################################

            # Load tabular dataset.
            data, target_col, target_col_name = load_task(dataset)

            # Create task table. #############################################

            task_df = target_col.to_frame().assign(_task_id=np.arange(len(target_col)))

            # Train & test split. ##########################################################

            train_ids, test_ids = train_test_split(
                np.arange(len(task_df)),
                test_size=0.2,
                random_state=0,
                shuffle=True,
                stratify=target_col,
            )

            # Sanity check: exactly once.
            assert set(train_ids).isdisjoint(set(test_ids))
            assert len(set(train_ids).union(set(test_ids))) == len(target_col)

            # Create RelICL tables. ########################################################

            # Tables.
            train_task_table = Table.create(
                name="task_table",
                df=task_df.iloc[train_ids],
                pkey_col_names={"_task_id"},
                time_col_names=set(),
                backend=cfg.backend,
            )
            test_task_table = Table.create(
                name="task_table",
                df=task_df.iloc[test_ids],
                pkey_col_names={"_task_id"},
                time_col_names=set(),
                backend=cfg.backend,
            )

            if split_seed == seeds[0]:
                # Get baseline score. ##########################################################

                tabicl = TabICLClassifier(
                    n_estimators=cfg.tabicl.n_estimators, random_state=0
                )
                tabicl.fit(data.iloc[train_ids], target_col.iloc[train_ids])
                preds_proba = tabicl.predict_proba(data.iloc[test_ids])
                baseline_metrics = compute_binary_classification_metrics(
                    target_col.iloc[test_ids], preds_proba, print_metrics=False
                )
                print(f"Baseline metrics: {baseline_metrics}")

                results.append(
                    dict(
                        dataset=dataset,
                        reduction_method="BASELINE",
                        reduction_dim=-1,
                        n_intermediate_tables=-1,
                        data_split=False,
                        **baseline_metrics,
                    )
                )

            # Iterate over experiment structure options. ###################################

            for n_inter_tables, data_split in product(
                n_inter_tables_options, data_split_options
            ):
                if n_inter_tables == 0 and data_split:
                    continue

                if not data_split and split_seed != seeds[0]:
                    continue

                print(
                    f"n_inter_tables={n_inter_tables}, data_split={data_split}, seed={split_seed}"
                )

                # Create entity & intermediate dataframes. ##################################

                entity_df = data.assign(_task_id=np.arange(len(data)))
                intermediate_dfs = {
                    n: pd.DataFrame(
                        {
                            "id": entity_df["_task_id"],
                            "_task_id": entity_df["_task_id"],
                        }
                    )
                    for n in range(n_inter_tables)
                }

                # Optionally, split cols randomly between dataframes. ##########################

                if data_split:
                    split_rng = np.random.default_rng(split_seed)
                    shuffled_cols = split_rng.permutation(data.columns)
                    col_groups = np.array_split(shuffled_cols, n_inter_tables + 1)
                    entity_df = data[list(col_groups[0])].assign(
                        _task_id=np.arange(len(data))
                    )
                    for n, cols in enumerate(col_groups[1:]):
                        intermediate_dfs[n] = pd.DataFrame(
                            {
                                **{col: data[col] for col in cols},
                                "_task_id": entity_df["_task_id"],
                            }
                        )

                # Create RelICL tables. ######################################################

                entity_table = Table.create(
                    df=entity_df,
                    name="entity_table",
                    pkey_col_names={"_task_id"},
                    time_col_names=set(),
                    backend=cfg.backend,
                )

                intermediate_dict = {}
                for n in range(n_inter_tables):
                    intermediate_dict[n] = Table.create(
                        df=intermediate_dfs[n],
                        name=f"intermediate_table_{n + 1}",
                        pkey_col_names={"_task_id"},
                        time_col_names=set(),
                        backend=cfg.backend,
                    )

                # Iterate over grid. ###########################################################

                keys = list(grid.keys())
                for values in product(*grid.values()):
                    print(values)
                    params = dict(zip(keys, values))

                    # Config. ##################################################################

                    cfg.output.trace.enable = False
                    cfg.fusion.type = FusionType.early
                    cfg.fusion.early = EarlyFusionConfig(
                        reduction_dim=params["reduction_dim"],
                        reduction_method=params["reduction_method"],
                    )
                    cfg.method.test_batch_size = 100000000
                    cfg.sampling.train.sample_size = 100000000
                    cfg.sampling.context_tables_sample_size = 100000000
                    cfg.sampling.train.restrict_to_test_keys = False

                    # Trace and RNG. ###########################################################

                    with suppressed_output():
                        if rng._rng is None:
                            seed(cfg.method.seed, False)

                    # Create database & schema. ################################################

                    schema_graph = Database()
                    schema_graph.add_table(train_task_table)
                    schema_graph.add_table(entity_table)

                    previous_table = "entity_table"
                    for n_inter_table in range(n_inter_tables):
                        schema_graph.add_table(intermediate_dict[n_inter_table])
                        schema_graph.add_many_to_one(
                            f"intermediate_table_{n_inter_table + 1}",
                            "_task_id",
                            previous_table,
                            "_task_id",
                        )
                        previous_table = f"intermediate_table_{n_inter_table + 1}"
                    schema_graph.add_many_to_one(
                        "task_table", "_task_id", previous_table, "_task_id"
                    )

                    schema_graph.assert_is_valid()

                    # Create prediction task. ##################################################

                    task = PredictionTask.create(
                        target_col_name=target_col_name,
                        id_col_name="_task_id",
                        train_table=train_task_table,
                        test_table=test_task_table,
                        task_type=TaskType.BINARY_CLASSIFICATION,
                        entity_key_col_name="_task_id",
                        db=schema_graph,
                    )

                    # Run RDL model on the prediction task. ####################################

                    with suppressed_output():
                        preds_proba, targets_test, _ = RelICLModel().predict(task)

                    # Logging. #################################################################

                    metrics = compute_binary_classification_metrics(
                        targets_test, preds_proba, print_metrics=False
                    )
                    print(f"Params: {params}")
                    print(f"Metrics: {metrics}")

                    results.append(
                        dict(
                            dataset=dataset,
                            reduction_method=params["reduction_method"],
                            reduction_dim=params["reduction_dim"],
                            n_intermediate_tables=n_inter_tables,
                            data_split=data_split,
                            seed=split_seed,
                            **metrics,
                        )
                    )

                    pd.DataFrame(results).to_csv(
                        osp.join(out_dir, "results.tsv"), sep="\t"
                    )

                    # Reset. ###################################################################

                    Trace.reset()


if __name__ == "__main__":
    main()  # type: ignore
