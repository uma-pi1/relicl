import os.path as osp
import sys

import hydra
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from relbench.base import TaskType
from sklearn.model_selection import train_test_split
from tabicl import TabICLClassifier  # type: ignore[attr-defined]

from paper.config import EARLY_FUSION_RESULTS
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
from relicl.schema import Database, Table
from relicl.trace import Trace
from relicl.utils import compute_binary_classification_metrics

# Reuse the task registry and the output helpers of the sibling `early_fusion_sandbox.py`.
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from early_fusion_sandbox import load_task, suppressed_output  # noqa: E402

cs = ConfigStore.instance()
cs.store(name="base_config", node=RelICLConfig)

DATASETS = [
    "credit-g",
    "phoneme",
    "compass",
    "bioresponse",
    "philippine",
    "kr-vs-kp",
    "qsar-biodeg",
    "spambase",
    "jasmine",
    "churn",
]
ROUND_SEEDS = [0, 1, 2]
N_ESTIMATORS = 1
TABICL_EMBED_DIM = 512  # Fixed by the backbone (`model/relicl/extractor.py:123`).
SETTINGS = [
    ("reduced", DimReductionMethod.gaussian_random_projection, 32),
    ("un-reduced", DimReductionMethod.first_k, N_ESTIMATORS * TABICL_EMBED_DIM),
]


@hydra.main(
    version_base="1.3", config_path=str(REPO_ROOT / "config"), config_name="relicl"
)
def main(cfg: RelICLConfig) -> None:
    """Three repetitions of the two-table early fusion experiment, plus its baseline.

    Each round draws its own split and its own model randomness from one round seed,
    so the spread across rounds is the noise of repeating the whole pipeline.
    """
    RelICLConfig.set(cfg)
    # Written straight to where `poetry run paper` reads it.
    out_path = EARLY_FUSION_RESULTS

    # Fixed config for every run below.
    cfg.output.trace.enable = False
    # Both arms read this: the RelICL arm through the config, the baseline when it
    # constructs its own `TabICLClassifier` below.
    cfg.tabicl.n_estimators = N_ESTIMATORS
    cfg.fusion.type = FusionType.early
    cfg.method.test_batch_size = 100000000
    cfg.sampling.train.sample_size = 100000000
    cfg.sampling.context_tables_sample_size = 100000000
    cfg.sampling.train.restrict_to_test_keys = False
    # This toy database has no time column, so recency weighting cannot apply.
    cfg.sampling.train.recency_half_life_steps = None

    # The global RNG can only be seeded once per process; per-run variation then
    # comes from restarting it on an independent stream.
    if rng._rng is None:
        rng.seed(cfg.method.seed, False)

    results: list[dict] = []

    for dataset in DATASETS:
        data, target_col, target_col_name = load_task(dataset)

        # Shuffle once, before `_task_id` is handed out below.
        order = np.random.RandomState(0).permutation(len(data))
        data = data.iloc[order].reset_index(drop=True)
        target_col = target_col.iloc[order].reset_index(drop=True)

        # Task table carries the target and the key only, entity table the features.
        task_df = target_col.to_frame().assign(_task_id=np.arange(len(target_col)))
        entity_df = pd.concat(
            [data, pd.Series(np.arange(len(data)), index=data.index, name="_task_id")],
            axis=1,
        )

        for round_seed in ROUND_SEEDS:
            # Two independent seeds out of the round seed, rather than the same integer
            # handed to the splitter and to the models.
            split_seed, model_seed = (
                int(s) for s in np.random.SeedSequence(round_seed).generate_state(2)
            )
            print(f"{dataset}, round {round_seed}")

            # A fresh split per round. Every arm below scores the same rows, so the
            # comparison stays paired within a round.
            train_ids, test_ids = train_test_split(
                np.arange(len(data)),
                test_size=0.2,
                random_state=split_seed,
                shuffle=True,
                stratify=target_col,
            )

            # Baseline: the backbone on the unsplit table.
            tabicl = TabICLClassifier(
                n_estimators=cfg.tabicl.n_estimators, random_state=model_seed
            )
            with suppressed_output():
                tabicl.fit(data.iloc[train_ids], target_col.iloc[train_ids])
                preds_proba = tabicl.predict_proba(data.iloc[test_ids])
            baseline_metrics = compute_binary_classification_metrics(
                target_col.iloc[test_ids], preds_proba, print_metrics=False
            )
            print(f"  baseline auroc: {baseline_metrics['auroc']:.4f}")
            results.append(
                dict(
                    dataset=dataset,
                    model="BASELINE",
                    setting="-",
                    reduction_method="-",
                    reduction_dim=-1,
                    round=round_seed,
                    **baseline_metrics,
                )
            )

            # RelICL on the two-table database, once per reduction setting. The
            # database is rebuilt per setting so that no state of the previous run
            # (materialized frames, fitted reducers) can leak into the next one.
            for setting, reduction_method, reduction_dim in SETTINGS:
                cfg.fusion.early = EarlyFusionConfig(
                    reduction_dim=reduction_dim,
                    reduction_method=reduction_method,
                )
                rng.set_stream(model_seed)

                entity_table = Table.create(
                    df=entity_df,
                    name="entity_table",
                    pkey_col_names={"_task_id"},
                    time_col_names=set(),
                    backend=cfg.backend,
                )
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

                schema_graph = Database()
                schema_graph.add_table(train_task_table)
                schema_graph.add_table(entity_table)
                schema_graph.add_many_to_one(
                    "task_table", "_task_id", "entity_table", "_task_id"
                )
                schema_graph.assert_is_valid()

                task = PredictionTask.create(
                    target_col_name=target_col_name,
                    id_col_name="_task_id",
                    train_table=train_task_table,
                    test_table=test_task_table,
                    task_type=TaskType.BINARY_CLASSIFICATION,
                    entity_key_col_name="_task_id",
                    db=schema_graph,
                )

                with suppressed_output():
                    preds_proba, targets_test, _ = RelICLModel().predict(task)
                metrics = compute_binary_classification_metrics(
                    targets_test, preds_proba, print_metrics=False
                )
                print(f"  relicl {setting} auroc: {metrics['auroc']:.4f}")
                results.append(
                    dict(
                        dataset=dataset,
                        model="RelICL",
                        setting=setting,
                        reduction_method=reduction_method.value,
                        reduction_dim=reduction_dim,
                        round=round_seed,
                        **metrics,
                    )
                )

                Trace.reset()

                # Write after every run so a crash keeps what already finished.
                pd.DataFrame(results).to_csv(out_path, sep="\t")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()  # type: ignore
