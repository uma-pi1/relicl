import logging
import os
import os.path as osp
import subprocess
from argparse import ArgumentParser
from typing import Any

import optuna
import pandas as pd
import yaml
from omegaconf import OmegaConf
from optuna import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial, TrialState

from hpo.utils_hpo import (
    RelICLSubprocessError,
    _get_results,
    resolve_hp_metric,
    top_k_test_csv,
)
from relicl.config import RESULTS_FILENAME

logger = logging.getLogger(__name__)

HP_METRIC_ATTR = "hp_metric"


# Trial selection. #####################################################################


def select_top_k(study: Study, k: int) -> list[FrozenTrial]:
    """The k best completed trials, best first."""

    completed = [
        t
        for t in study.trials
        if t.state is TrialState.COMPLETE and t.value is not None
    ]
    reverse = study.direction is StudyDirection.MAXIMIZE
    return sorted(completed, key=lambda t: t.value, reverse=reverse)[:k]  # type: ignore[arg-type,return-value]


# Replaying a trial on the test split. #################################################


def _trial_dir(run_dir: str, trial: FrozenTrial) -> str:
    return osp.join(run_dir, f"trial-{trial.number}")


def _replay_overrides(trial_dir: str) -> list[str]:
    """The trial's own Hydra overrides, switched from the val to the test split."""

    with open(osp.join(trial_dir, ".hydra/overrides.yaml"), "r") as f:
        overrides: list[str] = yaml.safe_load(f)

    # Everything else is replayed verbatim, including the seeds the study sampled for
    # this trial: they are drawn from one sequential stream in `Objective`, so the
    # trial's own overrides are the only place they survive.
    kept = [o for o in overrides if not o.startswith(("run.val=", "run.test="))]
    return kept + ["run.val=False", "run.test=True"]


def _run_test(run_dir: str, trial: FrozenTrial) -> dict[str, float]:
    """Run the test split for one trial and return its metrics."""

    out_path = _trial_dir(run_dir, trial) + "-test"

    # Reuse a completed replay. A test run costs as much as a trial, and this pass is
    # repeated whenever a sweep is interrupted after some of its trials were tested.
    if osp.exists(osp.join(out_path, RESULTS_FILENAME)):
        metrics = _get_results(out_path)["test"]
        if metrics:
            logger.info("Reusing the test run in %s.", out_path)
            return metrics

    overrides = _replay_overrides(_trial_dir(run_dir, trial))
    overrides.append(f"hydra.run.dir={out_path}")

    # Reproducibility. `method.use_max_reproducibility` is already among the replayed
    # overrides, these two env vars are what the sweep set around it.
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    result = subprocess.run(
        ["poetry", "run", "relicl", "run"] + overrides,
        capture_output=True,
        text=True,
    )

    # Keep the subprocess output next to the run, as the sweep does for its trials.
    os.makedirs(out_path, exist_ok=True)
    with open(osp.join(out_path, "trial.log"), "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n=== stderr ===\n")
            f.write(result.stderr)

    if result.returncode != 0:
        logger.error(
            "RelICL failed\nstdout:\n%s\n\nstderr:\n%s", result.stdout, result.stderr
        )
        raise RelICLSubprocessError

    metrics = _get_results(out_path)["test"]

    # Sanity check: the replay ran with `run.test=True`, so metrics must be present.
    assert metrics, f"No test metrics in {out_path}"
    return metrics


# Metric to report. ####################################################################


def _resolve_hp_metric(study: Study, run_dir: str, trial: FrozenTrial) -> str:
    """The study's optimized metric, from the study or from a trial's overrides."""

    hp_metric = study.user_attrs.get(HP_METRIC_ATTR)
    if hp_metric:
        return str(hp_metric)

    # Studies from before the metric was recorded: recover the task from the overrides
    # the trial ran with, and take that task type's default metric.
    with open(osp.join(_trial_dir(run_dir, trial), ".hydra/overrides.yaml"), "r") as f:
        overrides: list[str] = yaml.safe_load(f)
    values = dict(o.split("=", 1) for o in overrides if "=" in o)
    relbench_cfg = OmegaConf.create(
        dict(db=values["relbench.db"], task=values["relbench.task"], hp_metric=None)
    )
    return resolve_hp_metric(relbench_cfg)


# Entry point. #########################################################################


def run_top_k_test(path_to_db: str, k: int) -> None:
    """Run the test split for the k best trials and record their metrics."""

    if k <= 0:
        return

    study = optuna.load_study(study_name=None, storage=f"sqlite:///{path_to_db}")

    run_dir = osp.dirname(osp.abspath(path_to_db))

    trials = select_top_k(study, k)
    if not trials:
        logger.warning("No completed trials in %s; nothing to test.", path_to_db)
        return

    hp_metric = _resolve_hp_metric(study, run_dir, trials[0])

    rows: list[dict[str, Any]] = []
    for rank, trial in enumerate(trials):
        # Trials of a sweep that already ran the test split need no second run.
        if "test_value" in trial.user_attrs:
            logger.info("Trial %d already has test metrics; skipping.", trial.number)
            metrics = {
                name[len("test_") :]: value
                for name, value in trial.user_attrs.items()
                if name.startswith("test_") and name != "test_value"
            }
        else:
            logger.info(
                "Testing trial %d (rank %d, val %s=%s).",
                trial.number,
                rank,
                hp_metric,
                trial.value,
            )
            metrics = _run_test(run_dir, trial)

        rows.append(
            dict(rank=rank, number=trial.number, value=trial.value)
            | {"test_value": metrics[hp_metric]}
            | {f"test_{name}": value for name, value in metrics.items()}
            | {f"params_{name}": value for name, value in trial.params.items()}
        )

    # The CSV is where these metrics live: optuna refuses to annotate a finished trial,
    # so `trials_dataframe` joins this file back into the study's trials instead.
    out_file = top_k_test_csv(path_to_db, study.study_name)
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_file, index=False)
    print(f"Wrote {out_file}")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--path", "-p", type=str, required=True, help="path to optuna's sqlite DB"
    )
    parser.add_argument(
        "--top_k",
        "-k",
        type=int,
        default=1,
        help="run the test split for the TOP_K best trials (default: 1)",
    )
    args = parser.parse_args()
    run_top_k_test(args.path, args.top_k)
