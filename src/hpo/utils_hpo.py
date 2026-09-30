from os import path as osp
from typing import Any

import pandas as pd
import yaml
from optuna import Study, Trial
from relbench.base import TaskType
from relbench.tasks import get_task

from relicl.config import RESULTS_FILENAME
from relicl.effective_config import EFFECTIVE_FILENAME

# Get return values from RelICL. #######################################################


def _get_results(out_path: str) -> dict[str, dict[str, float]]:
    with open(osp.join(out_path, RESULTS_FILENAME), "r") as f:
        results: dict[str, dict[str, float]] = yaml.safe_load(f)
    return results


def _get_effective(out_path: str) -> dict[str, Any]:
    """Departures from the requested config; empty when the run had none."""
    path = osp.join(out_path, EFFECTIVE_FILENAME)
    if not osp.exists(path):
        return {}
    with open(path, "r") as f:
        facts = yaml.safe_load(f) or {}
    # A list does not survive into a flat CSV column, so join names into one string.
    return {k: ",".join(v) if isinstance(v, list) else v for k, v in facts.items()}


# Top-k test metrics. ##################################################################

TEST_TOP_K_FILENAME = "top-k-test.csv"


def top_k_test_csv(path_to_db: str, study_name: str) -> str:
    """Where `test-top-k` writes the test metrics of a study's best trials."""

    run_dir = osp.dirname(osp.abspath(path_to_db))
    return osp.join(run_dir, "_ANALYSIS", f"{study_name}-{TEST_TOP_K_FILENAME}")


# Trials data frame. ###################################################################


def trials_dataframe(study: Study, path_to_db: str | None = None) -> pd.DataFrame:
    """Optuna's trials, with `duration` in seconds and the top-k test metrics joined in.

    Test metrics are joined from `test-top-k`'s CSV when `path_to_db` locates it.
    """

    df: pd.DataFrame = study.trials_dataframe()

    # Optuna reports `duration` as a timedelta, which is awkward to use from a CSV.
    # Trials which are never completed have no duration and become NaN.
    if "duration" in df.columns:
        duration = pd.to_timedelta(df["duration"], errors="coerce")
        position = list(df.columns).index("duration") + 1
        df.insert(position, "duration_s", duration.dt.total_seconds())

    # Test metrics live outside the study; they are joined back in here. Only
    # the tested trials have them, every other trial gets NaN.
    if path_to_db is not None:
        csv_file = top_k_test_csv(path_to_db, study.study_name)
        if osp.exists(csv_file):
            test_df = pd.read_csv(csv_file)
            columns = [c for c in test_df.columns if c.startswith("test_")]
            df = df.merge(test_df[["number"] + columns], on="number", how="left")

    return df


# Write intermediate results callback. #################################################


class WriteResultsCallback:
    def __init__(self, file_path: str) -> None:
        self._file_path = file_path

    def __call__(self, study: Study, trial: Trial) -> None:
        trials_dataframe(study).to_csv(self._file_path)


# Raise exception when RelICL subprocess fails. ########################################


class RelICLSubprocessError(Exception):
    """Exception raised for a failed RelICL subprocess."""


# Task type. ###########################################################################


def get_task_type(db: str, task: str) -> TaskType:
    return get_task(db, task, download=False).task_type


def resolve_hp_metric(relbench_cfg: Any) -> str:
    if relbench_cfg.get("hp_metric"):
        return str(relbench_cfg.hp_metric)

    task_type = get_task_type(relbench_cfg.db, relbench_cfg.task)
    match task_type:
        case TaskType.BINARY_CLASSIFICATION:
            return "auroc"
        case TaskType.REGRESSION:
            return "mae"
        case _:
            raise ValueError(f"Unsupported task type: {task_type}")


# Get optimization direction from metric. ##############################################


def get_direction(hp_metric: str) -> str:
    # Names must match the metric keys produced by compute_binary_classification_metrics
    # / compute_regression_metrics (see relicl.utils).
    if hp_metric in [
        "auroc",
        "ap",
        "f1",
        "precision",
        "recall",
        "accuracy",
        "r2",
        "explained_variance",
        "spearman",
    ]:
        return "maximize"
    if hp_metric in ["mae", "mse", "rmse", "mape", "median_ae", "max_error"]:
        return "minimize"
    raise ValueError(f"Unknown metric: {hp_metric}; cannot determine direction.")


def _format_value(value: Any) -> str:
    if isinstance(value, dict):
        inner = ",".join(f"{k}:{_format_value(v)}" for k, v in value.items())
        return "{" + inner + "}"
    if isinstance(value, list):
        return "[" + ",".join(_format_value(v) for v in value) + "]"
    # Ensemble members are themselves override strings, so they contain `=`, which
    # Hydra's override grammar rejects unquoted. Only those are quoted: a config group
    # selection such as `fusion=late` has to stay bare.
    if isinstance(value, str) and "=" in value:
        return f'"{value}"'
    return str(value)


def _format_override(key: str, value: Any) -> str:
    return f"{key}={_format_value(value)}"
