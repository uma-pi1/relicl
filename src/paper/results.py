import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import yaml

from paper.config import (
    CONFIG_PREFIX,
    DURATION_RE,
    HEADLINE_COLUMNS,
    HEADLINE_METRIC,
    NON_SCALAR_METRICS,
    SPLITS,
    START_EVENT_LINE,
)
from relicl.config import REPO_ROOT, RESULTS_FILENAME

logger = logging.getLogger(__name__)

# Collecting the whole run set. ########################################################


def collect(root: Path) -> pd.DataFrame:
    """One row per run, with each split's scalar metrics under a `val_`/`test_` prefix."""
    rows: list[dict[str, Any]] = []
    for stage, db, task, variant, run_dir in _find_runs(root):
        # A run directory without results is a job that died, not one never submitted.
        if not (run_dir / RESULTS_FILENAME).is_file():
            logger.warning(f"no results: {run_dir.relative_to(REPO_ROOT)}")
            continue

        # Create base columns.
        run = dict(stage=stage, db=db, task=task, variant=variant)

        # Create metadata columns.
        run |= read_run_metadata(run_dir)

        # Create one metric column per split and metric.
        run |= prefix_splits(read_split_metrics(run_dir))

        rows.append(run)

    return pd.DataFrame(rows)


# Find run directories. ################################################################


def _find_runs(root: Path) -> Iterator[tuple[str, str, str, str, Path]]:
    """Yield `(stage, db, task, variant, run_dir)` for every run directory under `root`.

    Definitions:

    - stage: A child directory of `outputs`. Contains `db_task` directories.
    - variant: A child directory of a `db_task` directory inside a stage.
    - run_dir: The directory which contains the `.hydra` directory (but not member dirs).
    """
    for stage_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        # Stage dirs are the children of the root.
        stage = stage_dir.name

        # Each stage dir contains a set of task dirs.
        for task_dir in sorted(stage_dir.glob("*_*")):
            # Split into db and task, drop the underscore in the middle.
            db, _, task = task_dir.name.partition("_")

            # Variant dirs are optional sub dirs of a task dir that contain a
            # .hydra dir.
            variant_dirs = [
                path
                for path in sorted(task_dir.glob("*"))
                if _is_run_dir(path) and not path.name.startswith("member_")
            ]
            if variant_dirs:
                yield from ((stage, db, task, d.name, d) for d in variant_dirs)
            else:
                yield stage, db, task, "", task_dir


def _is_run_dir(path: Path) -> bool:
    """A run dir is a dir with a `.hydra` sub dir, excluding ensemble members."""
    return (path / ".hydra").is_dir()


# Read one run directory. ##############################################################


def read_run_metadata(run_dir: Path) -> dict[str, Any]:
    """Extract run metadata from the `start` event from the trace."""
    start = _read_start_event(run_dir)

    return dict(
        # Include flattened config.
        **_flatten(start.get("config", dict()), CONFIG_PREFIX),
        # Remaining attrs.
        n_members=len(list(run_dir.glob("member_*"))),
        duration_s=_read_duration_s(run_dir),
        revision=start.get("revision", ""),
        run_dir=str(run_dir.relative_to(REPO_ROOT)),
    )


def _read_start_event(run_dir: Path) -> dict[str, Any]:
    """Retrieve start event (contains config and git revision)."""
    trace = run_dir / "trace.yaml"
    if not trace.is_file():
        return dict()
    with trace.open("r", encoding="utf-8") as f:
        event = yaml.safe_load(f.readlines()[START_EVENT_LINE])
    return event if isinstance(event, dict) else dict()


def _flatten(config: dict[str, Any], prefix: str) -> dict[str, str]:
    """One entry per config leaf, keyed by its dot path. A disabled section has none."""
    flat: dict[str, str] = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flat |= _flatten(value, f"{prefix}{key}.")
        else:
            flat[f"{prefix}{key}"] = str(value)
    return flat


def _read_duration_s(run_dir: Path) -> float:
    """Wall clock time of the run, summed over its ensemble members."""
    traces = sorted(run_dir.glob("member_*/trace.yaml")) or [run_dir / "trace.yaml"]
    total = 0.0
    for trace in traces:
        if not trace.is_file():
            continue
        for line in trace.open("r", encoding="utf-8"):
            match = DURATION_RE.search(line)
            if match:
                total += float(match.group(1))
    return total


def read_split_metrics(run_dir: Path) -> dict[str, dict[str, Any]]:
    """The scalar metrics of `_RESULTS.yaml`, per split. An unrun split is left out."""
    results = yaml.safe_load((run_dir / RESULTS_FILENAME).read_text()) or dict()
    return {
        # An unrun split is written as an empty mapping rather than omitted.
        split: {
            name: value
            for name, value in metrics.items()
            if name not in NON_SCALAR_METRICS
        }
        for split, metrics in results.items()
        if metrics
    }


def prefix_splits(per_split: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Flatten `{split: {metric: value}}` into one mapping of `<split>_<metric>`."""
    return {
        f"{split}_{name}": value
        for split, metrics in per_split.items()
        for name, value in metrics.items()
    }


# Extract important metrics. ###########################################################


def add_headline_metric(df: pd.DataFrame) -> pd.DataFrame:
    """Add the task type and the one metric per task that the paper reports, per split."""

    # A run without ensembling reports no member spread, and a validation-only stage no
    # test metrics at all, so those columns can be absent rather than merely null.
    for split in SPLITS:
        for column in HEADLINE_COLUMNS:
            if f"{split}_{column}" not in df:
                df[f"{split}_{column}"] = np.nan

    # Determine the task type through available metrics. Every stage runs validation,
    # so it is the split that always has them.
    df["task_type"] = np.where(
        df["val_mae"].notna(), "regression", "binary_classification"
    )

    # Determine headline metric.
    df["metric"] = df["task_type"].map(HEADLINE_METRIC)

    # Fill value and variance measures. `value` scores the ensembled predictions, which
    # is what the paper reports; `seed_mean` averages the members' own scores, which is
    # what a single seed is expected to reach. The two differ by the ensembling gain.
    is_regression = df["task_type"] == "regression"
    for split in SPLITS:
        df[f"{split}_value"] = np.where(
            is_regression, df[f"{split}_mae"], df[f"{split}_auroc"]
        )
        df[f"{split}_seed_mean"] = np.where(
            is_regression,
            df[f"{split}_mae_member_mean"],
            df[f"{split}_auroc_member_mean"],
        )
        df[f"{split}_seed_std"] = np.where(
            is_regression,
            df[f"{split}_mae_member_std"],
            df[f"{split}_auroc_member_std"],
        )
        df[f"{split}_seed_sem"] = np.where(
            is_regression,
            df[f"{split}_mae_member_sem"],
            df[f"{split}_auroc_member_sem"],
        )

    # Return result.
    return df
