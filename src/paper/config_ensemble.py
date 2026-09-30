import itertools
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from tqdm import tqdm

from paper.config import (
    BACKBONE_MODELS,
    ENSEMBLE_BASELINE,
    ENSEMBLE_MEMBER_STAGES,
    ENSEMBLE_MEMBERS,
    HEADLINE_METRIC,
    LOWER_IS_BETTER,
    OURS,
    RECIPES_FILE,
    SEED_ENSEMBLE_MODELS,
)
from relicl.config import REPO_ROOT
from relicl.ensemble import Predictions, ensemble_proba
from relicl.utils import (
    compute_binary_classification_metrics,
    compute_regression_metrics,
)

# Finding runs. ########################################################################


def _stage(models: list[str], backbone: str) -> str:
    """The stage of the model in `models` with this backbone."""
    (stage,) = [
        OURS[model]["stage"] for model in models if OURS[model]["backbone"] == backbone
    ]
    return str(stage)


def _tasks(outputs: Path, backbone: str) -> list[Path]:
    """The task dirs of the three-seed reference run."""
    reference = outputs / _stage(SEED_ENSEMBLE_MODELS, backbone)
    return sorted(path for path in reference.glob("*_*") if path.is_dir())


def _member_dir(
    outputs: Path, backbone: str, task_dir: Path, member: str
) -> Path | None:
    """One member's run dir, or None if it did not run."""
    if member == ENSEMBLE_BASELINE:
        candidates = [task_dir / "member_00"]
    else:
        candidates = [
            outputs / pattern.format(task=task_dir.name, member=member)
            for pattern in ENSEMBLE_MEMBER_STAGES[backbone]
        ]
    for candidate in candidates:
        if (candidate / "predictions-val.parquet").is_file():
            return candidate
    return None


def _task_type(preds: Predictions) -> str:
    return "regression" if preds.is_regression else "binary_classification"


# Selecting recipes. ###################################################################


def select(outputs: Path) -> None:
    """Pick the best subset on val per backbone and task type, and write the recipes."""
    recipes = {
        backbone: _select(outputs, backbone) for backbone in ENSEMBLE_MEMBER_STAGES
    }
    RECIPES_FILE.write_text(yaml.safe_dump(recipes), encoding="utf-8")
    print(f"\nwrote {RECIPES_FILE.relative_to(REPO_ROOT)}")


def _select(outputs: Path, backbone: str) -> dict[str, list[str]]:
    """The subset with the best mean relative val gain over the three-seed run."""
    # One row per task, split and subset, holding that subset's gain.
    rows: list[dict[str, Any]] = []
    for task_dir in tqdm(_tasks(outputs, backbone), desc=backbone):
        # Find every member's run, and skip tasks where one is missing.
        found = {
            member: _member_dir(outputs, backbone, task_dir, member)
            for member in [ENSEMBLE_BASELINE, *ENSEMBLE_MEMBERS]
        }
        dirs = {member: path for member, path in found.items() if path is not None}
        missing = [member for member in found if member not in dirs]
        if missing:
            tqdm.write(f"{backbone}/{task_dir.name}: missing {', '.join(missing)}")
            continue

        for split in ("val", "test"):
            # Load the three-seed run and every member's predictions.
            file = f"predictions-{split}.parquet"
            reference = Predictions.load(str(task_dir / file))
            y_true = reference.df["y_true"].to_numpy()
            values: dict[str, npt.NDArray] = {}
            for member, path in dirs.items():
                preds = Predictions.load(str(path / file))
                # Sanity check: members score the same rows in the same order.
                assert (
                    preds.df["key"].to_numpy() == reference.df["key"].to_numpy()
                ).all()
                values[member] = _values(preds)

            # Score every subset relative to the three-seed run, since MAE is not
            # comparable across tasks. Positive means the subset is better.
            task_type = _task_type(reference)
            sign = 1 if LOWER_IS_BETTER[HEADLINE_METRIC[task_type]] else -1
            ref = _score(y_true, [_values(reference)])
            for subset in _subsets():
                ours = _score(y_true, [values[member] for member in subset])
                rows.append(
                    dict(
                        task=task_dir.name,
                        task_type=task_type,
                        subset="+".join(subset),
                        split=split,
                        gain=sign * (ref - ours) / ref,
                    )
                )

    # Per task type, average the gains over tasks and pick the best subset on val.
    recipes: dict[str, list[str]] = {}
    for key, group in pd.DataFrame(rows).groupby("task_type"):
        gains = group.pivot_table(
            index="subset", columns="split", values="gain", aggfunc="mean"
        ).sort_values("val", ascending=False)
        n_tasks = group["task"].nunique()
        print(f"\n{backbone}/{key} ({n_tasks} tasks), gain over three seeds (%):")
        print((gains[["val", "test"]] * 100).round(2).to_string())
        recipes[str(key)] = str(gains.index[0]).split("+")
    return recipes


def _values(preds: Predictions) -> npt.NDArray:
    """Point predictions or class probabilities."""
    if preds.is_regression:
        return preds.df["y_pred"].to_numpy().astype(float)
    return np.stack(list(preds.df["proba"]))


def _score(y_true: npt.NDArray, values: list[npt.NDArray]) -> float:
    """The headline metric of an ensemble, combined like `ensemble_predictions`."""
    if values[0].ndim == 1:
        y_pred = np.mean(values, axis=0)
        return compute_regression_metrics(y_true, y_pred, print_metrics=False)["mae"]
    proba = (
        ensemble_proba(values, average_logits=True) if len(values) > 1 else values[0]
    )
    metrics = compute_binary_classification_metrics(y_true, proba, print_metrics=False)
    return metrics["auroc"]


def _subsets() -> Iterator[tuple[str, ...]]:
    """The baseline plus every subset of the members."""
    for size in range(len(ENSEMBLE_MEMBERS) + 1):
        for combination in itertools.combinations(ENSEMBLE_MEMBERS, size):
            yield (ENSEMBLE_BASELINE, *combination)


# Assembling recipes. ##################################################################


def assemble(outputs: Path) -> None:
    """Rebuild every backbone's config-ensemble stage from the recipes file."""
    # The recipes `paper select` wrote, per backbone and task type.
    recipes = yaml.safe_load(RECIPES_FILE.read_text(encoding="utf-8"))
    for backbone, backbone_recipes in recipes.items():
        # The stage the paper reports for this backbone.
        stage = outputs / _stage(BACKBONE_MODELS, backbone)
        for task_dir in tqdm(_tasks(outputs, backbone), desc=stage.name):
            # Pick the recipe by the task type of the three-seed run.
            reference = Predictions.load(str(task_dir / "predictions-val.parquet"))
            recipe = backbone_recipes[_task_type(reference)]
            # Find every member's run, and skip tasks where one is missing.
            found = [_member_dir(outputs, backbone, task_dir, m) for m in recipe]
            sources = [path for path in found if path is not None]
            if len(sources) < len(recipe):
                tqdm.write(f"{stage.name}/{task_dir.name}: missing members")
                continue

            # Build the task dir, e.g. `final-multiple-configs/rel-hm_item-sales`.
            _assemble_task(stage / task_dir.name, recipe, sources)


def _assemble_task(dest: Path, recipe: list[str], sources: list[Path]) -> None:
    """Copy one task's members and ensemble them."""
    # Start empty, so no stale member is picked up.
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    # Whole dirs, so each member keeps its `.hydra` config.
    member_dirs: list[Path] = []
    for index, (member, source) in enumerate(zip(recipe, sources)):
        copied = dest / f"member_{index:02}-{member}"
        shutil.copytree(source, copied)
        # The copies must agree on `task_name` for `relicl ensemble`.
        _normalize_task_name(copied, dest.name)
        member_dirs.append(copied)

    # Make the task dir a run dir that `paper collect` finds. Both markers come from
    # `Full`, so they describe the baseline config, not the ensemble.
    for marker in (".hydra", "trace.yaml"):
        source = member_dirs[0] / marker
        if source.is_dir():
            shutil.copytree(source, dest / marker)
        else:
            shutil.copy2(source, dest / marker)

    # Ensemble the members on both splits. This writes `predictions-{val,test}.parquet`
    # and `_RESULTS.yaml` into the working directory, i.e., into `dest`.
    subprocess.run(
        [
            "poetry",
            "-P",
            str(REPO_ROOT),
            "run",
            "relicl",
            "ensemble",
            *[str(path.resolve()) for path in member_dirs],
            "--mode",
            "valtest",
            "--overwrite",
        ],
        cwd=dest,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def _normalize_task_name(copied: Path, task: str) -> None:
    """Set `task_name` on a copy, which the reference runs recorded as `cli`, since
    `ensemble_predictions` asserts members agree."""
    for split in ("val", "test"):
        path = copied / f"predictions-{split}.parquet"
        preds = Predictions.load(str(path))
        if preds.task_name != task:
            preds.task_name = task
            preds.save(str(path))
