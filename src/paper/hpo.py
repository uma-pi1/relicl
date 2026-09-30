import logging
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from paper.config import HPO_STAGE, SORTED_RESULTS_GLOB, TEST_DIR_SUFFIX
from paper.results import prefix_splits, read_run_metadata, read_split_metrics
from relicl.config import REPO_ROOT, RESULTS_FILENAME

logger = logging.getLogger(__name__)

# Collecting the best trial of every study. ############################################


def collect(root: Path) -> pd.DataFrame:
    """One row per study, for the trial that won its study on validation."""
    rows: list[dict[str, Any]] = []
    for db, task, trial_dir in _find_best_trials(root):
        # Create base and metadata columns from the trial itself.
        run = dict(stage=HPO_STAGE, db=db, task=task, variant=trial_dir.name)
        run |= read_run_metadata(trial_dir)
        run |= prefix_splits(read_split_metrics(trial_dir))

        # Add the test columns from the sibling directory `test-top-k` wrote.
        test_dir = trial_dir.with_name(trial_dir.name + TEST_DIR_SUFFIX)
        if (test_dir / RESULTS_FILENAME).is_file():
            run |= prefix_splits(read_split_metrics(test_dir))
        else:
            logger.warning(f"no results: {test_dir.relative_to(REPO_ROOT)}")

        rows.append(run)

    return pd.DataFrame(rows)


# Helpers. #############################################################################


def _find_best_trials(root: Path) -> Iterator[tuple[str, str, Path]]:
    """Yield `(db, task, trial_dir)` for the winning trial of every study under `root`."""
    for sorted_file in sorted(root.glob(SORTED_RESULTS_GLOB)):
        study_dir = sorted_file.parent.parent
        db, _, task = study_dir.name.partition("_")

        # `analysis` writes the file already sorted best first, but a failed trial has
        # no value at all and sorts wherever pandas puts NaN.
        trials = pd.read_csv(sorted_file)
        complete = trials[(trials["state"] == "COMPLETE") & trials["value"].notna()]
        if complete.empty:
            continue

        yield db, task, study_dir / f"trial-{int(complete.iloc[0]['number'])}"
