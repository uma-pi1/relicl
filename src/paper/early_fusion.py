from collections.abc import Mapping

import numpy as np
import pandas as pd

from paper.config import (
    EARLY_FUSION_BASELINE,
    EARLY_FUSION_BASELINE_COLUMN,
    EARLY_FUSION_DATASET_COLUMNS,
    EARLY_FUSION_DATASET_SPEC,
    EARLY_FUSION_DATASETS,
    EARLY_FUSION_DELTA_COLUMN,
    EARLY_FUSION_MAIN_SETTINGS,
    EARLY_FUSION_METRIC,
    EARLY_FUSION_METRIC_COLUMN,
    EARLY_FUSION_RESULTS,
    EARLY_FUSION_ROUND,
    EARLY_FUSION_SETTINGS,
    EARLY_FUSION_SPREAD_FORMAT,
    EARLY_FUSION_SUMMARY,
    GENERATED_DIR,
    PERCENT_DECIMALS,
)
from paper.dataset_stats import write as write_tabular
from paper.tables import bold, format_summary_row_header, write

# The results table. ###################################################################


def results_table(runs: pd.DataFrame, settings: Mapping[str, str]) -> pd.DataFrame:
    """The baseline against each row-embedding setting in `settings`, one row per
    dataset.

    Rounds are the unit of repetition: every arm of a round scores the same rows, so
    the reported difference is paired and its spread is that of the pairs, not the
    quadrature of two absolute spreads.
    """
    scores = runs.pivot_table(
        index=["dataset", EARLY_FUSION_ROUND],
        columns="setting",
        values=EARLY_FUSION_METRIC,
    )

    # The baseline carries no setting of its own, so it arrives under a placeholder.
    baseline = _baseline_column(runs)
    scores = scores.rename(columns={baseline: EARLY_FUSION_BASELINE_COLUMN})

    # The dataset heads no group, so its top-level header is empty. The baseline heads
    # its own score, as the settings do, so that all three read along one header row.
    columns: list[tuple[str, str]] = [
        ("", bold(EARLY_FUSION_DATASET_COLUMNS["dataset"])),
        (bold(EARLY_FUSION_BASELINE_COLUMN), bold(EARLY_FUSION_METRIC_COLUMN)),
    ]
    cells: list[list[str]] = [
        _labels(scores),
        _cells(scores[EARLY_FUSION_BASELINE_COLUMN]),
    ]

    # One pair of columns per setting: its own score, then its paired regret.
    for setting, label in settings.items():
        deltas = scores[setting] - scores[EARLY_FUSION_BASELINE_COLUMN]
        columns += [
            (bold(label), bold(EARLY_FUSION_METRIC_COLUMN)),
            (bold(label), bold(EARLY_FUSION_DELTA_COLUMN)),
        ]
        cells += [_cells(scores[setting]), _cells(deltas, signed=True)]

    return pd.DataFrame(dict(zip(columns, cells))).set_axis(
        pd.MultiIndex.from_tuples(columns), axis="columns"
    )


def _baseline_column(runs: pd.DataFrame) -> str:
    """The placeholder `setting` the baseline runs are stored under."""
    settings = set(runs.loc[runs["model"] == EARLY_FUSION_BASELINE, "setting"])

    # Sanity check: the baseline is one arm, so it occupies exactly one setting.
    assert len(settings) == 1, f"baseline settings: {settings}"

    return settings.pop()


def _labels(scores: pd.DataFrame) -> list[str]:
    """One label per dataset, in the order the table reports them, then the summary."""
    datasets = sorted(set(scores.index.get_level_values("dataset")))
    return [*datasets, format_summary_row_header(EARLY_FUSION_SUMMARY)]


def _cells(scores: pd.Series, *, signed: bool = False) -> list[str]:
    """One cell per dataset, then one summarizing all of them.

    A dataset cell averages over the rounds of that dataset. The summary cell averages
    those per-dataset means, so its unit is the dataset and its spread is the scatter
    across datasets, which answers a different question than the cells above it.
    """
    by_dataset = scores.groupby("dataset")
    mean, sem = by_dataset.mean(), by_dataset.sem()

    cells = [
        _cell(mean[dataset], sem[dataset], signed=signed)
        for dataset in sorted(mean.index)
    ]

    # The summary treats each dataset as one observation, hence the spread over the ten
    # per-dataset means rather than over the thirty rounds behind them.
    summary = mean.to_numpy(dtype=float)
    spread = summary.std(ddof=1) / np.sqrt(len(summary))
    return [*cells, _cell(float(summary.mean()), float(spread), signed=signed)]


def _cell(value: float, spread: float, *, signed: bool) -> str:
    """One `mean $\\pm$ spread` cell, both numbers in AUROC points."""
    return EARLY_FUSION_SPREAD_FORMAT.format(
        value=_points(value, signed=signed), spread=_points(spread)
    )


def _points(value: float, *, signed: bool = False) -> str:
    """AUROC in points, at the precision the other percentage tables use."""
    # Sign after rounding, so a difference that rounds to zero reads "0.0", not "-0.0".
    points = round(value * 100, PERCENT_DECIMALS) + 0.0
    sign = "+" if signed and points > 0 else ""
    return f"{sign}{points:.{PERCENT_DECIMALS}f}"


# The dataset-statistics table. ########################################################


def datasets_table(stats: pd.DataFrame) -> pd.DataFrame:
    """One row per OpenML table, counting the original table before any split."""
    stats = stats.sort_values("dataset")
    columns: dict[str, list] = {
        bold(label): stats[column].tolist()
        for column, label in EARLY_FUSION_DATASET_COLUMNS.items()
        if column != "pos_rate"
    }
    columns[bold(EARLY_FUSION_DATASET_COLUMNS["pos_rate"])] = [
        f"{rate * 100:.{PERCENT_DECIMALS}f}" for rate in stats["pos_rate"]
    ]
    return pd.DataFrame(columns)


# Writing. #############################################################################


def write_all() -> None:
    """Build both tables of `app:early-fusion` from the checked-in run outputs."""
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(EARLY_FUSION_RESULTS, sep="\t", index_col=0)
    # Neither table has splits, so "test" only selects an unsuffixed file name.
    main = {s: EARLY_FUSION_SETTINGS[s] for s in EARLY_FUSION_MAIN_SETTINGS}
    for name, settings in [
        ("early-fusion", main),
        ("early-fusion-full", EARLY_FUSION_SETTINGS),
    ]:
        write(results_table(runs, settings), name, "test")

    stats = pd.read_csv(EARLY_FUSION_DATASETS)
    write_tabular(
        datasets_table(stats), "early-fusion-datasets", EARLY_FUSION_DATASET_SPEC
    )
