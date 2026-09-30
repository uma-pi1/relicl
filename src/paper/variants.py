from dataclasses import dataclass

import numpy as np
import pandas as pd

from paper.config import (
    HEADLINE_METRIC,
    OURS_VALUE,
    SPLITS,
    TASK_COLUMN,
    TASK_HEADER,
    TIME_COLUMN,
    TIME_FORMAT,
)
from paper.tables import (
    bold,
    format_summary_row_header,
    format_values,
    our_value,
    task_labels,
)


@dataclass(frozen=True)
class VariantTable:
    stage: str  # the dir
    header: str  # the human-readable version
    variants: dict[str, str]  # directory name to printable label

    # Add a reference row (i.e., a first row to compare values with).
    reference_stage: str = "final"
    reference_label: str | None = None  # human-readable label of the reference row

    # Tasks down instead of across.
    tasks_as_rows: bool = False

    # Rotate the task labels (only when tasks are column headers).
    rotate_tasks: bool = True

    # Add wall clock time at the end.
    time: bool = False

    # The splits the stage ran.
    splits: tuple[str, ...] = SPLITS


VARIANT_TABLES = dict(
    # Fusion ablations.
    fusion=VariantTable(
        stage="fusion",
        header="Fusion",
        variants=dict(none="None", early="Early", late="Late"),
        tasks_as_rows=True,
    ),
    # No. of estimators ablations.
    nestimators=VariantTable(
        stage="n-estimators",
        header="Estimators",
        variants={f"n{n}": str(n) for n in (1, 2, 4, 8, 16, 32)},
        rotate_tasks=False,
        time=True,
        splits=("val",),
    ),
    # Other ablations.
    ablations=VariantTable(
        stage="ablations",
        header="Ablation",
        reference_label="Full",
        variants={
            "no-simplify": "No schema rewrites",
            "no-count-features": "No count features",
            "no-relative-time": "No relative time",
            "no-text": "No text",
            "no-anchors": "No anchors",
            "count-windows": "Count windows",
            "junction-depth-2": "Junction depth 2",
            "future-context": "Future context",
            "reduce-per-estimator": "Reduce per estimator",
            "shortest-path-pooling": "Shortest-path pooling",
        },
    ),
)

# The tables of one stage's variants. ##################################################


def variants_table(
    df: pd.DataFrame,
    split: str,
    task_type: str,
    variant_spec: VariantTable,
    *,
    rank: bool = True,
    wins: bool = True,
) -> pd.DataFrame:
    """One stage's variants against each other on one task type."""
    metric = HEADLINE_METRIC[task_type]
    values = _variant_values(df, variant_spec, split, task_type)

    # Add reference row first.
    if variant_spec.reference_label is not None:
        values = pd.concat(
            [_reference(df, variant_spec, split, task_type, values.columns), values]
        )

    # Format values.
    out_df = format_values(values, metric, compute_ranks=rank, count_wins=wins)

    # Add wall clock time at the end.
    if variant_spec.time:
        out_df[TIME_COLUMN] = _relative_time(df, variant_spec, task_type)

    # Get human-readable task labels.
    labels = task_labels(list(values.columns))

    # Either transpose the DF (variants as columns) or keep as is (tasks as columns).
    if variant_spec.tasks_as_rows:
        # Sanity check: a cost column is one per variant, so it cannot survive the flip.
        assert not variant_spec.time, (
            f"{variant_spec.stage}: tasks_as_rows and time are exclusive"
        )

        out_df.columns = [
            labels.get(c, format_summary_row_header(c)) for c in out_df.columns
        ]
        flipped = out_df.T.reset_index()
        flipped.columns = [bold(TASK_COLUMN), *(bold(v) for v in out_df.index)]

        return flipped

    # Only the task labels head a column here, and only they are worth rotating.
    out_df.columns = [
        TASK_HEADER.format(task=bold(labels[c]))
        if c in labels and variant_spec.rotate_tasks
        else format_summary_row_header(labels.get(c, c))
        for c in out_df.columns
    ]

    return out_df.rename_axis(bold(variant_spec.header)).reset_index()


def _variant_values(
    df: pd.DataFrame, spec: VariantTable, split: str, task_type: str
) -> pd.DataFrame:
    """One stage's variants, one row per variant, and one column per task."""
    runs = df[(df["stage"] == spec.stage) & (df["task_type"] == task_type)]

    # Sanity check: a variant the stage never ran would be a silent row of NA.
    missing = set(spec.variants) - set(runs["variant"])
    assert not missing, f"{spec.stage}: no runs for {sorted(missing)}"

    # The union, so that a variant not run on every task still leaves the others whole.
    tasks = sorted({(run["db"], run["task"]) for run in runs.to_dict("records")})

    return pd.DataFrame(
        {
            f"{db}/{task}": [
                _variant(runs, variant, db, task, split) for variant in spec.variants
            ]
            for db, task in tasks
        },
        index=pd.Index(list(spec.variants.values()), name=spec.header),
    )


def _reference(
    df: pd.DataFrame,
    spec: VariantTable,
    split: str,
    task_type: str,
    tasks: pd.Index,
) -> pd.DataFrame:
    """The row every variant is one config field away from, taken from the base stage."""
    stage = spec.reference_stage
    ours = df[df["task_type"] == task_type]
    cells = [
        our_value(ours, stage, str(task).split("/")[0], str(task).split("/")[1], split)
        for task in tasks
    ]

    return pd.DataFrame(
        [cells],
        columns=tasks,
        index=pd.Index([spec.reference_label], name=spec.header),
    )


def _variant(df: pd.DataFrame, variant: str, db: str, task: str, split: str) -> float:
    """One variant's number for this cell, or NaN where it did not run the task."""
    match = df[(df["variant"] == variant) & (df["db"] == db) & (df["task"] == task)][
        f"{split}_{OURS_VALUE}"
    ]

    # Sanity check: a variant runs a task once, so there is at most one.
    assert len(match) <= 1, f"{variant} {db}/{task} {split}: {len(match)} rows"

    return float(match.iloc[0]) if len(match) else np.nan


def _relative_time(df: pd.DataFrame, spec: VariantTable, task_type: str) -> list[str]:
    """Cost per variant as a multiple of the cheapest, over the tasks in the table."""
    runs = df[(df["stage"] == spec.stage) & (df["task_type"] == task_type)]
    per_task = runs.pivot_table(
        index="variant", columns="task", values="duration_s"
    ).reindex(list(spec.variants))

    # Compute geometric mean since the ratios differ several-fold across tasks and an
    # arithmetic one would follow whichever task is slowest.
    ratios = per_task / per_task.min()
    mean = np.exp(ratios.apply(np.log).mean(axis="columns"))

    return [TIME_FORMAT.format(ratio=ratio) for ratio in mean]
