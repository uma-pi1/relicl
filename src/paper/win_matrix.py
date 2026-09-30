from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd

from paper.config import (
    KIND_COLUMN,
    LOWER_IS_BETTER,
    MISSING_CELL,
    MODEL_HEADER,
    REPORTED_METRICS,
    TASK_HEADER,
    UNDERLINE,
    WIN_MATRIX_CELL,
    WIN_MATRIX_FALLBACK,
    WIN_MATRIX_SPLIT,
    WIN_MATRIX_SUMMARY,
)
from paper.tables import PHANTOM, bold, model_values, value_column

# A model as the matrices index it: its kind and its name.
Model = tuple[str, str]


class Pair(NamedTuple):
    """How the row model fares against the column model over the tasks they share."""

    wins: int
    losses: int
    shared: int
    fallback: bool  # Whether a fallback metric decided any of the tasks.


def win_matrix_table(
    df: pd.DataFrame,
    task_type: str,
    *,
    published: Sequence[str],
    models: Sequence[str],
    split: str = WIN_MATRIX_SPLIT,
    pairs_won: bool = False,
) -> pd.DataFrame:
    """
    Every model against every other one: cell `(i, j)` counts the tasks row model `i`
    beat column model `j` on, out of the tasks the pair shares. `pairs_won` adds a
    summary column of the pairs each row model wins.

    A pair is compared on the first metric of `REPORTED_METRICS` both models report for
    that task, so regression falls back from MAE to $R^2$ where one of the two reports
    only the latter. A cell any fallback decided carries `WIN_MATRIX_FALLBACK`.
    """
    index, pairs = _pairs(df, task_type, published, models, split)
    return _table(index, pairs, pairs_won=pairs_won)


def combined_win_matrix_table(
    df: pd.DataFrame,
    published: Mapping[str, Sequence[str]],
    *,
    models: Sequence[str],
    split: str = WIN_MATRIX_SPLIT,
) -> pd.DataFrame:
    """
    As `win_matrix_table`, but over the tasks of several task types at once: a pair's
    wins and shared tasks add up across the task types in `published`, which maps each
    task type to the baselines that report it. Laid out compactly for the main text.
    """

    # Models in order of first appearance, and each pair summed over the task types.
    index: list[Model] = []
    pairs: dict[tuple[Model, Model], Pair] = {}
    for task_type, published_models in published.items():
        type_index, type_pairs = _pairs(df, task_type, published_models, models, split)
        index += [model for model in type_index if model not in index]
        for key, pair in type_pairs.items():
            pairs[key] = _add(pairs[key], pair) if key in pairs else pair

    return _table(index, pairs, compact=True)


def _pairs(
    df: pd.DataFrame,
    task_type: str,
    published: Sequence[str],
    models: Sequence[str],
    split: str,
) -> tuple[list[Model], dict[tuple[Model, Model], Pair]]:
    """The models of one task type, and every ordered pair of them that shares a task."""

    metrics = REPORTED_METRICS[task_type]

    # One raw value frame per metric, all sharing the model index and the task columns.
    values = {
        metric: model_values(
            df,
            split,
            task_type,
            metric,
            models,
            value_column(metric, task_type),
            published=published,
        )
        for metric in metrics
    }
    index = list(values[metrics[0]].index)

    # Sanity check: the frames line up cell by cell, so a pair can be looked up by
    # position across all of them.
    for frame in values.values():
        assert list(frame.index) == index, "models differ between metrics"
        assert frame.columns.equals(values[metrics[0]].columns), "tasks differ"

    # Stack the metrics into one array of shape (metric, model, task), which puts the
    # fallback order on the first axis.
    scores = np.stack([values[metric].to_numpy(dtype=float) for metric in metrics])
    lower = np.array([LOWER_IS_BETTER[metric] for metric in metrics])

    # Every pair's outcome, leaving out the diagonal and pairs that share no task.
    pairs = {
        (index[row], index[column]): pair
        for row in range(len(index))
        for column in range(len(index))
        if row != column and (pair := _pair(scores, lower, row, column)) is not None
    }
    return index, pairs


def _table(
    index: list[Model],
    pairs: dict[tuple[Model, Model], Pair],
    *,
    compact: bool = False,
    pairs_won: bool = False,
) -> pd.DataFrame:
    """
    Lays the pairs out as a matrix, with a summary column of pairs won where `pairs_won`
    is set. `compact` drops the kind column, which repeats the kinds heading the
    columns, and leaves the model headers upright.
    """

    # Every number is padded to the widest of the matrix (the summary's included).
    width = len(str(max([len(index), *(pair.shared for pair in pairs.values())])))
    cells = [
        [_cell(pairs.get((row, column)), width) for column in index] for row in index
    ]

    # The summary counts the pairs a row model wins, out of the pairs it is compared in.
    summary = []
    for row in index:
        compared = [pairs[row, column] for column in index if (row, column) in pairs]
        summary.append(
            _format(
                wins=sum(pair.wins > pair.losses for pair in compared),
                shared=len(compared),
                width=width,
            )
        )

    table = pd.DataFrame(cells, index=pd.MultiIndex.from_tuples(index))
    table = table.assign(summary=summary).reset_index()

    # Models head the columns under their kinds, rotated and bold as the task headers
    # elsewhere are. The label columns have no kind above them.
    table.columns = pd.MultiIndex.from_tuples(
        [
            ("", bold(KIND_COLUMN)),
            ("", bold(MODEL_HEADER)),
            *(
                (
                    bold(kind),
                    bold(model) if compact else TASK_HEADER.format(task=bold(model)),
                )
                for kind, model in index
            ),
            ("", bold(WIN_MATRIX_SUMMARY)),
        ]
    )
    if compact:
        table = table.drop(columns=("", bold(KIND_COLUMN)))
    if not pairs_won:
        table = table.drop(columns=("", bold(WIN_MATRIX_SUMMARY)))
    return table


def _pair(scores: np.ndarray, lower: np.ndarray, row: int, column: int) -> Pair | None:
    """The row model against the column model, `None` where they share no task."""

    # Per task, the first metric both models report, or -1 where they share none.
    both = ~np.isnan(scores[:, row, :]) & ~np.isnan(scores[:, column, :])
    chosen = np.where(both.any(axis=0), both.argmax(axis=0), -1)

    shared = chosen >= 0
    if not shared.any():
        return None

    # Read the pair's numbers off the metric chosen for each shared task.
    tasks = np.flatnonzero(shared)
    metric = chosen[tasks]
    ours, theirs = scores[metric, row, tasks], scores[metric, column, tasks]

    # A tie counts for neither side, so it only shows up in the denominator.
    return Pair(
        wins=int(np.sum(np.where(lower[metric], ours < theirs, ours > theirs))),
        losses=int(np.sum(np.where(lower[metric], ours > theirs, ours < theirs))),
        shared=len(tasks),
        fallback=bool(metric.any()),
    )


def _add(a: Pair, b: Pair) -> Pair:
    """One pair's outcomes over the tasks of both `a` and `b`."""
    return Pair(
        wins=a.wins + b.wins,
        losses=a.losses + b.losses,
        shared=a.shared + b.shared,
        fallback=a.fallback or b.fallback,
    )


def _cell(pair: Pair | None, width: int) -> str:
    """One cell: the row model's wins, marked by who wins the pair."""

    if pair is None:
        return MISSING_CELL

    # Bold where the row model wins the pair, underlined where the pair draws.
    style: Callable[[str], str] = lambda text: text
    if pair.wins > pair.losses:
        style = bold
    elif pair.wins == pair.losses:
        style = lambda text: UNDERLINE.format(text=text)
    cell = _format(pair.wins, pair.shared, width, style)
    return cell + WIN_MATRIX_FALLBACK if pair.fallback else cell


def _format(
    wins: int, shared: int, width: int, style: Callable[[str], str] = lambda text: text
) -> str:
    """
    One `WIN_MATRIX_CELL`, both numbers padded with invisible digits to `width`, so that
    the slash sits in the middle of its centered column, under its header however wide.
    The padding is styled with the numbers, since bold digits are wider.
    """

    def pad(number: int) -> str:
        missing = width - len(str(number))
        return PHANTOM.format(text="0" * missing) if missing > 0 else ""

    return style(
        WIN_MATRIX_CELL.format(
            wins=pad(wins) + str(wins), shared=str(shared) + pad(shared)
        )
    )
