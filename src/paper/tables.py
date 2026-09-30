import logging
import re
from collections.abc import Sequence
from functools import reduce
from typing import Any

import numpy as np
import pandas as pd

from paper import baselines
from paper.config import (
    BOLD,
    CELL_SEPARATOR,
    DB_LABEL_CHARS,
    DB_LABEL_FORMAT,
    DB_LABEL_PREFIX,
    GENERATED_DIR,
    GROUP_RULE,
    HEADLINE_METRIC,
    HPO_BEST_LABEL,
    HPO_GAIN_COLUMN,
    HPO_REFERENCE_STAGE,
    HPO_SPLIT,
    HPO_STAGE,
    KIND_COLUMN,
    KIND_LABEL,
    LABEL_SPEC,
    LOWER_IS_BETTER,
    MAIN_MODEL,
    MID_RULE,
    MISSING_CELL,
    MODEL_HEADER,
    MODEL_LABEL,
    MODEL_ORDER,
    NUMBER_SPEC,
    OURS,
    OURS_LABEL,
    OURS_VALUE,
    PAIRED_SPEC,
    PERCENT_DECIMALS,
    PERCENT_METRICS,
    RANK_COLUMN,
    RANK_DECIMALS,
    REPORTED_METRICS,
    SEED_SPREAD_COLUMNS,
    SEED_SPREAD_OF,
    SEED_VALUE_COLUMNS,
    SPREAD_FORMAT,
    SUMMARY_COLUMNS,
    SUMMARY_HEADERS,
    SUMMARY_RULE,
    TABLE_PREAMBLE,
    TASK_COLUMN,
    TASK_HEADER,
    TOP_RULE,
    VALUE_DECIMALS,
    WINS_COLUMN,
)
from relicl.config import REPO_ROOT

logger = logging.getLogger(__name__)

# A number cell, bold, underlined or plain: the digits start right away, after an
# optional sign or invisible padding digits.
NUMBER_PATTERN = re.compile(r"(\\textbf\{|\\underline\{)?(\\phantom\{0+\})?[-+]?\d")

# A header cell spanning several columns, e.g. "\multicolumn{2}{c}{\textbf{Trained}}".
SPAN_PATTERN = re.compile(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}\s*$")
SPAN_FORMAT = "\\multicolumn{{{width}}}{{c}}{{{text}}}"

# A bold cell, and an invisible stand-in of the width of its text.
BOLD_PATTERN = re.compile(r"\\textbf\{(.*)\}")
PHANTOM = "\\phantom{{{text}}}"

# Separates the cells of a table row. An escaped ampersand is text inside a cell.
UNESCAPED_AMPERSAND = re.compile(r"(?<!\\)&")

# The main results table. ##############################################################


def final_table(
    df: pd.DataFrame,
    split: str,
    task_type: str,
    *,
    models_as_rows: bool = False,
    rank: bool = True,
    wins: bool = True,
    models: Sequence[str] = (MAIN_MODEL,),
    metrics: Sequence[str] | None = None,
    published: Sequence[str] = MODEL_ORDER,
) -> pd.DataFrame:
    """
    RelICL against the published baselines on one task type. Pass no `published` models
    to compare the reported models against each other alone, as the backbone table
    does.

    A task type may report several metrics (regression reports MAE and $R^2$). They
    then share one table, every cell carrying them in `REPORTED_METRICS` order and
    separated by `CELL_SEPARATOR`, the summary rows included. Each metric is compared
    on its own, so bold cells, wins and ranks all refer to the metric beside them.
    """

    # Report the metrics of the task type by default.
    metrics = metrics or REPORTED_METRICS[task_type]

    # Format one table per metric, then merge them cell by cell.
    formatted = [
        format_values(
            model_values(
                df,
                split,
                task_type,
                metric,
                models,
                value_column(metric, task_type),
                published=published,
            ),
            metric,
            compute_ranks=rank,
            count_wins=wins,
        )
        for metric in metrics
    ]
    out_df = _merge_metrics(formatted)

    # Get human-readable strings for the tasks.
    labels = task_labels([c for c in out_df.columns if c not in SUMMARY_COLUMNS])

    if models_as_rows:
        # Keep models as rows.
        #
        # In this case, the tasks need to be rotated (for space) and made bold (for
        # readability).

        out_df.columns = [
            format_summary_row_header(c)  # Make bold.
            if c in SUMMARY_COLUMNS
            else TASK_HEADER.format(task=bold(labels[c]))  # Rotate and make bold.
            for c in out_df.columns
        ]
        return out_df.rename_axis([bold(KIND_COLUMN), bold(MODEL_HEADER)]).reset_index()

    # Transpose the DF so that each model has its own column.

    # Format summary row headers (make bold and add arrow).
    out_df.columns = [
        format_summary_row_header(c) if c in SUMMARY_COLUMNS else labels[c]
        for c in out_df.columns
    ]

    # Transpose the DF.
    flipped = out_df.T.reset_index()

    # Create new index of like that:
    # ------- | -------- | ------- | --------- | ...
    # (empty) | Kind I   | ...     | Kind II   | ...
    # Task    | Model Ia | ...     | Model IIa | ...
    # ------- | -------- | ------- | --------- | ...
    #
    # Then make those column headers.
    flipped.columns = pd.MultiIndex.from_tuples(
        [("", bold(TASK_COLUMN)), *((bold(k), bold(m)) for k, m in out_df.index)]
    )
    return flipped


def value_column(metric: str, task_type: str) -> str:
    """Which of our own result columns holds this metric."""
    return OURS_VALUE if metric == HEADLINE_METRIC[task_type] else metric


def _merge_metrics(formatted: list[pd.DataFrame]) -> pd.DataFrame:
    """Join the formatted tables of several metrics cell by cell, in the given order."""

    first, *rest = formatted

    # Sanity check: every metric reports the same models on the same tasks, so the
    # cells line up. Summary columns are part of that.
    for table in rest:
        assert table.shape == first.shape, (
            f"shapes differ: {table.shape}, {first.shape}"
        )
        assert list(table.columns) == list(first.columns), "columns differ"

    merged = reduce(lambda left, right: left + CELL_SEPARATOR + right, formatted)

    # A cell that reports no metric at all reads as one dash, not as a row of them.
    return merged.map(
        lambda cell: (
            MISSING_CELL if set(cell.split(CELL_SEPARATOR)) == {MISSING_CELL} else cell
        )
    )


def format_values(
    values: pd.DataFrame, metric: str, *, compute_ranks: bool, count_wins: bool
) -> pd.DataFrame:
    """
    Highlights the best value per task in bold (if more than one value to compare).
    Adds summary rows at the end (rank or wins) if requested.
    """

    # Highlighting only makes sense for more than one value.
    compare = len(values.dropna(how="all")) > 1

    table = pd.DataFrame(
        {
            task: _column(list(column), metric, compare=compare)
            for task, column in values.items()
        },
        index=values.index,
    )
    assert len(table.columns) == len(values.columns), "task labels are not unique"

    # A lone reporting row wins every task and ranks first on all of them, so there is
    # nothing left for a summary to say.
    if not compare:
        return table

    # Count wins by counting bold values (important: this is done before ranks are
    # added to the table).
    wins = _count_wins(table, values) if count_wins else None

    # Add wins to table and make best value bold.
    if wins is not None:
        table[WINS_COLUMN] = _bold_best(
            list(wins),
            [MISSING_CELL if pd.isna(c) else f"{c:.0f}" for c in wins],
            False,
        )

    # Add ranks to table and make best value bold.
    if compute_ranks:
        ranks = _mean_ranks(values, metric).reindex(values.index)
        table[RANK_COLUMN] = _bold_best(
            list(ranks),
            [MISSING_CELL if pd.isna(r) else f"{r:.{RANK_DECIMALS}f}" for r in ranks],
            True,
        )

    return table


def _count_wins(table: pd.DataFrame, values: pd.DataFrame) -> pd.Series:
    """Tasks a model is best on, read off by counting bolded cells (a bit hacky)."""
    counts = table.map(is_bold).sum(axis="columns").astype(float)
    return counts.where(values.notna().any(axis="columns"))


def model_values(
    df: pd.DataFrame,
    split: str,
    task_type: str,
    metric: str,
    models: Sequence[str] = (MAIN_MODEL,),
    value_column: str = OURS_VALUE,
    *,
    published: Sequence[str] = MODEL_ORDER,
) -> pd.DataFrame:
    """
    Collects the raw numbers with one row per model and one column per task. Our values
    are always last. There may be missing numbers (`np.nan`). Without `published` models, the
    baselines are left out and only the reported models remain.
    """

    # Get baseline values.
    baseline_values = baselines.load()
    published_models = published

    # Get our values.
    all_our_values = df[df["task_type"] == task_type]

    # Get sorted tasks, as the union of every reported model's own tasks (different
    # backbones need not have run the exact same task set).
    our_stages = {OURS[model]["stage"] for model in models}
    our_values = all_our_values[all_our_values["stage"].isin(our_stages)]
    tasks = sorted(set(zip(our_values["db"], our_values["task"])))

    return pd.DataFrame(
        {
            # Get specific baseline value (or np.nan).
            f"{db}/{task}": [
                _published(baseline_values, model, db, task, split, metric)
                for model in published_models
            ]
            # Get our value (or np.nan), one per reported model.
            + [
                our_value(
                    all_our_values, OURS[model]["stage"], db, task, split, value_column
                )
                for model in models
            ]
            for db, task in tasks
        },
        index=model_index(models, published=published),
    )


def our_value(
    df: pd.DataFrame,
    stage: str,
    db: str,
    task: str,
    split: str,
    value_column: str = OURS_VALUE,
) -> float:
    """Our own number for this cell, or NaN when that stage did not run the task."""
    match = df[(df["stage"] == stage) & (df["db"] == db) & (df["task"] == task)][
        f"{split}_{value_column}"
    ]

    # Sanity check: a stage runs a task once, so there is at most one.
    assert len(match) <= 1, f"{stage} {db}/{task} {split}: {len(match)} rows"

    return float(match.iloc[0]) if len(match) else np.nan


def task_labels(tasks: list[str]) -> dict[str, str]:
    """
    Label each `db/task` by its task name, appending the database only on a collision.
    """

    # Get task names.
    task_names = [task.split("/")[1] for task in tasks]

    # Format task names, taking care of possible collisions.
    labels = {
        task: name
        if task_names.count(name) == 1
        else DB_LABEL_FORMAT.format(
            task=name,
            db=task.split("/")[0].removeprefix(DB_LABEL_PREFIX)[:DB_LABEL_CHARS],
        )
        for task, name in zip(tasks, task_names)
    }

    # Sanity check: Check that no new duplicate was introduced.
    assert len(set(labels.values())) == len(labels), f"labels collide: {labels}"

    return labels


def model_index(
    models: Sequence[str] = (MAIN_MODEL,), *, published: Sequence[str] = MODEL_ORDER
) -> pd.MultiIndex:
    """The models the table reports, each under its kind, ours last."""

    # Get all possible "kinds".
    kinds = baselines.kinds(baselines.load()) | {m: OURS[m]["kind"] for m in models}

    # Transform to human-readable name.
    kinds = {name: KIND_LABEL[kind] for name, kind in kinds.items()}

    # Define order.
    ordered_models = list(published) + list(models)

    # Sanity check: A "kind" label is not interrupted through order.
    ordered = [kinds[model] for model in ordered_models]
    assert ordered == sorted(ordered, key=ordered.index), f"kinds interleave: {ordered}"

    # Transform to human-readable name, where the stored one is not. The main variant
    # of ours goes without its backbone where it is the only one.
    names = {MAIN_MODEL: OURS_LABEL} if list(models) == [MAIN_MODEL] else {}
    labels = [names.get(m, MODEL_LABEL.get(m, m)) for m in ordered_models]

    return pd.MultiIndex.from_tuples(
        zip(ordered, labels), names=[KIND_COLUMN, MODEL_HEADER]
    )


def _mean_ranks(values: pd.DataFrame, metric: str) -> pd.Series:
    """Mean rank over every task, 1 being best, missing numbers ranked last."""

    # Remove models that do not have any values.
    values_not_na = values[values.notna().any(axis="columns")]

    # A model that reports some tasks but not others still competes, and its missing
    # cells move every rank in those columns.
    if values_not_na.isna().to_numpy().any():
        logger.warning(
            "WARNING: Computing ranks even though some models are missing tasks. Those "
            "rank below every model reporting the task."
        )

    # `na_option = bottom` => missing values get last rank (needs discussing!).
    ranks = values_not_na.rank(ascending=LOWER_IS_BETTER[metric], na_option="bottom")

    return ranks.mean(axis="columns")


def _published(
    df: pd.DataFrame, model: str, db: str, task: str, split: str, metric: str
) -> float:
    """The one published number for this cell, or NaN when the model does not report it."""
    match = df[
        (df["model"] == model)
        & (df["db"] == db)
        & (df["task"] == task)
        & (df["split"] == split)
        & (df["metric"] == metric)
    ]["value"]

    # Sanity check: `baselines.load` rejects duplicates, so there is at most one.
    assert len(match) <= 1, f"{model} {db}/{task} {split} {metric}: {len(match)} rows"

    return float(match.iloc[0]) if len(match) else np.nan


# The appendix table of seed spread. ###################################################


def seeds_table(
    df: pd.DataFrame, split: str, task_type: str, spread: str, model: str
) -> pd.DataFrame:
    """Spread across the seed ensemble members of `model`, one row per task."""
    metric = HEADLINE_METRIC[task_type]
    percent = metric in PERCENT_METRICS
    runs = _our_runs(df, task_type, str(OURS[model]["stage"]))

    # Sanity check: a spread only compares across tasks at one member count.
    members = sorted(set(runs["n_members"]))
    assert len(members) == 1, f"member counts differ: {members}"

    tasks = [f"{db}/{task}" for db, task in zip(runs["db"], runs["task"])]
    labels = task_labels(tasks)
    table = pd.DataFrame({bold(TASK_COLUMN): [labels[task] for task in tasks]})

    # The scores and the spread read at the metric's own precision, and the mean
    # carries the spread named by `spread`, which is one run column and so one table.
    spreads = [_fmt(value, percent=percent) for value in runs[f"{split}_{spread}"]]
    for column, header in SEED_VALUE_COLUMNS.items():
        values = [_fmt(value, percent=percent) for value in runs[f"{split}_{column}"]]
        if column == SEED_SPREAD_OF:
            header = SPREAD_FORMAT.format(
                value=header, spread=SEED_SPREAD_COLUMNS[spread]
            )
            values = [
                SPREAD_FORMAT.format(value=value, spread=s)
                for value, s in zip(values, spreads)
            ]
        table[bold(header)] = values

    return table


def _our_runs(df: pd.DataFrame, task_type: str, stage: str) -> pd.DataFrame:
    """Our own runs of one stage on one task type, one row per task, in table order."""
    runs = df[(df["stage"] == stage) & (df["task_type"] == task_type)]
    return runs.sort_values(["db", "task"], ignore_index=True)


# The appendix table of per-task HPO winners. ##########################################


def hpo_best_table(df: pd.DataFrame, task_type: str) -> pd.DataFrame:
    """The best trial of each study, against the configuration the paper reports."""
    metric = HEADLINE_METRIC[task_type]
    percent = metric in PERCENT_METRICS
    lower = LOWER_IS_BETTER[metric]

    winners = _our_runs(df, task_type, HPO_STAGE)
    reference = _our_runs(df, task_type, HPO_REFERENCE_STAGE)
    runs = winners.merge(reference, on=["db", "task"], suffixes=("", "_ref"))

    # Sanity check: every tuned task has a reported configuration to compare to.
    assert len(runs) == len(winners), f"{task_type}: {len(runs)}/{len(winners)}"

    tasks = [f"{db}/{task}" for db, task in zip(runs["db"], runs["task"])]
    labels = task_labels(tasks)
    table = pd.DataFrame({bold(TASK_COLUMN): [labels[task] for task in tasks]})

    # Bold the better of the pair within each row.
    ours, ref = runs[f"{HPO_SPLIT}_value"], runs[f"{HPO_SPLIT}_value_ref"]
    cells = [
        _bold_best(
            [our, other],
            [_fmt(our, percent=percent), _fmt(other, percent=percent)],
            lower,
        )
        for our, other in zip(ours, ref)
    ]
    for index, label in enumerate((HPO_BEST_LABEL, OURS_LABEL)):
        table[bold(label)] = [row[index] for row in cells]

    # Signed so that a positive number always means tuning per task won.
    gains = (ref - ours) if lower else (ours - ref)
    table[bold(HPO_GAIN_COLUMN)] = [_signed(gain, percent=percent) for gain in gains]

    return table


# Formatting. ##########################################################################


def _column(values: list[Any], metric: str, *, compare: bool) -> list[str]:
    """
    Returns one formatted task column. The best value is bold when
    requested (`compare`). Some numbers (`PERCENT_METRICS`) are represented as
    percentages.
    """
    strings = [_fmt(value, percent=metric in PERCENT_METRICS) for value in values]
    if not compare:
        return strings
    return _bold_best(values, strings, LOWER_IS_BETTER[metric])


def _fmt(value: Any, *, percent: bool) -> str:
    """Formats a single number, as a percentage where `percent` is set."""
    if pd.isna(value):
        return MISSING_CELL
    if percent:
        return f"{value * 100:.{PERCENT_DECIMALS}f}"
    return f"{value:.{VALUE_DECIMALS}f}"


def _signed(value: Any, *, percent: bool) -> str:
    """Format a difference, always carrying its sign so its direction reads off."""
    if pd.isna(value):
        return MISSING_CELL
    return f"{'+' if value >= 0 else ''}{_fmt(value, percent=percent)}"


def _bold_best(
    values: list[Any], strings: list[str], lower_is_better: bool
) -> list[str]:
    """Makes the winner of a formatted column bold.

    :param values: raw values (used for comparison)
    :param strings: already formatted values (formatted as strings)
    """

    # Nothing to do if there are only missing values.
    known = [value for value in values if not pd.isna(value)]
    if not known:
        return strings

    # Determine best value.
    best = min(known) if lower_is_better else max(known)

    # Determine the str of the best value. This ensures that more than one value is
    # bold, even if one is slightly worse but invisible to formatting precision.
    formatted_winner = next(
        s for v, s in zip(values, strings) if not pd.isna(v) and v == best
    )
    return [bold(s) if s == formatted_winner else s for s in strings]


def bold(text: str) -> str:
    """Make one cell bold."""
    return BOLD.format(text=text)


def format_summary_row_header(column: str) -> str:
    """
    Format summary row header by (i) making it bold and (ii) adding an arrow for
    readability.
    """
    return bold(SUMMARY_HEADERS.get(column, column))


def is_bold(text: str) -> bool:
    """Check whether a value is bold."""
    return text.startswith(bold("").removesuffix("}"))


# Writing. #############################################################################


def write(
    table: pd.DataFrame,
    name: str,
    split: str,
    *,
    keep_kinds: bool = False,
    split_pairs: bool = True,
) -> None:
    """
    Write a bare `tabular` the paper inputs. `keep_kinds` keeps the kind header row
    even for a lone kind, so that the table lines up row by row with one it is set
    beside. Without `split_pairs`, a paired column stays one centered column whose
    metrics are padded to line up instead (see `_pad_pairs`).
    """
    path = GENERATED_DIR / (
        f"table-{name}.tex" if split == "test" else f"table-{name}-{split}.tex"
    )

    # A lone kind heads every column and so says nothing: drop that header row.
    if isinstance(table.columns, pd.MultiIndex) and not keep_kinds:
        kinds = {kind for kind in table.columns.get_level_values(0) if kind}
        if len(kinds) < 2:
            table = table.droplevel(0, axis=1)

    # Pad the pairs of a table that keeps them in one column.
    if not split_pairs:
        table = _pad_pairs(table)

    latex = table.to_latex(
        index=False,
        escape=False,
        multicolumn=True,
        multicolumn_format="c",
    )

    # Set the column spec here rather than through `column_format`, which cuts a spec
    # at the first ";" (e.g., the "\;" of `PAIRED_SPEC`).
    first, rest = latex.split("\n", 1)
    assert first.startswith("\\begin{tabular}"), first
    latex = (
        f"\\begin{{tabular}}{{{''.join(_column_specs(table, split_pairs))}}}\n{rest}"
    )
    if split_pairs:
        latex = _split_paired(
            latex, [_is_paired(table[column]) for column in table.columns]
        )
    latex = _rule_below_kinds(latex)
    path.write_text(TABLE_PREAMBLE + _rule_above_summary(latex))
    print(f"wrote {path.relative_to(REPO_ROOT)} ({len(table)} rows)")


def _column_specs(table: pd.DataFrame, split_pairs: bool = True) -> list[str]:
    """One spec per column, read off the cells rather than the header, since the variant
    tables head their label column with a name of their own."""
    return [
        (PAIRED_SPEC if split_pairs else NUMBER_SPEC)
        if _is_paired(table[column])
        else NUMBER_SPEC
        if _is_number_column(table[column])
        else LABEL_SPEC
        for column in table.columns
    ]


def _is_paired(cells: pd.Series) -> bool:
    """Whether a column carries several metrics per cell, i.e. takes `PAIRED_SPEC`."""
    return any(CELL_SEPARATOR in str(cell) for cell in cells)


def _pad_pairs(table: pd.DataFrame) -> pd.DataFrame:
    """
    Pad the metrics of every paired cell with invisible digits to the widest of their
    column, the first metric on the left and the second on the right. Digits and the
    dash of a missing metric are equally wide, so the separators line up while the
    column stays centered under a header wider than its cells. A minus sign is
    narrower, so a negative value sits slightly off.
    """

    def visible(text: str) -> int:
        return len(BOLD_PATTERN.sub(r"\1", text))

    def pad(width: int) -> str:
        return PHANTOM.format(text="0" * width) if width > 0 else ""

    table = table.copy()
    for column in table.columns:
        if not _is_paired(table[column]):
            continue
        parts = [str(cell).split(CELL_SEPARATOR) for cell in table[column]]
        pairs = [p for p in parts if len(p) == 2]
        left = max(visible(first) for first, _ in pairs)
        right = max(visible(second) for _, second in pairs)
        table[column] = [
            pad(left - visible(p[0]))
            + p[0]
            + CELL_SEPARATOR
            + p[1]
            + pad(right - visible(p[1]))
            if len(p) == 2
            else p[0]
            for p in parts
        ]
    return table


def _split_paired(latex: str, paired: list[bool]) -> str:
    """
    Split every paired column into the two table columns of `PAIRED_SPEC`: a cell's
    metrics go either side of the separator, and a cell reporting none, as well as
    every header cell, spans both.
    """
    if not any(paired):
        return latex

    lines = latex.split("\n")
    header = lines.index(TOP_RULE) + 1
    body = lines.index(MID_RULE)

    for i in range(header, len(lines)):
        line = lines[i]
        if i == body or not line.endswith("\\\\"):
            continue

        # Cells split on unescaped ampersands; header cells may span several columns.
        cells = UNESCAPED_AMPERSAND.split(line.removesuffix("\\\\"))
        column = 0
        out = []
        for cell in cells:
            cell = cell.strip()
            match = SPAN_PATTERN.match(cell) if i < body else None
            span = int(match.group(1)) if match else 1
            text = match.group(2) if match else cell

            # A body cell with several metrics fills both of its table columns.
            if i > body and paired[column] and CELL_SEPARATOR in cell:
                out.append(cell.replace(CELL_SEPARATOR, " & "))
            else:
                width = span + sum(paired[column : column + span])
                out.append(
                    SPAN_FORMAT.format(width=width, text=text) if width > 1 else cell
                )
            column += span
        lines[i] = " & ".join(out) + " \\\\"

    return "\n".join(lines)


def _is_number_column(cells: pd.Series) -> bool:
    """Whether every cell reporting anything is a number, a spread ("0.040 $\\pm$
    0.002") and a unit ("1.00$\\times$") included. A cell may report several metrics
    at once, and one of them missing does not make the column a label column. A column
    reporting nothing counts too, so that its dashes line up with the cells beside
    them."""
    return all(
        NUMBER_PATTERN.match(part)
        for cell in cells
        for part in str(cell).split(CELL_SEPARATOR)
        if part != MISSING_CELL
    )


def _rule_below_kinds(latex: str) -> str:
    """
    Underline every kind of the top header row, so that the columns it heads read as
    one group. Tables that head no group at all are left alone; `write` has already
    dropped the header row of a lone kind.
    """

    lines = latex.split("\n")

    # The kinds head the row right below the top rule, where there is one at all.
    if TOP_RULE not in lines:
        return latex
    header = lines.index(TOP_RULE) + 1
    if header >= len(lines):
        return latex

    # A single header row names the columns; kinds need a row of their own above it.
    if lines[header + 1] == MID_RULE:
        return latex

    spans = _header_spans(lines[header])

    # No kinds head this table.
    if not spans:
        return latex

    rules = "".join(GROUP_RULE.format(first=first, last=last) for first, last in spans)
    return "\n".join([*lines[: header + 1], rules, *lines[header + 1 :]])


def _header_spans(header: str) -> list[tuple[int, int]]:
    """First and last column of every group the header row spans, counting from one."""

    spans = []
    column = 1
    for cell in header.removesuffix("\\\\").split("&"):
        match = SPAN_PATTERN.match(cell.strip())
        width = int(match.group(1)) if match else 1

        # An empty cell heads nothing, so it is not a group. A kind over a single
        # column is a plain cell, but a group all the same.
        text = match.group(2) if match else cell
        if text.strip():
            spans.append((column, column + width - 1))

        column += width

    return spans


def _rule_above_summary(latex: str) -> str:
    """
    Rule off the summary rows from the tasks above them.

    Only where a summary heads a row of its own: in the other layout it is a column, so
    its label sits mid-line and is left alone. The rule goes above the first summary
    row present, so that rank and wins are ruled off together as one block.
    """

    for column in SUMMARY_COLUMNS:
        label = f"\n{format_summary_row_header(column)} &"
        if label in latex:
            return latex.replace(label, f"\n{SUMMARY_RULE}{label}", 1)

    return latex
