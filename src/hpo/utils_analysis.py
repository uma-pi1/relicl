from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from optuna import Study
from optuna.study import StudyDirection
from pandas import DataFrame


@dataclass
class AnalysisArgs:
    path_to_db: str
    n_top_trials: int = 5
    with_plots: bool = False


def get_hyperparameter_ranges_df(
    study: Study, importances_df: DataFrame, trials_df: DataFrame, n_top_trials: int
) -> DataFrame:
    # HP ranges. #######################################################################

    # Best trials.
    top_trials = (
        trials_df.nlargest(n=n_top_trials, columns=["value"])
        if study.direction == StudyDirection.MAXIMIZE
        else trials_df.nsmallest(n=n_top_trials, columns=["value"])
    )

    # Most important HPs.
    top_hps = importances_df.set_index("param")["importance"].to_dict()

    # CSV.
    hp_summary_rows: list[dict[str, float]] = []
    for param, importance in top_hps.items():
        col_name = f"params_{param}"
        series = top_trials[col_name]

        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
            series
        ):
            row = dict(
                param=param,
                type="numeric",
                importance=importance,
                n=series.count(),
                mean=series.mean(),
                std=series.std(),
                min=series.min(),
                q25=series.quantile(0.25),
                median=series.median(),
                q75=series.quantile(0.75),
                max=series.max(),
            )

        elif pd.api.types.is_bool_dtype(series):
            row = dict(
                param=param,
                type="bool",
                importance=importance,
                n=series.count(),
                true_rate=series.mean(),  # proportion of True
                false_rate=1 - series.mean(),
            )

        else:
            row = dict(
                param=param,
                type="categorical",
                importance=importance,
                n=series.count(),
                mode=series.mode().iloc[0] if not series.mode().empty else None,
                unique=series.nunique(),
                value_counts=series.value_counts().to_dict(),
            )

        hp_summary_rows.append(row)  # type: ignore

    hp_summary_df = pd.DataFrame(
        hp_summary_rows,
        columns=np.array(
            [
                "param",
                "type",
                "importance",
                "n",
                "mean",
                "std",
                "min",
                "q25",
                "median",
                "q75",
                "max",
                "true_rate",
                "false_rate",
                "mode",
                "unique",
                "value_counts",
            ]
        ),
    )
    return hp_summary_df


# Hyperparameter marginals. ############################################################


# The three per-task metrics the marginals summarize, each as
# (summary column, per-task column prefix). All three are "lower is better" and already
# direction-free: rank is 1 for the best trial in a study whichever way it optimizes,
# and regret is the distance from that study's best value, so it is >= 0 either way.
MARGINAL_METRICS: list[tuple[str, str]] = [
    # Scale-free, so it is the one to read first.
    ("mean_rank", "rank_"),
    # Raw regret, in the task's own units. Not comparable across tasks; reference only.
    ("mean_regret", "regret_"),
    # Regret in interquartile ranges of the task's own spread across configurations,
    # which is comparable across tasks where raw regret is not.
    ("mean_normalized_regret", "normalized_regret_"),
]


def _per_task_cols(df: DataFrame, prefix: str) -> list[str]:
    """The per-task columns for one metric, excluding its own summary columns."""
    return [c for c in df.columns if c.startswith(prefix)]


def get_marginals_df(df: DataFrame, param_col_names: list[str]) -> DataFrame:
    """One row per hyperparameter level, averaged over every other hyperparameter."""

    # Per-task columns per metric, kept only long enough to count the wins below.
    task_cols = {prefix: _per_task_cols(df, prefix) for _, prefix in MARGINAL_METRICS}

    tables: list[DataFrame] = []
    for param in param_col_names:
        rows: list[dict[str, object]] = []
        for level, group in df.groupby(param, dropna=False):
            row: dict[str, object] = dict(param=param, level=level, n=len(group))

            for summary, prefix in MARGINAL_METRICS:
                row[summary] = group[summary].mean()
                # Standard error of the level mean. The yardstick for a gap between two
                # levels is these two values added in quadrature, not either one alone.
                row[f"se_{prefix.rstrip('_')}"] = group[summary].sem()
                row |= {c: group[c].mean() for c in task_cols[prefix]}

            rows.append(row)

        table = DataFrame(rows).sort_values("mean_rank")

        # Wins: tasks (not trials) whose own mean metric is best at this level, so each
        # wins column sums to the number of tasks. A level that wins on most tasks is a
        # real effect, one that wins on `n_tasks / n_levels` of them is a coin flip.
        # Wins count only which level led per task, never by how much, so a level can
        # lead here on one task's large margin plus five ties. All three metrics are
        # minimized, so "best" is the smallest value in every case.
        for _, prefix in MARGINAL_METRICS:
            cols = task_cols[prefix]
            best = table[cols].min(axis=0)
            table[f"{prefix.rstrip('_')}_wins"] = (table[cols] == best).sum(axis=1)

        drop = [c for cols in task_cols.values() for c in cols]
        tables.append(table.drop(columns=drop))

    return pd.concat(tables, ignore_index=True)


def get_combined_marginals_df(
    per_group: dict[str, tuple[pd.DataFrame, list[str]]],
) -> pd.DataFrame:
    """Marginals over the tasks of every direction group at once.

    The groups are stacked by row rather than joined by trial number, because they do
    not necessarily draw  the same configurations: for instance, a regression study
    may sample one extra parameter (`method.target_transform`).
    """

    # Stack, tagging each configuration with the group it came from. A row carries only
    # its own group's per-task columns; the other group's are NaN and skipped below.
    stacked = pd.concat(
        [df.assign(group=name) for name, (df, _) in per_group.items()],
        ignore_index=True,
    )

    # Which groups searched each parameter.
    param_groups: dict[str, list[str]] = defaultdict(list)
    for name, (_, param_col_names) in per_group.items():
        for param in param_col_names:
            param_groups[param].append(name)

    n_tasks = len([c for c in stacked.columns if c.startswith("rank_")])

    # A parameter searched in only some groups (`method.target_transform` is regression
    # only) is marginalized over those groups alone. Pooling it over every task would
    # collect every configuration of the other groups into a single NaN level.
    tables: list[pd.DataFrame] = []
    shared = [p for p, names in param_groups.items() if len(names) == len(per_group)]
    if shared:
        table = get_marginals_df(stacked, shared)
        # Raw regret is in each task's own units, so pooling MAE against AUROC would
        # produce a number with no meaning. Its per-task wins survive (each is decided
        # inside one task's own column), the aggregate does not.
        table[["mean_regret", "se_regret"]] = np.nan
        table["n_tasks"] = n_tasks
        table["groups"] = "all"
        tables.append(table)

    for name, (df, _) in per_group.items():
        own = [p for p, names in param_groups.items() if names == [name]]
        if not own:
            continue
        table = get_marginals_df(df, own)
        table["n_tasks"] = len([c for c in df.columns if c.startswith("rank_")])
        table["groups"] = name
        tables.append(table)

    return pd.concat(tables, ignore_index=True)


# Parse args. ##########################################################################


def parse_args() -> AnalysisArgs:
    parser = ArgumentParser()
    parser.add_argument(
        "--path", "-p", type=str, required=True, help="path to optuna's sqlite DB"
    )
    parser.add_argument(
        "--top_n_trials",
        "-n",
        type=int,
        default=10,
        help="produce hyperparameter summary stats for the TOP_N_TRIALS (default: 10)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_false",
        dest="with_plots",
        default=True,
        help="Disable plots (e.g., if viz dependencies are not available)",
    )
    parsed_args = parser.parse_args()
    return AnalysisArgs(
        path_to_db=parsed_args.path,
        n_top_trials=parsed_args.top_n_trials,
        with_plots=parsed_args.with_plots,
    )
