import os
import os.path as osp
import re
import time
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import optuna
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from optuna.importance import FanovaImportanceEvaluator, get_param_importances
from optuna.samplers import BaseSampler, TPESampler
from optuna.study import MaxTrialsCallback, StudyDirection
from optuna.trial import TrialState
from optuna.visualization import (
    plot_edf,
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
    plot_timeline,
)

from hpo.config import REPO_ROOT
from hpo.objective import Objective
from hpo.params import parse_conditional_groups, parse_param
from hpo.samplers import ResumableRandomSampler
from hpo.test_top_k import HP_METRIC_ATTR, run_top_k_test
from hpo.utils_analysis import (
    AnalysisArgs,
    get_combined_marginals_df,
    get_hyperparameter_ranges_df,
    get_marginals_df,
    parse_args,
)
from hpo.utils_hpo import (
    RelICLSubprocessError,
    WriteResultsCallback,
    get_direction,
    get_task_type,
    resolve_hp_metric,
    trials_dataframe,
)


@hydra.main(
    version_base="1.3", config_path=str(REPO_ROOT / "config"), config_name="hpo"
)
def hpo(cfg: DictConfig) -> None:
    run_dir = HydraConfig.get().run.dir
    job_name = HydraConfig.get().job.name

    # The task type fixes the default metric and gates the task-type conditional groups.
    task_type = get_task_type(cfg.relbench.db, cfg.relbench.task)
    cfg.relbench.hp_metric = resolve_hp_metric(cfg.relbench)

    # Initialize optuna.
    sampler: BaseSampler
    if cfg.optuna.sampler == "tpe":
        sampler = TPESampler(seed=cfg.optuna.seed)  # type: ignore
    elif cfg.optuna.sampler == "random":
        sampler = ResumableRandomSampler(seed=cfg.optuna.seed)
    else:
        raise NotImplementedError(f"Unknown sampler: {cfg.optuna.sampler}")

    try:
        study = optuna.create_study(
            storage="sqlite:///" + osp.join(run_dir, f"{job_name}.db"),
            study_name=cfg.optuna.study_name,
            direction=get_direction(cfg.relbench.hp_metric),
            sampler=sampler,
            load_if_exists=cfg.optuna.load_if_exists,
        )
    except optuna.exceptions.DuplicatedStudyError as e:
        raise ValueError(
            "Study already exists. To continue, set optuna.load_if_exists=True."
        ) from e

    # Record the optimized metric, which `test-top-k` needs when run standalone.
    study.set_user_attr(HP_METRIC_ATTR, cfg.relbench.hp_metric)

    # Parse parameters.
    param_definitions = {
        name: parse_param(name, spec) for name, spec in cfg.params.items()
    }

    # Parse conditional parameters.
    conditional_groups = parse_conditional_groups(cfg.conditional_params)

    # Optuna counts the budget in seconds.
    timeout_h = cfg.optuna.get("timeout_h", None)
    timeout_s = None if timeout_h is None else float(timeout_h) * 3600.0

    # The deadline caps each individual trial, so none can overrun the study budget.
    # Optuna only checks its own timeout between trials.
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    timeout_multiplier = cfg.optuna.get("trial_timeout_multiplier", None)

    # Optimize.
    objective = Objective(
        cfg.relbench,
        param_definitions,
        conditional_groups,
        cfg.optuna.seed,
        cfg.trial_seeds,
        task_type,
        cfg.run_test,
        deadline=deadline,
        timeout_multiplier=(
            None if timeout_multiplier is None else float(timeout_multiplier)
        ),
        timeout_warmup=int(cfg.optuna.get("trial_timeout_warmup", 3)),
    )
    # noinspection PyTypeChecker
    study.optimize(
        objective,
        timeout=timeout_s,
        callbacks=[
            WriteResultsCallback(osp.join(run_dir, f"{job_name}.csv")),  # type: ignore[list-item]
            MaxTrialsCallback(n_trials=cfg.optuna.n_trials),
        ],
        catch=(RelICLSubprocessError,),
    )

    # Test the best trials before the analysis so its CSVs carry the test columns.
    path_to_db = osp.join(run_dir, f"{job_name}.db")
    run_top_k_test(path_to_db, cfg.test_top_k)

    # Produce analysis.
    analysis(AnalysisArgs(path_to_db=path_to_db))


def analysis(analysis_args: AnalysisArgs | None = None) -> None:
    if analysis_args is None:
        analysis_args = parse_args()

    # Load study. ######################################################################

    # Get output dir.
    out_dir = osp.join(osp.dirname(analysis_args.path_to_db), "_ANALYSIS")
    os.makedirs(out_dir, exist_ok=True)

    # Load study.
    study = optuna.load_study(
        study_name=None,
        storage=f"sqlite:///{analysis_args.path_to_db}",
    )

    # Abort if less than two completed trials.
    n_completed_trials = sum(
        [1 for t in study.trials if t.state == TrialState.COMPLETE]
    )
    if n_completed_trials < 2:
        print(f"Less than two completed trials found in {analysis_args.path_to_db}.")
        return

    # Plain results. ###################################################################

    # Sorted CSV.
    trials_df: pd.DataFrame = trials_dataframe(study, analysis_args.path_to_db)
    trials_df.sort_values(
        "value", ascending=False if study.direction == StudyDirection.MAXIMIZE else True
    ).to_csv(osp.join(out_dir, f"{study.study_name}-results_sorted.csv"), index=False)

    # Hyperparameter importances. ######################################################

    importances = get_param_importances(
        study, evaluator=FanovaImportanceEvaluator(seed=0)
    )
    importances_df = pd.DataFrame(
        importances.items(),
        columns=["param", "importance"],  # type: ignore
    ).sort_values("importance", ascending=False)

    hp_summary_df = get_hyperparameter_ranges_df(
        study, importances_df, trials_df, analysis_args.n_top_trials
    )
    hp_summary_df.to_csv(
        osp.join(
            out_dir,
            f"{study.study_name}-hp-summary-stats-{analysis_args.n_top_trials}-trials.csv",
        ),
        index=False,
    )

    # Optuna plots. ####################################################################
    if analysis_args.with_plots:
        plot_dir = osp.join(out_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_fns = [
            plot_param_importances,
            plot_slice,
            plot_edf,
            plot_optimization_history,
            plot_timeline,
        ]
        for plot_fn in plot_fns:
            plot = plot_fn(study)  # type: ignore[operator]
            plot.write_image(
                osp.join(plot_dir, f"{study.study_name}-{plot_fn.__name__}.pdf")
            )


def combined_analysis() -> None:
    # Parse args. ######################################################################

    parser = ArgumentParser()
    parser.add_argument("--min_trial_success_rate", "-r", type=float, default=0.7)
    parser.add_argument(
        "--dir",
        "-d",
        type=Path,
        default=REPO_ROOT / "hpo-outputs",
        help="Directory searched recursively for study databases.",
    )
    parsed_args = parser.parse_args()
    min_trial_success_rate: float = parsed_args.min_trial_success_rate
    hpo_dir: Path = parsed_args.dir

    # Find all *.db files. #############################################################
    db_files = list(hpo_dir.rglob("*.db"))

    # Load each DB and group the studies by direction. #################################

    groups: dict[StudyDirection, dict[str, pd.DataFrame]] = defaultdict(dict)
    for db_file in db_files:
        study = optuna.load_study(study_name=None, storage=f"sqlite:///{db_file}")
        groups[study.direction][study.study_name] = trials_dataframe(
            study, str(db_file)
        )

    # Analyze each group on its own. ###################################################
    per_group: dict[str, tuple[pd.DataFrame, list[str]]] = {}
    for study_dir, dfs in groups.items():
        name = study_dir.name.lower()
        out_file = hpo_dir / f"combined-analysis-{name}.csv"
        print(f"Combining {len(dfs)} {name} studies into {out_file.name}:")
        for study_name in sorted(dfs):
            print(f"  {study_name}")
        df, param_col_names = _combine_studies(
            dfs, study_dir, min_trial_success_rate, out_file
        )
        per_group[name] = (df, param_col_names)

        # Marginals.
        marginals_df = get_marginals_df(df, param_col_names)
        marginals_file = hpo_dir / f"marginals-{name}.csv"
        marginals_df.to_csv(marginals_file, index=False)

    # Marginals over every task at once. ###############################################
    # One configuration ships in the end, so the levels have to be compared over all
    # tasks and not once per direction group.
    if len(per_group) > 1:
        combined = get_combined_marginals_df(per_group)
        combined_file = hpo_dir / "marginals-combined.csv"
        combined.to_csv(combined_file, index=False)
        print(f"Wrote {combined_file.name} over {len(per_group)} direction groups.")


def _combine_studies(
    dfs: dict[str, pd.DataFrame],
    study_dir: StudyDirection,
    min_trial_success_rate: float,
    out_file: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Join the trials of several studies by trial number and rank them jointly."""

    # Join all data frames into one. ###################################################
    df: pd.DataFrame | None = None
    left_on: str | None = None
    for study_name, other_df in dfs.items():
        if df is None:
            df = other_df.add_suffix(f"_{study_name}")
            left_on = f"number_{study_name}"
            df = df.assign(number=df[left_on])
        else:
            assert left_on is not None
            other_df = other_df.add_suffix(f"_{study_name}")
            df = df.merge(
                other_df,
                left_on=left_on,
                right_on=f"number_{study_name}",
                suffixes=(None, None),
                how="outer",
            )
    assert df is not None

    # Clean up parameter columns. ######################################################

    full_param_col_names: list[str] = [c for c in df.columns if c.startswith("params_")]

    param_groups: dict[str, list[str]] = defaultdict(list)
    for param in full_param_col_names:
        cleaned = re.sub(r"^params_", "", re.sub(r"_rel-.*$", "", param))
        param_groups[cleaned].append(param)

    for cleaned_name, full_names in param_groups.items():
        # Sanity check: a parameter must hold the same value in every study that
        # recorded it, otherwise joining trials by number compares different configs.
        conflicting = df.loc[df[full_names].nunique(axis=1, dropna=True) > 1, "number"]
        assert conflicting.empty, (
            f"Parameter '{cleaned_name}' has differing values across studies in "
            f"{len(conflicting)} trial(s), first is trial {conflicting.iloc[0]}. "
            "The studies did not suggest the same configurations."
        )

    # Keep only one occurrence per parameter.
    cleaned_params = pd.DataFrame(
        {name: df[full_names[0]] for name, full_names in param_groups.items()},
        index=df.index,
    )
    df = pd.concat([df.drop(columns=full_param_col_names), cleaned_params], axis=1)

    param_col_names = list(param_groups.keys())

    # Remove trials with too few values. ###############################################
    value_col_names = [c for c in df.columns if c.startswith("value_")]

    # Replace inf/-inf with nan.
    df = df.replace([np.inf, -np.inf], np.nan)

    # Track how many datasets actually contributed a value to each trial.
    df["n_datasets"] = df[value_col_names].notna().sum(axis=1)

    # Remove trials with too little data.
    success_fraction = df[value_col_names].notna().mean(axis=1)
    df = df[success_fraction >= min_trial_success_rate]

    # Compute regret per study (best value in that study minus this trial's value, so
    # regret is always >= 0 and lower is always better, regardless of direction).
    regret_col_names: list[str] = []
    normalized_regret_col_names: list[str] = []
    for col in value_col_names:
        study_name = col[len("value_") :]

        best = (
            df[col].max(skipna=True)
            if study_dir is StudyDirection.MAXIMIZE
            else df[col].min(skipna=True)
        )

        regret_col_name = f"regret_{study_name}"

        match study_dir:
            case StudyDirection.MAXIMIZE:
                df[regret_col_name] = best - df[col]
            case StudyDirection.MINIMIZE:
                df[regret_col_name] = df[col] - best
            case _:
                raise ValueError(f"Unknown direction: {study_dir}")

        regret_col_names.append(regret_col_name)

        # Normalized regret (by interquartile range).
        iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
        normalized_regret_col_name = f"normalized_regret_{study_name}"

        # A task whose trials all score alike is insensitive to every parameter and
        # carries no signal, so it drops out of the aggregate rather than exploding.
        df[normalized_regret_col_name] = (
            df[regret_col_name] / iqr if iqr > 0 else np.nan
        )

        normalized_regret_col_names.append(normalized_regret_col_name)

    # Aggregate raw performance across studies. ########################################

    df["mean_value"] = df[value_col_names].mean(axis=1)
    df["median_value"] = df[value_col_names].median(axis=1)
    df["std_value"] = df[value_col_names].std(axis=1)
    value_summary_col_names = ["mean_value", "median_value", "std_value"]

    # Aggregate regret across studies. #################################################

    df["mean_regret"] = df[regret_col_names].mean(axis=1)
    df["std_regret"] = df[regret_col_names].std(axis=1)
    df["median_regret"] = df[regret_col_names].median(axis=1)
    regret_summary_col_names = ["mean_regret", "std_regret", "median_regret"]

    df["mean_normalized_regret"] = df[normalized_regret_col_names].mean(axis=1)
    df["std_normalized_regret"] = df[normalized_regret_col_names].std(axis=1)
    df["median_normalized_regret"] = df[normalized_regret_col_names].median(axis=1)
    normalized_regret_summary_col_names = [
        "mean_normalized_regret",
        "std_normalized_regret",
        "median_normalized_regret",
    ]

    # Compute individual ranks per study. ##############################################
    rank_col_names: list[str] = []
    for col in value_col_names:
        study_name = col[len("value_") :]

        rank_col = f"rank_{study_name}"
        df[rank_col] = df[col].rank(
            method="dense",
            ascending=False if study_dir == StudyDirection.MAXIMIZE else True,
            na_option="bottom",
        )
        rank_col_names.append(rank_col)

    # Aggregate ranks across studies. ##################################################

    df["mean_rank"] = df[rank_col_names].mean(axis=1)
    df["std_rank"] = df[rank_col_names].std(axis=1)
    df["median_rank"] = df[rank_col_names].median(axis=1)
    rank_summary_col_names = ["mean_rank", "std_rank", "median_rank"]

    # Aggregate per-trial runtime across studies. ######################################
    duration_col_names = [c for c in df.columns if c.startswith("duration_s_")]
    duration_summary_col_names: list[str] = []
    if duration_col_names:
        df["mean_duration_s"] = df[duration_col_names].mean(axis=1)
        df["total_duration_s"] = df[duration_col_names].sum(axis=1)
        duration_summary_col_names = ["mean_duration_s", "total_duration_s"]

    # Put metric cols at the front. ####################################################
    test_value_col_names = [c for c in df.columns if "test_value" in c]

    df = df[
        ["number", "n_datasets"]
        + normalized_regret_summary_col_names
        + regret_summary_col_names
        + value_summary_col_names
        + rank_summary_col_names
        + duration_summary_col_names
        + test_value_col_names
        + value_col_names
        + param_col_names
        + normalized_regret_col_names
        + regret_col_names
        + rank_col_names
        + duration_col_names
    ].sort_values("mean_regret")

    # Strip 'user_attrs_*' from all column names. ######################################
    df = df.rename(columns=lambda c: c.removeprefix("user_attrs_"))

    # Store to disk as CSV.
    df.to_csv(out_file, index=False)

    return df, param_col_names


if __name__ == "__main__":
    hpo()  # type: ignore[arg-list]
