from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd

from paper import (
    baselines,
    config_ensemble,
    dataset_stats,
    early_fusion,
    hpo,
    results,
    tables,
    variants,
    win_matrix,
)
from paper.config import (
    ALL_MODEL_ORDER,
    BACKBONE_MODELS,
    EXPERIMENTS_DIR,
    GENERATED_DIR,
    HEADLINE_METRIC,
    HPO_OUTPUTS_SUBDIR,
    MAIN_MODELS,
    OURS,
    OUTPUTS_SUBDIR,
    REPORTED_METRICS,
    RESULT_KEYS,
    RESULTS_FILE,
    SEED_ENSEMBLE_MODELS,
    SEED_SPREAD_COLUMNS,
    SPLITS,
    TABPFN,
    WIN_MATRIX_SPLIT,
    WIN_MATRIX_TABLES,
)
from paper.variants import VARIANT_TABLES
from relicl.config import REPO_ROOT

# Commands. ############################################################################


def _collect(args: Namespace) -> None:
    """
    Go through the `EXPERIMENTS_DIR` directory and collect results. The outputs will go
    to `RESULTS_FILE` in the `GENERATED_DIR`.
    """

    # Make output dir.
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    # Create DF with rows from the winning HPO trials and all further experiments.
    outputs = args.experiments / OUTPUTS_SUBDIR
    df = pd.concat(
        [results.collect(outputs), hpo.collect(args.experiments / HPO_OUTPUTS_SUBDIR)],
        ignore_index=True,
    )
    df = results.add_headline_metric(df).sort_values(RESULT_KEYS, ignore_index=True)

    # Write to disk.
    df.to_csv(RESULTS_FILE, index=False)
    print(f"wrote {RESULTS_FILE.relative_to(REPO_ROOT)} ({len(df)} rows)")

    # Logging.
    print(f"{len(df)} rows from {df['run_dir'].nunique()} runs")
    for stage, group in df.groupby("stage"):
        revisions = "/".join(sorted(set(group["revision"])))
        print(f"  {stage:<13} {group['run_dir'].nunique():>3} runs  {revisions}")


def _tables(args: Namespace) -> None:
    """
    Produce many output LaTeX tables based on the produced output CSV
    from `_collect` (above).
    """
    # Load the exported table from `_collect`.
    df = pd.read_csv(RESULTS_FILE, low_memory=False)

    # Write the main results table, one per task type, for both splits. The main
    # tables carry no summary rows: the win matrix summarizes them instead.
    for split in SPLITS:
        for task_type in HEADLINE_METRIC:
            table = tables.final_table(
                df,
                split,
                task_type,
                models_as_rows=args.models_as_rows,
                rank=False,
                wins=False,
                models=MAIN_MODELS,
            )
            tables.write(table, f"final-{task_type}", split)

    # Our two backbones against each other, without the baselines: the main table
    # reports the tuned backbone alone. No summary rows, as above.
    for split in SPLITS:
        for task_type in HEADLINE_METRIC:
            table = tables.final_table(
                df,
                split,
                task_type,
                models_as_rows=args.models_as_rows,
                rank=False,
                wins=False,
                models=BACKBONE_MODELS,
                published=[],
            )
            tables.write(table, f"backbones-{task_type}", split)

    # The TabPFN backbone alone, which the paper sets beside the main table. A table
    # of its own, so it takes no part in the main table's bolding. It borrows the task
    # labels of the main table and keeps its kind row, so that the rows line up.
    for task_type in HEADLINE_METRIC:
        table = tables.final_table(
            df,
            "test",
            task_type,
            rank=False,
            wins=False,
            models=[TABPFN],
            published=[],
        )
        table = table.drop(columns=table.columns[0])

        # Its lone header is much wider than the metrics below it, and a split pair
        # would leave all the extra width on the right, so pad the pair instead.
        tables.write(
            table, f"tabpfn-{task_type}", "test", keep_kinds=True, split_pairs=False
        )

    # The same comparison for the seed ensemble (i.e., before config-ensembling), both
    # backbones side by side.
    for split in SPLITS:
        for task_type in HEADLINE_METRIC:
            table = tables.final_table(
                df,
                split,
                task_type,
                models_as_rows=args.models_as_rows,
                rank=args.ranks,
                wins=args.wins,
                models=SEED_ENSEMBLE_MODELS,
            )
            tables.write(table, f"final-seeds-{task_type}", split)

    # The appendix comparison against every published model, test only, one row per
    # model since there are many of them.
    published = baselines.load()

    # Leave out models without any number for the task type at hand.
    def reporting(models: list[str], task_type: str) -> list[str]:
        has_metric = published["metric"].isin(REPORTED_METRICS[task_type])
        reported = set(published[has_metric]["model"])
        return [model for model in models if model in reported]

    for task_type in HEADLINE_METRIC:
        table = tables.final_table(
            df,
            "test",
            task_type,
            models_as_rows=True,
            rank=args.ranks,
            wins=args.wins,
            models=BACKBONE_MODELS,
            published=reporting(ALL_MODEL_ORDER, task_type),
        )
        tables.write(table, f"all-models-{task_type}", "test")

    # The pairwise win matrices: how often each model beats each other model, test only.
    for matrix, matrix_spec in WIN_MATRIX_TABLES.items():
        for task_type in HEADLINE_METRIC:
            table = win_matrix.win_matrix_table(
                df,
                task_type,
                published=reporting(matrix_spec["published"], task_type),
                models=matrix_spec["models"],
                pairs_won=args.pairs_won,
            )
            tables.write(table, f"{matrix}-{task_type}", WIN_MATRIX_SPLIT)

        # The same matrix over the tasks of both task types at once.
        table = win_matrix.combined_win_matrix_table(
            df,
            {
                task_type: reporting(matrix_spec["published"], task_type)
                for task_type in HEADLINE_METRIC
            },
            models=matrix_spec["models"],
        )
        tables.write(table, f"{matrix}-combined", WIN_MATRIX_SPLIT)

    # Write one pair of tables per run stage whose variants are
    # compared (e.g., ablations).
    for name, spec in VARIANT_TABLES.items():
        for split in spec.splits:
            for task_type in HEADLINE_METRIC:
                table = variants.variants_table(
                    df, split, task_type, spec, rank=args.ranks, wins=args.wins
                )
                tables.write(table, f"{name}-{task_type}", split)

    # Write the appendix tables of seed spread, one per backbone and measure of spread.
    for split in SPLITS:
        for task_type in HEADLINE_METRIC:
            for model in SEED_ENSEMBLE_MODELS:
                backbone = OURS[model]["backbone"]
                for spread, label in SEED_SPREAD_COLUMNS.items():
                    table = tables.seeds_table(df, split, task_type, spread, model)
                    name = f"seeds-{backbone}-{label.lower()}-{task_type}"
                    tables.write(table, name, split)

    # Write the appendix tables of what per-task tuning would have won.
    for task_type in HEADLINE_METRIC:
        table = tables.hpo_best_table(df, task_type)
        # Both splits are columns here, so "test" only selects an unsuffixed name.
        tables.write(table, f"hpo-best-{task_type}", "test")

    # Write the appendix tables of RelBench dataset/task statistics. Reads
    # RESULTS_FILE itself, to filter tasks down to the ones we actually ran.
    dataset_stats.write_all()

    # Write the appendix tables of the OpenML early-fusion experiment. Reads its own
    # run outputs, which come from `sandbox/` rather than from `EXPERIMENTS_DIR`.
    early_fusion.write_all()


def _select(args: Namespace) -> None:
    """Pick the config-ensemble recipes on val."""
    config_ensemble.select(args.experiments / OUTPUTS_SUBDIR)


def _assemble(args: Namespace) -> None:
    """Rebuild the config-ensemble stages from the recipes."""
    config_ensemble.assemble(args.experiments / OUTPUTS_SUBDIR)


# Entry-point. #########################################################################

COMMANDS = dict(collect=_collect, tables=_tables, select=_select, assemble=_assemble)


def main() -> None:
    # Parse args. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    parser = ArgumentParser(
        "paper", description="Build the paper's tables and figures."
    )
    parser.add_argument("command", choices=list(COMMANDS))
    parser.add_argument(
        "--experiments",
        type=Path,
        default=EXPERIMENTS_DIR,
        help=f"run set to read (default: {EXPERIMENTS_DIR.relative_to(REPO_ROOT)})",
    )
    # Off by default: at the reported precision, rounding creates ties that the
    # counts would take as wins.
    parser.add_argument(
        "--ranks",
        action="store_true",
        help="add mean ranks per model",
    )
    parser.add_argument(
        "--wins",
        action="store_true",
        help="add the number of tasks won per model",
    )
    parser.add_argument(
        "--pairs-won",
        action="store_true",
        help="add the number of pairs won per model to the win matrices",
    )
    parser.add_argument(
        "--models-as-rows",
        action="store_true",
        help="models down and tasks across, with the task labels rotated",
    )
    args = parser.parse_args()

    # Check the run set is there. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if not (args.experiments / OUTPUTS_SUBDIR).is_dir():
        raise SystemExit(f"no run set at {args.experiments}")

    # Run command. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
