import pandas as pd

from paper.config import (
    DATASET_STATS_DIR,
    GENERATED_DIR,
    PERCENT_DECIMALS,
    RESULTS_FILE,
    TABLE_PREAMBLE,
    VALUE_DECIMALS,
)
from paper.tables import bold
from relicl.config import REPO_ROOT


def database_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per RelBench database."""
    df = df.sort_values("dataset")
    return pd.DataFrame(
        {
            bold("Dataset"): df["dataset"],
            bold("Tables"): df["num_tables"],
            bold("Rows"): [f"{n:,}" for n in df["num_rows"]],
            bold("Columns"): df["num_cols"],
        }
    )


def classification_table(df: pd.DataFrame, tasks: set[tuple[str, str]]) -> pd.DataFrame:
    """One row per classification task; class balance is the test split's."""
    rows = _task_frame(df, tasks)
    return pd.DataFrame(
        {
            bold("Dataset"): rows["dataset"],
            bold("Task"): rows["task"],
            bold("Train"): [f"{n:,.0f}" for n in rows["rows_train"]],
            bold("Val"): [f"{n:,.0f}" for n in rows["rows_val"]],
            bold("Test"): [f"{n:,.0f}" for n in rows["rows_test"]],
            bold("Pos. rate (\\%)"): [
                f"{p * 100:.{PERCENT_DECIMALS}f}" for p in rows["pos_rate"]
            ],
        }
    )


def regression_table(df: pd.DataFrame, tasks: set[tuple[str, str]]) -> pd.DataFrame:
    """One row per regression task; target mean/std are the test split's."""
    rows = _task_frame(df, tasks)
    mean, std = zip(*(_mean_std(m, s) for m, s in zip(rows["mean"], rows["std"])))
    return pd.DataFrame(
        {
            bold("Dataset"): rows["dataset"],
            bold("Task"): rows["task"],
            bold("Train"): [f"{n:,.0f}" for n in rows["rows_train"]],
            bold("Val"): [f"{n:,.0f}" for n in rows["rows_val"]],
            bold("Test"): [f"{n:,.0f}" for n in rows["rows_test"]],
            "target": mean,
            "sd": std,
        }
    )


def _task_frame(df: pd.DataFrame, tasks: set[tuple[str, str]]) -> pd.DataFrame:
    """One row per task actually in `tasks`: split row counts pivoted into
    columns, joined onto the test split's own row, since that is where the
    reported statistics (class balance, target mean/std) come from."""
    pivot = df.pivot(index=["dataset", "task"], columns="split", values="rows")
    test = df[df["split"] == "test"].set_index(["dataset", "task"])
    out = test.join(pivot.add_prefix("rows_"))
    out = out[out.index.isin(tasks)]
    return out.reset_index().sort_values(["dataset", "task"], ignore_index=True)


def _mean_std(mean: float, std: float) -> tuple[str, str]:
    return f"{mean:.{VALUE_DECIMALS}f}", f"{std:.{VALUE_DECIMALS}f}"


# Writing. #############################################################################


def write(
    table: pd.DataFrame,
    name: str,
    column_format: str,
    header: list[str] | None = None,
) -> None:
    """Write a bare `tabular` the paper inputs."""
    path = GENERATED_DIR / f"table-{name}.tex"
    latex = table.to_latex(
        index=False, escape=False, header=header is None, column_format=column_format
    )
    if header is not None:
        header_line = " & ".join(header) + r" \\"
        latex = latex.replace("\\toprule\n", f"\\toprule\n{header_line}\n", 1)
    path.write_text(TABLE_PREAMBLE + latex)
    print(f"wrote {path.relative_to(REPO_ROOT)} ({len(table)} rows)")


def write_all() -> None:
    """Build the appendix's RelBench dataset-stats tables from the CSVs in
    `DATASET_STATS_DIR`, filtered to the tasks in `RESULTS_FILE`."""
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(RESULTS_FILE, usecols=["db", "task"], low_memory=False)
    tasks = set(zip(results["db"], results["task"]))

    databases = pd.read_csv(DATASET_STATS_DIR / "databases.csv")
    write(database_table(databases), "dataset-stats", "lrrr")

    classification = pd.read_csv(DATASET_STATS_DIR / "classification.csv")
    write(classification_table(classification, tasks), "tasks-classification", "llrrrr")

    regression = pd.read_csv(DATASET_STATS_DIR / "regression.csv")
    write(
        regression_table(regression, tasks),
        "tasks-regression",
        "llrrrr@{ $\\pm$ }l",
        header=[
            bold("Dataset"),
            bold("Task"),
            bold("Train"),
            bold("Val"),
            bold("Test"),
            r"\multicolumn{2}{c}{" + bold("Target (mean $\\pm$ SD)") + "}",
        ],
    )
