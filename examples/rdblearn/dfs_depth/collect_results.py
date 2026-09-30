import pandas as pd

from ..common import PAPER_GENERATED, latex_auroc_table, load_results
from .config import RESULTS_DIR

TABLE_PATH = PAPER_GENERATED / "table-dfs-depth.tex"
# RelICL has no depth bound, as against the RDBLearn rows.
SYSTEM_NAMES = dict(
    tabicl="RelICL (TabICL, unbounded)", tabpfn="RelICL (TabPFN, unbounded)"
)
# Main text: RDBLearn at its stock depth bound only, next to RelICL.
MAIN_TABLE_PATH = PAPER_GENERATED / "table-dfs-depth-main.tex"
MAIN_METHODS = [
    "RDBLearn (depth bound 2)",
    SYSTEM_NAMES["tabicl"],
    SYSTEM_NAMES["tabpfn"],
]


def _method(row: pd.Series) -> str:
    """Display name, with RDBLearn's depth bound."""
    if row["system"] == "rdblearn":
        return f"RDBLearn (depth bound {int(row['dfs_max_depth'])})"
    return SYSTEM_NAMES[row["model"]]


if __name__ == "__main__":
    df = load_results(RESULTS_DIR)

    # Sanity check: The table has a single AUROC column, so all runs share one depth.
    assert df["depth"].nunique() == 1, df["depth"].unique()

    df["method"] = df.apply(_method, axis=1)

    # RDBLearn first, ordered by depth bound, then RelICL.
    df["_order"] = df["system"].eq("relicl").astype(int)
    df = df.sort_values(["_order", "method", "seed"])

    columns = ["method", "system", "depth", "seed", "auroc", "accuracy"]
    df[columns].to_csv(RESULTS_DIR / "results.csv", index=False)

    # Mean and SEM over seeds, per method.
    summary = (
        df.assign(auroc=df["auroc"] * 100)
        .groupby("method", sort=False)["auroc"]
        .agg(["mean", "sem", "count"])
    )
    module = f"{__package__}.collect_results"
    TABLE_PATH.write_text(latex_auroc_table(module, summary))
    main = summary.loc[MAIN_METHODS]
    MAIN_TABLE_PATH.write_text(latex_auroc_table(module, main, digits=1, bold=False))

    print(summary.to_string())
    print(f"wrote {TABLE_PATH} and {MAIN_TABLE_PATH}")
