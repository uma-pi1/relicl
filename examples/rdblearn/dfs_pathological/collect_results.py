import pandas as pd

from ..common import (
    PAPER_GENERATED,
    latex_auroc_table,
    latex_cell,
    latex_header,
    load_results,
)
from .config import RESULTS_DIR

TABLE_PATH = PAPER_GENERATED / "table-dfs-pathological.tex"
# Main text: the ordered variant only, transposed to one row per system.
MAIN_TABLE_PATH = PAPER_GENERATED / "table-dfs-pathological-main.tex"
MAIN_TASK = "ordered"
SYSTEM_NAMES = {
    "rdblearn": "RDBLearn",
    "tabicl": "RelICL (TabICL)",
    "tabpfn": "RelICL (TabPFN)",
}


def _latex(mean: pd.DataFrame, sem: pd.DataFrame) -> str:
    """Task variants as rows and systems as columns, bolding the best mean per row."""
    cols = [c for c in SYSTEM_NAMES if c in mean.columns]
    lines = [
        latex_header(f"{__package__}.collect_results"),
        "\\begin{tabular}{l" + "r" * len(cols) + "}",
        "\\toprule",
        "\\textbf{Task} & "
        + " & ".join(f"\\textbf{{{SYSTEM_NAMES[c]}}}" for c in cols)
        + " \\\\",
        "\\midrule",
    ]
    for task in mean.index:
        row_mean = mean[cols].loc[[task]].to_numpy(dtype=float)[0]
        row_sem = sem[cols].loc[[task]].to_numpy(dtype=float)[0]
        best = row_mean.max()
        cells = [latex_cell(m, v, m == best) for m, v in zip(row_mean, row_sem)]
        lines.append(f"{task} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    df = load_results(RESULTS_DIR)

    # RelICL rows carry `model`; RDBLearn rows carry `system`.
    df["system"] = df["system"].fillna(df["model"])

    columns = ["task", "system", "seed", "auroc", "accuracy"]
    df = df.sort_values(["task", "system", "seed"])
    df[columns].to_csv(RESULTS_DIR / "results.csv", index=False)

    # Mean and SEM over seeds, per task variant and system.
    scaled = df.assign(auroc=df["auroc"] * 100)
    mean = scaled.pivot_table(
        index="task", columns="system", values="auroc", aggfunc="mean"
    )
    sem = scaled.pivot_table(
        index="task", columns="system", values="auroc", aggfunc="sem"
    ).reindex_like(mean)
    TABLE_PATH.write_text(_latex(mean, sem))
    main = (
        scaled[scaled["task"] == MAIN_TASK]
        .groupby("system")["auroc"]
        .agg(["mean", "sem"])
        .loc[list(SYSTEM_NAMES)]
        .rename(index=SYSTEM_NAMES)
    )
    MAIN_TABLE_PATH.write_text(
        latex_auroc_table(f"{__package__}.collect_results", main, digits=1, bold=False)
    )

    summary = scaled.groupby(["task", "system"])["auroc"].agg(["mean", "sem", "count"])
    print(summary.to_string())
    print(f"wrote {TABLE_PATH} and {MAIN_TABLE_PATH}")
