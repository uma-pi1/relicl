import pandas as pd

from .common import PAPER_GENERATED, latex_cell, latex_header
from .dfs_depth.collect_results import SYSTEM_NAMES as DEPTH_METHODS
from .dfs_depth.config import RESULTS_DIR as DEPTH_DIR
from .dfs_pathological.config import RESULTS_DIR as PATHOLOGICAL_DIR

TABLE_PATH = PAPER_GENERATED / "table-dfs-synthetic.tex"
SYSTEM_NAMES = dict(
    rdblearn="RDBLearn", tabicl="RelICL (TabICL)", tabpfn="RelICL (TabPFN)"
)
TASK_NAMES = dict(
    depth="Bounded depth",
    ordered="Cross-column, ordered",
    signed="Cross-column, signed",
)

# RDBLearn's stock depth bound, the only RDBLearn run shown in the combined table.
STOCK_DEPTH_METHOD = "RDBLearn (depth bound 2)"


def _load() -> pd.DataFrame:
    """Per-seed AUROC of both examples, as columns task, system, seed, auroc."""
    # Written by the `collect_results` module of each example.
    depth = pd.read_csv(DEPTH_DIR / "results.csv")
    pathological = pd.read_csv(PATHOLOGICAL_DIR / "results.csv")

    # Map the depth methods to systems, dropping RDBLearn at non-stock bounds. The
    # RelICL methods carry the names the depth example gives them.
    depth_systems = {
        STOCK_DEPTH_METHOD: "rdblearn",
        DEPTH_METHODS["tabicl"]: "tabicl",
        DEPTH_METHODS["tabpfn"]: "tabpfn",
    }
    depth = depth[depth["method"].isin(depth_systems)]
    depth = depth.assign(task="depth", system=depth["method"].map(depth_systems))

    # Sanity check: Every system has seeds on the stock depth bound.
    assert set(depth["system"]) == set(SYSTEM_NAMES), set(depth["system"])

    columns = ["task", "system", "seed", "auroc"]
    return pd.concat([depth[columns], pathological[columns]], ignore_index=True)


def _latex(mean: pd.DataFrame, sem: pd.DataFrame) -> str:
    """Tasks as rows and systems as columns, bolding the best mean per row."""
    cols = list(SYSTEM_NAMES)
    lines = [
        latex_header(f"{__package__}.collect_synthetic"),
        "\\begin{tabular}{l" + "r" * len(cols) + "}",
        "\\toprule",
        "\\textbf{Task} & "
        + " & ".join(f"\\textbf{{{SYSTEM_NAMES[c]}}}" for c in cols)
        + " \\\\",
        "\\midrule",
    ]
    for task in TASK_NAMES:
        row_mean = mean.loc[task, cols].to_numpy(dtype=float)
        row_sem = sem.loc[task, cols].to_numpy(dtype=float)
        best = row_mean.max()
        cells = [latex_cell(m, s, m == best) for m, s in zip(row_mean, row_sem)]
        lines.append(f"{TASK_NAMES[task]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    df = _load()

    # Mean and SEM over seeds, per task and system.
    scaled = df.assign(auroc=df["auroc"] * 100)
    grouped = scaled.groupby(["task", "system"])["auroc"]
    mean = grouped.mean().unstack()
    sem = grouped.sem().unstack()
    TABLE_PATH.write_text(_latex(mean, sem))

    print(grouped.agg(["mean", "sem", "count"]).to_string())
    print(f"wrote {TABLE_PATH}")
