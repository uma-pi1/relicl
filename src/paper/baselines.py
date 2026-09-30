from pathlib import Path

import pandas as pd

from paper.config import BASELINE_COLUMNS, BASELINES_FILE, MODEL_ORDER


def load(path: Path = BASELINES_FILE) -> pd.DataFrame:
    """The literature store, validated against the schema the tables assume."""
    df = pd.read_csv(path, dtype=dict(note=str)).fillna(dict(note=""))
    assert list(df.columns) == BASELINE_COLUMNS, (
        f"{path}: expected columns {BASELINE_COLUMNS}, got {list(df.columns)}"
    )

    # Sanity check: a duplicate row means two numbers claim the same cell.
    keys = ["model", "db", "task", "split", "metric"]
    duplicated = df[df.duplicated(keys, keep=False)]
    assert duplicated.empty, f"{path}: duplicate rows\n{duplicated}"

    return df


def pivot(df: pd.DataFrame, split: str, metric: str) -> pd.DataFrame:
    """One row per task, one column per model, for the models the paper reports."""
    subset = df[(df["split"] == split) & (df["metric"] == metric)]
    wide = subset.pivot(index=["db", "task"], columns="model", values="value")
    return wide.reindex(columns=[m for m in MODEL_ORDER if m in wide.columns])


def kinds(df: pd.DataFrame) -> dict[str, str]:
    """The kind of every published model, which the table groups its columns by."""
    per_model = df.groupby("model")["kind"].unique()

    # Sanity check: a model of two kinds cannot sit under one spanning label.
    assert all(len(k) == 1 for k in per_model), f"model of several kinds\n{per_model}"

    return {str(model): kind[0] for model, kind in per_model.items()}
