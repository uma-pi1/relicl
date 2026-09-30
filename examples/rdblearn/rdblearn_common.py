from importlib.metadata import version

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from .common import TASK_ID_COL

# Both examples join the task table to the entity table.
KEY_MAPPINGS = {"entity_id": "entities.entity_id"}


def _task_xy(task_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Get features and targets for task."""
    y = task_df["label"].astype(int)
    x = task_df.drop(columns=["label", TASK_ID_COL])
    return x, y


def _downstream_features(clf) -> list[str]:
    """Columns that actually reached the base estimator.

    Note that `feature_names_in_` is the raw input X, not the DFS output.
    """
    names = getattr(clf, "downstream_feature_columns_", None)
    return [str(n) for n in names] if names is not None else []


# noinspection unresolved-references
def _dfs_columns(rdb, x: pd.DataFrame, dfs_config) -> dict[str, int]:
    """Generated DFS columns, mapped to how many distinct values each takes.

    Recomputed because RDBLearn does not expose the features before it drops
    the constant ones. Takes the fitted classifier's own DFS config so it
    cannot drift from what was actually used.
    """
    import fastdfs

    x = x.copy()
    x["entity_id"] = x["entity_id"].astype(str)
    out = fastdfs.compute_dfs_features(
        rdb, x, key_mappings=KEY_MAPPINGS, cutoff_time_column="date", config=dfs_config
    )
    generated = [c for c in out.columns if c not in x.columns]
    return {c: int(out[c].nunique(dropna=False)) for c in generated}


# noinspection unresolved-references
def make_classifier(seed: int, **dfs_overrides):
    """RDBLearn with TabPFN at their defaults, apart from the seed and `dfs_overrides`."""
    # Imports that are not available in the RelICL environment.
    from rdblearn.constants import RDBLEARN_DEFAULT_CONFIG
    from rdblearn.estimator import RDBLearnClassifier
    from tabpfn import TabPFNClassifier

    # A `dfs` entry replaces the default one as a whole, so start from the default.
    dfs = {**RDBLEARN_DEFAULT_CONFIG["dfs"], **dfs_overrides}
    return RDBLearnClassifier(
        base_estimator=TabPFNClassifier(random_state=seed),
        config=dict(dfs=dfs, random_seed=seed),
    )


def fit_and_score(clf, rdb, task: pd.DataFrame, n_train: int) -> dict:
    """Fit on the first `n_train` task rows, score the rest, and return the metrics."""
    x_train, y_train = _task_xy(task.iloc[:n_train])
    x_test, y_test = _task_xy(task.iloc[n_train:])

    clf.fit(
        X=x_train,
        y=y_train,
        rdb=rdb,
        key_mappings=KEY_MAPPINGS,
        cutoff_time_column="date",
    )
    scores = clf.predict_proba(X=x_test, rdb=rdb)[:, 1]

    dfs_columns = _dfs_columns(rdb, x_train, clf.config.dfs)
    return dict(
        auroc=float(roc_auc_score(y_test, scores)),
        accuracy=float(accuracy_score(y_test, (scores >= 0.5).astype(int))),
        n_train=len(x_train),
        n_test=len(x_test),
        n_dfs_columns=len(dfs_columns),
        dfs_columns_n_distinct=dfs_columns,
        downstream_features=_downstream_features(clf),
        packages=dict(
            rdblearn=version("rdblearn"),
            fastdfs=version("fastdfs"),
            tabpfn=version("tabpfn"),
        ),
    )
