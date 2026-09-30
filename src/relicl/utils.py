import logging
import os
import shutil
import subprocess
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Callable

import networkx as nx
import numpy as np
from hydra.core.hydra_config import HydraConfig
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.nn import Module

from relicl.config import REPO_ROOT
from relicl.schema import RDLTask, Ref, RefType
from relicl.schema.table import Table, shape_str

logger = logging.getLogger(__name__)

# Module hook context manager. #########################################################


# noinspection PyProtectedMember
@contextmanager
def attached_hook(module: Module, hook_fn: Callable):
    """
    Attaches a forward hook to the specified module.
    Cleanup removes the hook upon leaving context.
    """
    logger.debug(f"Attaching hook to module: {module._get_name()}")
    # Register hook.
    h = module.register_forward_hook(hook_fn, always_call=True)

    try:
        yield
    finally:
        # Cleanup: Remove hook.
        logger.debug(f"Removing hook from module: {module._get_name()}")
        h.remove()


# Print stats. #########################################################################


def compute_regression_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    print_metrics: bool = True,
) -> dict[str, float]:
    assert labels.shape == preds.shape

    residuals = preds - labels
    abs_errors = np.abs(residuals)

    # Compute metrics.
    # noinspection bad-argument-type
    metrics: dict[str, Any] = OrderedDict(
        mae=float(mean_absolute_error(labels, preds)),
        # Spread of the absolute error, and the standard error `mae` inherits from it.
        mae_var=float(abs_errors.var(ddof=1)),
        mae_std=float(abs_errors.std(ddof=1)),
        mae_sem=float(abs_errors.std(ddof=1) / np.sqrt(labels.size)),
        mse=float(mean_squared_error(labels, preds)),
        rmse=float(np.sqrt(mean_squared_error(labels, preds))),
        median_ae=float(median_absolute_error(labels, preds)),
        max_error=float(max_error(labels, preds)),
        r2=float(r2_score(labels, preds)),
        explained_variance=float(explained_variance_score(labels, preds)),
        mean_residual=float(residuals.mean()),
        median_residual=float(np.median(residuals)),
        spearman=_spearman(labels, preds),
        n=labels.size,
    )

    # Print metrics.
    if print_metrics:
        print("Regression metrics:")
        csv_header = ""
        csv_values = ""

        for key, value in metrics.items():
            isint = isinstance(value, int)
            if isint:
                print(f"  {key}: {value}")
                csv_values += f",{value}"
            else:
                print(f"  {key}: {value:<.6f}")
                csv_values += f",{value:<.6f}"

            csv_header += f",{key}"

        # Print CSV string for easy logging.
        print("\nCSV format:")
        print(csv_header[1:])
        print(csv_values[1:])

        # Ground truth statistics.
        print("\nGround truth statistics:")
        print(f"  Min:  {labels.min():<.6f}")
        print(f"  Max:  {labels.max():<.6f}")
        print(f"  Mean: {labels.mean():<.6f}")
        print(f"  Std:  {labels.std():<.6f}")
        print(f"  Head: {labels[:10]}")

        # Prediction statistics.
        print("\nPrediction statistics:")
        print(f"  Min:  {preds.min():<.6f}")
        print(f"  Max:  {preds.max():<.6f}")
        print(f"  Mean: {preds.mean():<.6f}")
        print(f"  Std:  {preds.std():<.6f}")
        print(f"  Head: {preds[:10]}")

        # Residual statistics.
        residuals = preds - labels
        print("\nResidual statistics:")
        print(f"  Min:  {residuals.min():<.6f}")
        print(f"  Max:  {residuals.max():<.6f}")
        print(f"  Mean: {residuals.mean():<.6f}")
        print(f"  Std:  {residuals.std():<.6f}")
        print(f"  Head: {residuals[:10]}")

    return metrics


def _distribution_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Min/max/mean/std plus a few quantiles, flattened into one prefixed dict."""
    p01, p25, p50, p75, p99 = np.percentile(values, [1, 25, 50, 75, 99])
    return {
        f"{prefix}_min": float(values.min()),
        f"{prefix}_p01": float(p01),
        f"{prefix}_p25": float(p25),
        f"{prefix}_p50": float(p50),
        f"{prefix}_p75": float(p75),
        f"{prefix}_p99": float(p99),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation via Pearson on ranks (ties broken arbitrarily)."""
    rank_a = np.argsort(np.argsort(a))
    rank_b = np.argsort(np.argsort(b))
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def compute_target_coverage(
    train_targets: np.ndarray,
    test_targets: np.ndarray,
    preds: np.ndarray,
) -> dict[str, float]:
    """
    Diagnostics for regression batches: how well the sampled training targets cover the
    test targets, and how skewed they are.
    """
    stats: dict[str, float] = dict(
        n_train=float(train_targets.size),
        n_test=float(test_targets.size),
    )
    stats |= _distribution_stats(train_targets, "train")
    stats |= _distribution_stats(test_targets, "test")
    stats |= _distribution_stats(preds, "pred")

    # Extrapolation: test targets the sampled training context cannot reach.
    train_min = float(train_targets.min())
    train_max = float(train_targets.max())
    below = float(np.mean(test_targets < train_min))
    above = float(np.mean(test_targets > train_max))
    stats |= dict(
        frac_test_below_train_min=below,
        frac_test_above_train_max=above,
        frac_test_outside_train_range=below + above,
    )

    # Flat-lining: predictions covering much less spread than the labels.
    test_spread = float(test_targets.max() - test_targets.min())
    pred_spread = float(preds.max() - preds.min())
    stats["pred_spread_ratio"] = (
        pred_spread / test_spread if test_spread > 0 else np.nan
    )

    # Shape of the training target distribution. `train_abs_z_max` is what reaches
    # TabICL's target-aware column embedder, which is an unbounded Linear(1, embed_dim)
    # for regression.
    std = float(train_targets.std())
    if std > 0:
        z = (train_targets - train_targets.mean()) / std
        stats |= dict(
            train_skew=float(np.mean(z**3)),
            train_excess_kurtosis=float(np.mean(z**4) - 3.0),
            train_abs_z_max=float(np.abs(z).max()),
        )

    # Rank agreement: stays high when only the scale of the predictions is off, which
    # points at the target transform rather than at the features.
    stats["spearman"] = _spearman(test_targets, preds)

    return stats


# noinspection DuplicatedCode
def compute_member_spread(member_metrics: list[dict[str, Any]]) -> dict[str, float]:
    """Spread of each metric across ensemble members."""
    if len(member_metrics) < 2:
        return {}

    # Focus only on the key metrics.
    keys = [
        k
        for k in ("mae", "r2", "auroc", "accuracy")
        if all(isinstance(m.get(k), (int, float)) for m in member_metrics)
    ]

    spread: dict[str, float] = {}
    for key in keys:
        values = np.asarray([float(m[key]) for m in member_metrics])
        spread[f"{key}_member_mean"] = float(values.mean())
        spread[f"{key}_member_var"] = float(values.var(ddof=1))
        spread[f"{key}_member_std"] = float(values.std(ddof=1))
        spread[f"{key}_member_sem"] = float(values.std(ddof=1) / np.sqrt(values.size))
    return spread


def compute_binary_classification_metrics(
    labels: np.ndarray, preds_proba: np.ndarray, print_metrics: bool = True
) -> dict[str, float]:
    assert preds_proba.shape[1] == 2
    pred = preds_proba.argmax(axis=1)

    # Compute metrics.
    cm = confusion_matrix(labels, pred)
    accuracy = float(accuracy_score(labels, pred))
    metrics: dict[str, Any] = OrderedDict(
        auroc=float(roc_auc_score(labels, preds_proba[:, 1])),
        ap=float(average_precision_score(labels, preds_proba[:, 1])),
        f1=float(f1_score(labels, pred)),
        precision=float(precision_score(labels, pred, zero_division=np.nan)),
        recall=float(recall_score(labels, pred, zero_division=np.nan)),
        accuracy=accuracy,
        # Binomial spread of the per-row correctness, and the standard error `accuracy`
        # inherits from it. AUROC has no comparable closed form; it needs a bootstrap.
        accuracy_var=float(accuracy * (1.0 - accuracy)),
        accuracy_std=float(np.sqrt(accuracy * (1.0 - accuracy))),
        accuracy_sem=float(np.sqrt(accuracy * (1.0 - accuracy) / labels.size)),
        confusion_matrix=cm.tolist(),  # TODO: Check whether this breaks loading in HPO (no longer floats only).
        n=labels.size,
        n_0=int(sum(labels == 0)),
        n_1=int(sum(labels == 1)),
    )

    # Print metrics.
    if print_metrics:
        print("Binary classification metrics:")
        csv_header = ""
        csv_values = ""
        for key, value in metrics.items():
            isint = isinstance(value, int)
            if isint:
                print(f"  {key}: {value}")
                csv_values += f",{value}"
            else:
                print(f"  {key}: {value:<.4f}")
                csv_values += f",{value:<.6f}"

            csv_header += f",{key}"

        # Print CSV string for easy logging.
        print("\nCSV format:")
        print(csv_header[1:])
        print(csv_values[1:])

        # Compute and print statistics on true labels.
        print("\nGround truth statistics:")
        print(f"  Min: {labels.min():<.4f}")
        print(f"  Max: {labels.max():<.4f}")
        print(f"  Mean: {labels.mean():<.4f}")
        print(f"  Std: {labels.std():<.4f}")
        # noinspection PyStringConversionWithoutDunderMethod
        print(f"  Head: {labels[:10]}")

        # Compute and print statistics on prediction (min, max, mean, std).
        print("\nPrediction statistics:")
        print(f"  Min: {pred.min():<.4f}")
        print(f"  Max: {pred.max():<.4f}")
        print(f"  Mean: {pred.mean():<.4f}")
        print(f"  Std: {pred.std():<.4f}")
        print(f"  Head: {pred[:10]}")

    return metrics


# noinspection DuplicatedCode
def print_multiclass_classification_metrics(
    labels: np.ndarray, preds_proba: np.ndarray
) -> None:
    """
    Print standard multiclass classification metrics and basic stats.

    Args:
        labels: (n_samples,) integer class labels.
        preds_proba: (n_samples, n_classes) predicted probabilities or scores.
    """
    # Predicted class labels.
    pred = preds_proba.argmax(axis=1)

    # Compute core metrics (macro average treats all classes equally).
    accuracy = accuracy_score(labels, pred)
    f1 = f1_score(labels, pred, average="macro")
    precision = precision_score(labels, pred, average="macro")
    recall = recall_score(labels, pred, average="macro")

    # Prepare one-hot encoding for probabilistic metrics.
    ground_truth_bin = label_binarize(labels, classes=np.unique(labels))
    # noinspection PyStringConversionWithoutDunderMethod
    print(f"Classes: {np.unique(labels)}")
    # noinspection PyStringConversionWithoutDunderMethod
    print(f"Class frequencies: {np.bincount(labels.astype(np.int16))}")

    # ROC AUC and Average Precision (macro averaged).
    auroc = roc_auc_score(
        ground_truth_bin, preds_proba, multi_class="ovr", average="macro"
    )
    apscore = average_precision_score(ground_truth_bin, preds_proba, average="macro")

    # Print metrics.
    print("Multiclass classification metrics:")
    print(f"  AUROC (macro): {auroc:<.4f}")
    print(f"  Accuracy: {accuracy:<.4f}")
    print(f"  F1-score (macro): {f1:<.4f}")
    print(f"  Precision (macro): {precision:<.4f}")
    print(f"  Recall (macro): {recall:<.4f}")
    print(f"  Average Precision Score (macro): {apscore:<.4f}")

    # Print ground truth stats.
    print("\nGround truth statistics:")
    print(f"  Min: {labels.min():<.4f}")
    print(f"  Max: {labels.max():<.4f}")
    print(f"  Mean: {labels.mean():<.4f}")
    print(f"  Std: {labels.std():<.4f}")
    # noinspection PyStringConversionWithoutDunderMethod
    print(f"  Head: {labels[:10]}")

    # Print prediction stats.
    print("\nPrediction statistics:")
    print(f"  Min: {pred.min():<.4f}")
    print(f"  Max: {pred.max():<.4f}")
    print(f"  Mean: {pred.mean():<.4f}")
    print(f"  Std: {pred.std():<.4f}")
    print(f"  Head: {pred[:10]}")


# Graph plotting. ######################################################################


def _graphviz_available() -> bool:
    # system graphviz binary
    # noinspection PyDeprecation
    if shutil.which("dot") is None:
        return False
    # python binding (pygraphviz) check
    try:
        import pygraphviz  # noqa: F401
    except ImportError:
        return False

    return True


def store_rdl_task_graph_to_disk(
    rdl_task: RDLTask,
    file_name: str | PdfPages,
    shapes: bool = False,
    title: str | None = None,
) -> None:
    store_schema_graph_to_disk(
        rdl_task.db.schema,
        file_name,
        rdl_task.db.tables,
        highlight_nodes=[rdl_task.task_table_name],
        shapes=shapes,
        title=title,
    )


def store_schema_graph_to_disk(
    graph: nx.MultiDiGraph,
    file_name: str | PdfPages,
    tables: dict[str, Table],
    directory: str = "graphs",
    *,
    shapes: bool = False,
    highlight_nodes=None,
    prog="neato",
    bend_edges=True,
    title=None,
) -> None:
    nx.set_node_attributes(
        graph,
        {
            name: f"{name}\n{shape_str(table.shape(eager=shapes))}"
            for name, table in tables.items()
        },
        "label",
    )

    store_graph_to_disk(
        graph,
        file_name,
        directory,
        highlight_nodes=highlight_nodes,
        prog=prog,
        bend_edges=bend_edges,
        title=title,
    )


def store_graph_to_disk(
    graph: nx.MultiDiGraph,
    file_name: str | PdfPages,
    directory: str = "graphs",
    *,
    highlight_nodes: list[str | int] | None = None,
    prog: str = "neato",
    bend_edges: bool = True,
    title: str | None = None,
) -> None:
    if not _graphviz_available():
        logger.warning("Graphviz not available, cannot save graph to disk.")
        return

    # Silence matplotlib noise.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    import matplotlib.pyplot as plt

    # Layout (graphviz handles directed graphs nicely).
    pos = nx.nx_agraph.graphviz_layout(graph, prog=prog)

    # Create figure.
    plt.figure(figsize=(20, 20))  # type: ignore
    plt.gca().axis("off")
    if title is not None:
        plt.title(title)

    # Draw nodes and labels.
    nx.draw_networkx_nodes(graph, pos, margins=0.2)
    if highlight_nodes is not None:
        nx.draw_networkx_nodes(
            graph, pos, nodelist=highlight_nodes, node_color="#b41f1f"
        )
    labels = {
        n: n if l == "NO_LABEL" else l
        for n, l in nx.get_node_attributes(graph, "label", "NO_LABEL").items()
    }
    nx.draw_networkx_labels(graph, pos, labels=labels)

    # Draw edges individually (fix for MultiDiGraph). ##################################
    edge_count: dict[tuple[str, str], int] = {}

    for u, v, k, data in graph.edges(keys=True, data=True):
        # Count how many parallel edges exist between u and v.
        edge_count.setdefault((u, v), 0)
        edge_count[(u, v)] += 1

        # Spread edges using curvature.
        idx = edge_count[(u, v)]
        total = graph.number_of_edges(u, v)

        # Center edges around 0 curvature.
        base_rad = 0.15 if bend_edges else 0  # minimum curvature
        if total > 1:
            rad = base_rad * (idx - (total + 1) / 2)
        else:
            rad = base_rad

        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=[(u, v)],
            connectionstyle=f"arc3,rad={rad}",
        )

        # Draw edge label.
        if data.get("label", None) is not None:
            label = data["label"]
            nx.draw_networkx_edge_labels(
                graph,
                pos,
                edge_labels={(u, v, k): label},
                connectionstyle=f"arc3,rad={rad}",
                font_size=8,
            )
        elif isinstance(data.get("ref", None), Ref):
            ref = data["ref"]
            label = ref.from_col_name
            if ref.from_col_name != ref.to_col_name:
                label += f"->{ref.to_col_name}"

            match ref.type:
                case RefType.many_to_one:
                    pass
                case RefType.one_to_many:
                    label += " (1:N)"
                case RefType.many_to_many:
                    label += " (N:M)"

            nx.draw_networkx_edge_labels(
                graph,
                pos,
                edge_labels={(u, v, k): label},
                connectionstyle=f"arc3,rad={rad}",
                font_size=8,
            )

    if isinstance(file_name, PdfPages):
        file_name.savefig(bbox_inches="tight")
    else:
        # Output directory.
        base_dir = os.path.join(HydraConfig.get().runtime.output_dir, directory)
        os.makedirs(base_dir, exist_ok=True)

        # Save figure.
        output_path = os.path.join(base_dir, file_name)
        plt.savefig(output_path, bbox_inches="tight")

    plt.close()


def get_git_revision_short_hash():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT
            )
            .strip()
            .decode()
        )
    except:
        return "<unknown>"
