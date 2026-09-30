import logging

import numpy as np
import pandas as pd
import torch
from relbench.base import TaskType
from sklearn.base import ClassifierMixin, RegressorMixin
from tabfm import TabFMClassifier, TabFMRegressor
from tabicl import TabICLClassifier, TabICLRegressor
from torch import Tensor
from torch.nn import Module

from relicl.model import Extractor
from relicl.model.embeddings import RowEmbeddings
from relicl.timing import timed
from relicl.typing import TimingEvent
from relicl.utils import attached_hook

from .tab_model import (
    TabFMModel,
    TabICLModel,
    TabPFNModel,
    assert_tabicl_dtypes,
    tabicl_normalize,
)

########################################################################################
# Utils. ###############################################################################
########################################################################################

logger = logging.getLogger(__name__)


def _dummy_labels(df: pd.DataFrame, task_type: TaskType) -> np.ndarray:
    if task_type == TaskType.REGRESSION:
        # TabPFN breaks when there's an all-constant target.
        return np.arange(len(df), dtype=np.float64)
    return np.zeros(len(df))


class StopForward(Exception): ...


########################################################################################
# TabICL. ##############################################################################
########################################################################################


class TabICLExtractor(Extractor):
    def __init__(self, tab_model: TabICLModel):
        super().__init__()
        self._tabicl: TabICLClassifier | TabICLRegressor | None = None
        self._tab_model = tab_model

    def extract_without_context(
        self, df: pd.DataFrame, task_type: TaskType
    ) -> RowEmbeddings:
        # Get model.
        match task_type:
            case TaskType.BINARY_CLASSIFICATION:
                self._tabicl = self._tab_model.new_classification_model()
            case TaskType.REGRESSION:
                self._tabicl = self._tab_model.new_regression_model()
            case _:
                raise ValueError(f"Unsupported task type: {task_type}")

        # Fit. The frame is passed as-is so that TabICL types the columns individually.
        assert_tabicl_dtypes(df)
        self._tabicl.fit(df, _dummy_labels(df, task_type))
        assert self._tabicl is not None

        # Capture.
        return RowEmbeddings(_tabicl_capture(self._tabicl, df))

    def normalize(self, X: np.ndarray) -> np.ndarray:
        assert self._tabicl is not None
        return tabicl_normalize(self._tabicl, X)


# Capture function. ####################################################################


def _dummy_row(df: pd.DataFrame) -> pd.DataFrame:
    """One row of `df`, with no missing values, to predict on."""

    # Only the context embeddings are kept, so this row's values are thrown away. They
    # still have to be non-null: TabICL treats a column that is NaN throughout the frame
    # it predicts on as all-NaN and overwrites it with 0.0
    # (`tabicl/_sklearn/classifier.py:723-742`). On a single row that is just "this cell
    # is NaN", and writing a float into a string column raises.
    row = df.head(1).copy()
    na_cols = row.columns[row.iloc[0].isna()]
    if len(na_cols):
        # Backfilled from the rest of the frame; the vectorizer has already dropped
        # all-NaN columns, so every column has some value to take.
        row[na_cols] = df[na_cols].bfill().head(1)
    return row


@timed(event_type=TimingEvent.tabicl)
def _tabicl_capture(
    tabicl: TabICLRegressor | TabICLClassifier, df: pd.DataFrame
) -> np.ndarray:
    # Verify that the model is available.
    assert hasattr(tabicl, "model_"), (
        "Model may be missing when the classifier is not fitted yet."
    )

    # Variables for hook.
    embeddings: list[np.ndarray] = []
    context_size = len(df)
    n_expected_estimators = tabicl.n_estimators
    n_captured_estimators = 0

    def _store_row_embeddings_hook(
        _: Module, __: tuple[Tensor, ...], output: Tensor
    ) -> None:
        nonlocal embeddings, context_size, n_expected_estimators, n_captured_estimators

        # Validate input shapes.
        assert output.dim() == 3
        assert output.size(1) >= context_size
        assert output.size(2) == 512

        # Input logging.
        logger.debug("Capturing row embeddings:")
        logger.debug(f"Row embeddings shape (context + prediction): {output.shape}")

        # Move embeddings to CPU and convert to numpy.
        row_embeddings = output.cpu().numpy()

        # Store embeddings in state: Both the data from `fit` and `predict` are
        # embedded. Only keep the context embeddings (first `context_size` rows).
        new_embedding = row_embeddings[:, :context_size, :]
        embeddings.append(new_embedding)

        # The first dimension corresponds to the number of estimators in the
        # ensemble. Store this as a counter to determine when all estimators are
        # captured (see below).
        n_captured_estimators += new_embedding.shape[0]

        logger.debug(
            f"Stored row embeddings shape (context only): {new_embedding.shape}"
            f", iteration: {len(embeddings)}"
            f", total captured estimators: {n_captured_estimators}"
            f" (expected: {n_expected_estimators})"
        )

        # Terminate forward pass (for resource savings only).
        if n_captured_estimators >= n_expected_estimators:
            logger.debug("Captured all expected embeddings, aborting forward pass.")
            raise StopForward

    # Capture embeddings with a hook, terminate once embeddings are stored.
    with attached_hook(tabicl.model_.row_interactor, _store_row_embeddings_hook):
        try:
            tabicl.predict(_dummy_row(df))
        except StopForward:
            pass

    # Sanity check.
    assert len(embeddings) > 0, "No embeddings were captured."

    # Combine all captured embeddings along the estimator dimension.
    combined_embeddings = np.concatenate(embeddings, axis=0)

    # Get resulting shape.
    n_total_estimators, n_rows, embed_dim = combined_embeddings.shape

    # Increase the no. of estimators artificially if TabICL reduces it. The no. of
    # estimators is increased by duplicating existing estimators. This is likely better
    # than filling with NaNs because it causes no issues downstream.
    if n_total_estimators < n_expected_estimators:
        logger.debug(
            f"Number of captured estimators ({combined_embeddings.shape[0]}) does not"
            f" match expected number ({n_expected_estimators})."
        )

        # Get estimators to be used for filling.
        missing_estimators = n_expected_estimators - n_total_estimators
        artificial_estimator_embs = combined_embeddings[:missing_estimators]

        # Combine "true" estimators and "artificial" ones.
        combined_embeddings = np.concatenate(
            [combined_embeddings, artificial_estimator_embs], axis=0
        )

    return combined_embeddings


########################################################################################
# TabPFN. ##############################################################################
########################################################################################


class TabPFNExtractor(Extractor):
    def __init__(self, tab_model: TabPFNModel):
        super().__init__()
        self._tabpfn: ClassifierMixin | RegressorMixin | None = None
        self._tab_model = tab_model

    def extract_without_context(
        self, df: pd.DataFrame, task_type: TaskType
    ) -> RowEmbeddings:
        # Get model.
        match task_type:
            case TaskType.BINARY_CLASSIFICATION:
                self._tabpfn = self._tab_model.new_classification_model()
            case TaskType.REGRESSION:
                self._tabpfn = self._tab_model.new_regression_model()
            case _:
                raise ValueError(f"Unsupported task type: {task_type}")

        # Fit.
        self._tabpfn.fit(df, _dummy_labels(df, task_type))

        # Capture.
        assert self.config.tabpfn is not None, "tabpfn is unset"
        return RowEmbeddings(
            self._tabpfn.get_embeddings(df, data_source=self.config.tabpfn.data_source)
        )

    def normalize(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


########################################################################################
# TabFM. ###############################################################################
########################################################################################


class TabFMExtractor(Extractor):
    def __init__(self, tab_model: TabFMModel):
        super().__init__()
        self._tabfm: ClassifierMixin | RegressorMixin | None = None
        self._tab_model = tab_model

    def extract_without_context(
        self, df: pd.DataFrame, task_type: TaskType
    ) -> RowEmbeddings:
        # Get model.
        match task_type:
            case TaskType.BINARY_CLASSIFICATION:
                self._tabfm = self._tab_model.new_classification_model()
            case TaskType.REGRESSION:
                self._tabfm = self._tab_model.new_regression_model()
            case _:
                raise ValueError(f"Unsupported task type: {task_type}")

        # Fit.
        self._tabfm.fit(df, _dummy_labels(df, task_type))

        # Capture.
        return RowEmbeddings(_tabfm_capture(self._tabfm, df))

    def normalize(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


# Capture function. ####################################################################


@timed(event_type=TimingEvent.tabicl)
def _tabfm_capture(
    tabfm: TabFMClassifier | TabFMRegressor, df: pd.DataFrame
) -> np.ndarray:
    # first version, adapted from: https://github.com/vstenby/tfm-embeddings/blob/main/src/tfm_embeddings/adapters/tabfm.py

    embeddings: list[np.ndarray] = []
    n_rows = len(df)

    # row_interactor_2 produces the per-row representations consumed by
    # the in-context learning transformer.
    hook = tabfm.model.row_interactor_2.register_forward_hook(
        lambda module, args, output: embeddings.append(output.detach().float().cpu())
    )
    try:
        # predict() forwards [context; X] through the model per ensemble
        # batch; in the PyTorch path the encoded rows occupy the trailing
        # sequence positions of every forward pass.
        tabfm.predict(df)
    finally:
        hook.remove()

    members = torch.cat([reprs[:, -n_rows:, :] for reprs in embeddings], dim=0)  # type: ignore[misc]
    return members.numpy()
