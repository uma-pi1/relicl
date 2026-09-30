import logging
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.utils.validation import check_is_fitted

from relicl.config import RelICLConfig
from relicl.rng import get_rng, random_state
from relicl.timing import timed
from relicl.typing import DimReductionMethod, TimingEvent

logger = logging.getLogger(__name__)


@timed(event_type=TimingEvent.fusion)
def fit_dim_reducers(embs: npt.NDArray) -> list[Any]:
    """Initializes dimensionality reducers for all estimators."""
    early_config = RelICLConfig.instance().fusion.early

    # Sanity check: dimensionality reduction is part of early fusion only.
    assert early_config is not None, "fusion.early is unset; early fusion not selected"

    table_dim_reducers: list[Any] = []

    # Get dimensions.
    n_estimators, _, __ = embs.shape  # [n_estimators, n_rows, embed_dim]

    for estimator_idx in range(n_estimators):
        embeddings = embs[estimator_idx]  # [n_rows, embed_dim]
        dim_reducer = fit_dim_reducer(
            embeddings, early_config.reduction_method, early_config.reduction_dim
        )
        table_dim_reducers.append(dim_reducer)

    return table_dim_reducers


def fit_dim_reducer(
    embeddings: npt.NDArray, method: DimReductionMethod, reduction_dim: int
) -> Any:
    """Fits a dimensionality reducer on a single `[n_rows, embed_dim]` matrix."""

    if np.isnan(embeddings).all():
        # If embeddings are all NaN, return a reducer that only changes the output
        # dimensionality.
        dim_reducer = _FirstKDimReducer(reduction_dim)
    else:
        match method:
            #
            # Random projections.
            #
            case DimReductionMethod.gaussian_random_projection:
                dim_reducer = Pipeline(
                    [
                        ("imputer", SimpleImputer()),
                        (
                            "reducer",
                            GaussianRandomProjection(
                                n_components=reduction_dim,
                                random_state=random_state(),
                            ),
                        ),
                    ]
                )

            case DimReductionMethod.sparse_random_projection:
                dim_reducer = Pipeline(
                    [
                        ("imputer", SimpleImputer()),
                        (
                            "reducer",
                            SparseRandomProjection(
                                n_components=reduction_dim,
                                random_state=random_state(),
                            ),
                        ),
                    ]
                )

            case DimReductionMethod.sign_random_projection:
                dim_reducer = Pipeline(
                    [
                        ("imputer", SimpleImputer()),
                        (
                            "reducer",
                            _SignRandomProjection(n_components=reduction_dim),
                        ),
                    ]
                )
            #
            # PCA.
            #
            case DimReductionMethod.pca:
                n_components = max(
                    1,
                    min(reduction_dim, embeddings.shape[0], embeddings.shape[1]),
                )

                if n_components < reduction_dim:
                    logger.info(
                        "n_components has been clamped to %d (was: %d) "
                        "and will be zero-padded",
                        n_components,
                        reduction_dim,
                    )

                dim_reducer = Pipeline(
                    [
                        ("imputer", SimpleImputer()),
                        (
                            "reducer",
                            PCA(
                                n_components=n_components,
                                random_state=random_state(),
                            ),
                        ),
                        ("padder", _ZeroPadder(reduction_dim)),
                    ]
                )

            case DimReductionMethod.pca_whiten:
                n_components = max(
                    1,
                    min(reduction_dim, embeddings.shape[0], embeddings.shape[1]),
                )

                if n_components < reduction_dim:
                    logger.info(
                        "n_components has been clamped to %d (was: %d) "
                        "and will be zero-padded",
                        n_components,
                        reduction_dim,
                    )

                dim_reducer = Pipeline(
                    [
                        ("imputer", SimpleImputer()),
                        (
                            "reducer",
                            PCA(
                                n_components=reduction_dim,
                                whiten=True,
                                random_state=random_state(),
                            ),
                        ),
                        ("padder", _ZeroPadder(reduction_dim)),
                    ]
                )
            #
            # SVD.
            #
            case DimReductionMethod.svd:
                dim_reducer = Pipeline(
                    [
                        ("imputer", SimpleImputer()),
                        (
                            "reducer",
                            TruncatedSVD(
                                n_components=reduction_dim, random_state=random_state()
                            ),
                        ),
                    ]
                )

            case DimReductionMethod.svd_whiten:
                dim_reducer = Pipeline(
                    [
                        ("imputer", SimpleImputer()),
                        ("reducer", _WhitenedTruncatedSVD(n_components=reduction_dim)),
                    ]
                )
            #
            # Entries selection.
            #
            case DimReductionMethod.random_k:
                dim_reducer = _RandomKDimReducer(reduction_dim)

            case DimReductionMethod.first_k:
                dim_reducer = _FirstKDimReducer(reduction_dim)

    dim_reducer.fit(embeddings)

    return dim_reducer


# Dimensionality reducers. #############################################################


class _RandomKDimReducer(BaseEstimator, TransformerMixin):
    """
    Reduces the dimensionality of the embeddings by selecting a random but fixed subset
    of embedding dimensions.
    """

    def __init__(self, n_components: int):
        self._n_components = n_components
        self._indices_: npt.NDArray | None = None

    def fit(self, X: npt.NDArray, _: npt.NDArray | None = None) -> "BaseEstimator":
        _, embed_dim = X.shape  # [n_rows, embed_dim]
        self._indices_ = get_rng().choice(embed_dim, self._n_components)
        return self

    def transform(self, X: npt.NDArray) -> npt.NDArray:
        check_is_fitted(self, "_indices_")
        return X[:, self._indices_]


class _FirstKDimReducer(BaseEstimator, TransformerMixin):
    """
    Reduces the dimensionality of the embeddings by selecting the first
    `n_components` embedding dimensions.
    """

    def __init__(self, n_components: int) -> None:
        self._n_components = n_components

    def fit(self, _: npt.NDArray, __: npt.NDArray | None = None) -> "_FirstKDimReducer":
        return self

    def transform(self, X: npt.NDArray) -> npt.NDArray:
        return X[:, : self._n_components]


class _WhitenedTruncatedSVD(BaseEstimator, TransformerMixin):
    """TruncatedSVD followed by whitening (divide by sqrt of explained variance)."""

    def __init__(self, n_components: int) -> None:
        self.svd_ = TruncatedSVD(n_components=n_components, random_state=random_state())
        self.scale_ = None

    def fit(
        self, X: npt.NDArray, _: npt.NDArray | None = None
    ) -> "_WhitenedTruncatedSVD":
        self.svd_.fit(X)
        self.scale_ = np.sqrt(self.svd_.explained_variance_)
        return self

    def transform(self, X: npt.NDArray) -> npt.NDArray:
        check_is_fitted(self, ["svd_", "scale_"])
        return self.svd_.transform(X) / self.scale_


class _SignRandomProjection(BaseEstimator, TransformerMixin):
    """
    Projects onto a matrix of +/- 1 / sqrt(n_components) values.
    """

    def __init__(self, n_components: int) -> None:
        self.components_ = None
        self._n_components = n_components

    def fit(
        self, X: npt.NDArray, _: npt.NDArray | None = None
    ) -> "_SignRandomProjection":
        n_features = X.shape[1]

        # Rademacher distribution: +/- 1, then scale
        self.components_ = get_rng().choice(
            [-1, 1], size=(n_features, self._n_components)
        ).astype(np.float32) / np.sqrt(self._n_components)

        return self

    def transform(self, X: npt.NDArray) -> npt.NDArray:
        check_is_fitted(self, "components_")
        return X @ self.components_


class _ZeroPadder(BaseEstimator, TransformerMixin):
    def __init__(self, target_dim: int) -> None:
        self._target_dim = target_dim

    def fit(self, X: npt.NDArray, y: npt.NDArray | None = None) -> "_ZeroPadder":
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: npt.NDArray) -> npt.NDArray:
        check_is_fitted(self, "n_features_in_")
        n_samples, n_features = X.shape

        if n_features >= self._target_dim:
            return X[:, : self._target_dim]

        pad_width = self._target_dim - n_features
        padding = np.zeros((n_samples, pad_width))
        return np.hstack([X, padding])
