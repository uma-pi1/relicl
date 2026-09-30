import logging
import warnings
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import DataFrame

from relicl.config import WithConfig
from relicl.model.dimred import fit_dim_reducers
from relicl.timing import timed, timed_step
from relicl.typing import TimingEvent

logger = logging.getLogger(__name__)


@dataclass
class Embeddings(WithConfig):
    """Embeddings of things (such as rows or keys)"""

    embs: np.ndarray
    """Embeddings tensor [n_estimators, n_embs, embedding_dim]."""

    is_reduced: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        WithConfig.__init__(self)

    @timed(event_type=TimingEvent.fusion)
    def reduce_dimensionality(self) -> None:
        if self.is_reduced:
            raise ValueError("dimensionality is already reduced")

        early_config = self.config.fusion.early

        # Sanity check: dimensionality reduction is part of early fusion only.
        assert early_config is not None, "fusion.early is unset"

        # Get the embeddings and their shape.
        embs = self.embs
        n_estimators, n_rows, embed_dim = embs.shape

        # Reshape them in case we want to reduce dimensionality across all estimators.
        if not early_config.reduce_per_estimator:
            # Use a simple trick and merge all estimators into a larger one:
            # (n_estimators, n_rows, embed_dim) -> (1, n_rows, n_estimators * embed_dim)
            #
            # Reorder embs so that the first feature of each estimator occurs first,
            # then the second feature, and so on.
            embs = np.transpose(embs, axes=(1, 2, 0))  # estimators go last
            embs = np.reshape(
                embs, shape=(n_rows, n_estimators * embed_dim)
            )  # the desired ordering
            embs = embs[np.newaxis, :, :]  # add the axis for a single estimator

            # Update shapes.
            n_estimators, n_rows, embed_dim = 1, n_rows, n_estimators * embed_dim

        # Fit the reducers.
        dim_reducers = fit_dim_reducers(embs)

        # Apply the reducers.
        with timed_step("run_dim_reducers", event_type=TimingEvent.fusion):
            new_embs_per_estimator: list[npt.NDArray] = []
            for estimator_idx in range(n_estimators):
                new_embs_per_estimator.append(
                    dim_reducers[estimator_idx].transform(embs[estimator_idx])
                )
            new_embs = np.stack(new_embs_per_estimator)

        # Verify output shape.
        assert new_embs.shape == (
            n_estimators,
            n_rows,
            early_config.reduction_dim,
        )

        # Update internal state.
        self.embs = new_embs
        self.is_reduced = True


@dataclass
class RowEmbeddings(Embeddings):
    """One embedding per row of a table"""


@dataclass
class ContextRowEmbeddings(RowEmbeddings):
    """For each row of a table, context row embeddings obtained from a successor table.
    Like RowEmbeddings, but additionally has a count of the number of successor rows
    being represented by this embedding.

    """

    counts: np.ndarray
    """Vector of counts [n_rows]"""

    label: str
    """Label describing this context row embedding. Should be unique; e.g., used as
    column name."""

    def split_rows(
        self, idx: int
    ) -> Tuple["ContextRowEmbeddings", "ContextRowEmbeddings"]:
        """
        Splits the embeddings and counts into two parts, one for the first `idx` rows
        and one for the remaining rows.

        Assumes shape: `[n_estimators, n_embs, embedding_dim]`
        """
        split1 = ContextRowEmbeddings(
            embs=self.embs[:, :idx, :],
            counts=self.counts[:idx],
            label=self.label,
        )
        split2 = ContextRowEmbeddings(
            embs=self.embs[:, idx:, :],
            counts=self.counts[idx:],
            label=self.label,
        )
        return split1, split2

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(embs={self.embs.shape})"


@dataclass
class KeyEmbeddings(Embeddings):
    """One embedding per key"""

    keys: np.ndarray
    """[nkeys] — the (unique) grouping key values"""

    counts: np.ndarray
    """[n_keys] — number of source rows per key"""


@timed(event_type=TimingEvent.fusion)
def pool_context_embs_list(
    pooling_fn,
    context_embs_list: list[ContextRowEmbeddings],
    normalize_fn,
    only_counts=False,
) -> tuple[None, DataFrame] | tuple[RowEmbeddings, DataFrame]:
    """Pools the embeddings of multiple successor tables.

    Returns the pooled embeddings and a data frame of counts, with one column per
    successor table.

    Extractors can use this to avoid having to deal with multiple context tables.
    """
    assert len(context_embs_list) > 0

    pooled_counts_df = pd.DataFrame({c.label: c.counts for c in context_embs_list})
    if only_counts:
        return None, pooled_counts_df

    with warnings.catch_warnings():  # type: ignore
        warnings.simplefilter("ignore", RuntimeWarning)

        # If there is no instance of a foreign key in any successor table, then numpy
        # raises a warning about that ("RuntimeWarning: Mean of empty slice").
        #
        # Occurs, for instance, when there is a user without posts or transactions. This
        # leads to `np.nanmean([np.nan, np.nan])`. In this case, returning NaN is
        # expected and represents "no available data" for this entity.

        pooled_embs = pooling_fn(np.array([c.embs for c in context_embs_list]), axis=0)

    pooled_embs = normalize_fn(pooled_embs)

    return RowEmbeddings(pooled_embs), pooled_counts_df
