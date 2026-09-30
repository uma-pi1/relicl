import logging

import numpy as np
import pandas as pd

from relicl.model.embeddings import ContextRowEmbeddings
from relicl.timing import timed
from relicl.typing import TimingEvent

logger = logging.getLogger(__name__)


@timed(event_type=TimingEvent.fusion)
def fuse(
    df: pd.DataFrame, context_embs_list: list[ContextRowEmbeddings]
) -> pd.DataFrame:
    """
    Fuses the input dataframe with the provided context embeddings.
    """
    concatenated = _concat_embeddings(context_embs_list)

    # TabICL captures row embeddings in half precision (`relicl/model/relicl/
    # extractor.py:127`). Reducers that only slice (`first_k`, `random_k`) pass that
    # dtype through, as does `fusion.early.reduce_dim=false`, and pandas has no float16
    # kernel for `bfill`, which `_dummy_row` needs.
    if concatenated.dtype == np.float16:
        concatenated = concatenated.astype(np.float32)

    # Append the context embeddings columns to the input data frame.
    _, n_new_cols = concatenated.shape  # [n_rows, n_new_cols]
    new_cols = pd.DataFrame(
        concatenated,
        columns=np.array([f"_ctx_emb_{i}" for i in range(n_new_cols)]),
        index=df.index,
    )
    logger.debug(
        f"Fused context embeddings of {len(context_embs_list)} tables "
        f"into {n_new_cols} columns"
    )

    return pd.concat([df, new_cols], axis=1)


def _concat_embeddings(context_embs_list: list[ContextRowEmbeddings]) -> np.ndarray:
    """
    Converts the embeddings from the context tables into a single table. Does not drop
    any information.

    Each input context embedding has the shape: `[n_estimators, n_rows, embed_dim]`

    The output has the shape: `[n_rows, n_context_tables * n_estimators * embed_dim]`

    Note that the first dimension is a singleton dimension if dimensionality reduction
    was applied.
    """

    # embs shape: [n_estimators, n_rows, embed_dim]
    n_estimators = context_embs_list[0].embs.shape[0]
    n_rows = context_embs_list[0].embs.shape[1]
    embed_dim = context_embs_list[0].embs.shape[2]

    projections: list[np.ndarray] = []
    for i, context_embs in enumerate(context_embs_list):
        for estimator_idx in range(n_estimators):
            emb = context_embs.embs[estimator_idx]  # [n_rows, embed_dim]
            projections.append(emb)

    ret = np.concatenate(projections, axis=1)
    assert ret.shape == (n_rows, len(context_embs_list) * n_estimators * embed_dim)
    return ret
