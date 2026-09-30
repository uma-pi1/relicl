import gc
import logging
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as hf_logging

from relicl.config import RelICLConfig
from relicl.model.dimred import fit_dim_reducer
from relicl.timing import timed
from relicl.typing import TimingEvent

logger = logging.getLogger(__name__)

# Config. ##############################################################################

# Cache of text -> embedding, per encoder. The same strings are vectorized again for
# every test batch and every evaluation timestep, so without this the encoder would run
# on near-identical inputs dozens of times per run.
_EMBEDDING_CACHE: dict[str, dict[str, npt.NDArray]] = {}

# Fitted dimensionality reducers, per text column.
_REDUCER_CACHE: dict[str, Any] = {}

# The encoder itself, which is kept across calls. Constructing it re-validates every
# checkpoint file against HuggingFace, so it is built once and only moved between
# devices after.
_ENCODER: SentenceTransformer | None = None

# Very noisy third-party loggers.
_NOISY_LOGGERS = ("httpx", "huggingface_hub", "sentence_transformers", "transformers")

# Public methods. ######################################################################


def is_prose_column(series: pd.Series) -> bool:
    """
    Estimate whether a string column holds free text rather than a categorical label.
    """
    text_config = RelICLConfig.instance().vectorizer.text

    # Only object/string columns can hold text. See the note in `DefaultVectorizer` on
    # why the string dtype is tested with `isinstance` rather than against "string".
    if not pd.api.types.is_object_dtype(series) and not isinstance(
        series.dtype, pd.StringDtype
    ):
        return False

    # Estimate only on a sample.
    sample = series.dropna().head(text_config.heuristic_sample_size)
    if sample.empty or not isinstance(sample.iloc[0], str):
        return False

    # Consider a column with handful of repeated labels as a category.
    uniques = sample.unique()
    if len(uniques) < text_config.min_unique:
        return False

    # Consider column as prose if it is at least at the threshold.
    word_counts = pd.Series(uniques).str.count(r"\s+") + 1  # count whitespace
    return bool(word_counts.mean() >= text_config.min_mean_words)


@timed(event_type=TimingEvent.preprocessing)
def embed_column(series: pd.Series) -> pd.DataFrame:
    """Reduced sentence embeddings for one text column, as `{col}__txt_{i}` columns."""

    text_config = RelICLConfig.instance().vectorizer.text

    # Embed. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Encode the distinct strings only.
    not_na = series.notna().to_numpy()
    present = pd.Index(series[not_na].astype(str))
    uniques = present.unique()
    embeddings = encode_texts(uniques.tolist())  # [n_unique, encoder_dim]

    # Reduce dimensionality. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Fit dimensionality reducer only once.
    reducer = _REDUCER_CACHE.get(str(series.name))
    if reducer is None:
        reducer = fit_dim_reducer(
            embeddings, text_config.reduction_method, text_config.reduction_dim
        )
        _REDUCER_CACHE[str(series.name)] = reducer

    # Reduce dimensionality.
    reduced = reducer.transform(embeddings).astype("float32")

    # Prepare output. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Map back onto the original rows. Missing text stays NaN.
    out = np.full((len(series), reduced.shape[1]), np.nan, dtype="float32")
    out[not_na] = reduced[uniques.get_indexer(present)]

    return pd.DataFrame(
        out,
        columns=[f"{series.name}__txt_{i}" for i in range(reduced.shape[1])],
        index=series.index,
    )


def encode_texts(texts: list[str]) -> npt.NDArray:
    text_config = RelICLConfig.instance().vectorizer.text
    cache = _EMBEDDING_CACHE.setdefault(text_config.model_name, {})

    # Get strings that have never been embedded.
    missing = [text for text in dict.fromkeys(texts) if text not in cache]

    # Compute missing embeddings.
    if missing:
        encoder = _acquire_encoder()
        vectors = encoder.encode(
            missing,
            batch_size=text_config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Store computed embeddings in cache.
        cache.update(zip(missing, vectors.astype("float32")))

        # Free GPU memory.
        del vectors
        _move_encoder_to_cpu()

    # Return embeddings.
    return np.stack([cache[text] for text in texts])


# Private methods. #####################################################################


def _acquire_encoder() -> "SentenceTransformer":
    """The shared encoder, on the compute device and ready to encode."""
    global _ENCODER

    text_config = RelICLConfig.instance().vectorizer.text

    # Load encoder once if not present.
    if _ENCODER is None:
        logger.info(f"Loading text encoder {text_config.model_name}.")

        # Make HF less verbose.
        _quiet_hub_logging()

        try:
            # Load from local cache.
            _ENCODER = SentenceTransformer(
                text_config.model_name, device="cuda", local_files_only=True
            )
        except OSError:
            # Download if not present.
            logger.info("Downloading text encoder model ...")
            _ENCODER = SentenceTransformer(text_config.model_name, device="cuda")
            logger.info("Downloaded text encoder model.")
    else:
        # Load from CPU otherwise.
        _ENCODER.to("cuda")

    return _ENCODER


def _move_encoder_to_cpu() -> None:
    if _ENCODER is not None:
        _ENCODER.to("cpu")

    gc.collect()
    torch.cuda.empty_cache()


def _quiet_hub_logging() -> None:
    # Set HF logging to WARNING.
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    # Disable progress bar.
    hf_logging.disable_progress_bar()
