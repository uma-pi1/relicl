import numpy as np
import pandas as pd

from ..common import TASK_ID_COL, balanced_labels
from .config import CONFIG, Config, Frames, Task

# Signals. #############################################################################


def _signal_columns(
    config: Config, n_keys: int, sign: np.ndarray
) -> dict[str, np.ndarray]:
    """The two value columns, one row block per entity.

    Rows per entity at `rows_per_key=3`, positive | negative:

        signed   (1,1) (0,0) (-1,-1) | (1,-1) (0,0) (-1,1)
        ordered  (0,-1) (1,0) (2,1)  | (0,1) (1,0) (2,-1)
    """
    match config.task:
        case Task.signed:
            # `a` is symmetric about 0 and `b` mirrors it.
            a = b = np.linspace(1.0, -1.0, config.rows_per_key)
        case Task.ordered:
            # `a` is a position index and `b` ascends with it.
            a, b = (
                np.arange(config.rows_per_key, dtype=np.float64),
                np.linspace(-1.0, 1.0, config.rows_per_key),
            )

    # Adding 0.0 turns -0.0 back into 0.0, so the negative class carries no
    # sign bit that would give it away on its own.
    return dict(a=np.tile(a, (n_keys, 1)), b=sign[:, None] * b + 0.0)


# Data frames. #########################################################################


def frames_from_config(config: Config) -> Frames:
    # Get RNG:
    rng = np.random.default_rng(config.seed)

    # Get the total number of keys.
    n_keys = config.n_train + config.n_test

    # Create entity IDs.
    entity_ids = np.arange(1, n_keys + 1, dtype=np.int64)

    # Get balanced train and test labels.
    labels = np.concatenate(
        [balanced_labels(rng, config.n_train), balanced_labels(rng, config.n_test)]
    )

    # Create signal columns.
    sign = np.where(labels == 1, 1.0, -1.0)
    columns = _signal_columns(config, n_keys, sign)

    # One signals row per (entity, position), entity by entity.
    signal_entity_ids = np.repeat(entity_ids, config.rows_per_key)
    signal_ids = np.arange(1, len(signal_entity_ids) + 1)

    # Create data frames (using pandas dtypes).
    entities = pd.DataFrame(dict(entity_id=pd.array(entity_ids, dtype="Int64")))
    signals = pd.DataFrame(
        dict(
            # A surrogate primary key (required by RDBLearn).
            signal_id=pd.array(signal_ids, dtype="Int64"),
            entity_id=pd.array(signal_entity_ids, dtype="Int64"),
            # `a` and `b` as two columns.
            **{name: values.ravel() for name, values in columns.items()},
        )
    )
    task = pd.DataFrame(
        {
            # Surrogate primary key.
            TASK_ID_COL: pd.array(np.arange(n_keys), dtype="Int64"),
            # Foreign key.
            "entity_id": pd.array(entity_ids, dtype="Int64"),
            # A constant dummy (required by RelICL).
            "date": pd.to_datetime([pd.Timestamp("2020-01-01")] * n_keys),
            "label": pd.array(labels, dtype="Int64"),
        }
    )
    return Frames(entities=entities, signals=signals, task=task)


# Main method (for debugging only). ####################################################


if __name__ == "__main__":
    frames = frames_from_config(CONFIG)
    print(f"task = {CONFIG.task}")
    print(frames.signals.head(6))
    print(frames.task.head(2))
