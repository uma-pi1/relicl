import numpy as np
import pandas as pd

from ..common import TASK_ID_COL, balanced_labels
from .config import CONFIG, ChainTable, Config, Frames

# Helpers. #############################################################################


def _random_keys(rng: np.random.Generator, n: int) -> pd.Series:
    """A random permutation of 1..n, so that key values carry no signal."""
    return pd.Series(pd.array(rng.permutation(n) + 1, dtype="Int64"))


# Data frames. #########################################################################


def frames_from_config(config: Config) -> Frames:
    """A chain of `depth` one-to-one references with the label at its far end.

    task --entity_id--> entities <--entity_id-- hop2 <--hop2_id-- hop3 (bit)
    """
    # Sanity check: The chain needs at least the entity table.
    assert config.depth >= 1

    rng = np.random.default_rng(config.seed)
    n_keys = config.n_train + config.n_test
    labels = np.concatenate(
        [balanced_labels(rng, config.n_train), balanced_labels(rng, config.n_test)]
    )

    # Row j of every table belongs to task row j until the rows are shuffled below.
    chain: list[ChainTable] = []
    for i in range(1, config.depth + 1):
        name, pkey = ("entities", "entity_id") if i == 1 else (f"hop{i}", f"hop{i}_id")
        columns = {pkey: _random_keys(rng, n_keys)}
        if chain:
            # One-to-one reference into the previous table.
            columns[chain[-1].pkey] = chain[-1].df[chain[-1].pkey]
        if i == config.depth:
            # The only non-key column in the database.
            columns["bit"] = pd.Series(labels.astype(np.float64))
        chain.append(ChainTable(name=name, pkey=pkey, df=pd.DataFrame(columns)))

    task = pd.DataFrame(
        {
            # Surrogate primary key.
            TASK_ID_COL: pd.array(np.arange(n_keys), dtype="Int64"),
            # Foreign key.
            "entity_id": chain[0].df["entity_id"],
            # A constant dummy (required by RelICL).
            "date": pd.to_datetime([pd.Timestamp("2020-01-01")] * n_keys),
            "label": pd.array(labels, dtype="Int64"),
        }
    )

    # Shuffle the rows of the chain tables, so that row order carries no signal.
    for table in chain:
        table.df = table.df.iloc[rng.permutation(n_keys)].reset_index(drop=True)

    return Frames(chain=chain, task=task)


# Main method (for debugging only). ####################################################


if __name__ == "__main__":
    frames = frames_from_config(CONFIG)
    print(f"depth = {CONFIG.depth}")
    for table in frames.chain:
        print(f"--- {table.name} ---")
        print(table.df.head(3))
    print("--- task ---")
    print(frames.task.head(3))
