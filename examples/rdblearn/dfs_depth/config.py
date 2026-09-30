from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..common import env_int

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Helpers. #############################################################################


@dataclass
class ChainTable:
    name: str
    pkey: str
    df: pd.DataFrame


@dataclass
class Frames:
    # Ordered by distance from the task table. Every table but the first references
    # its predecessor through a column named like the predecessor's primary key.
    chain: list[ChainTable]
    task: pd.DataFrame


########################################################################################
# Adjust this. #########################################################################
########################################################################################

# Adjust the defaults here to make changes to the task.


@dataclass
class Config:
    # References from the task table to the table holding the signal. The DEPTH env
    # var overrides this.
    depth: int = env_int("DEPTH", 3)
    # RDBLearn's DFS depth bound. Its own default is 2. The DFS_MAX_DEPTH env var
    # overrides this, so run_all.sh can sweep it.
    dfs_max_depth: int = env_int("DFS_MAX_DEPTH", 2)
    n_train: int = 1000
    n_test: int = 300
    # The SEED env var overrides this. It seeds the data and both models.
    seed: int = env_int("SEED", 0)


########################################################################################
########################################################################################
########################################################################################

CONFIG = Config()
