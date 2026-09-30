import os
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path

import pandas as pd

from ..common import env_int

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Helpers. #############################################################################


class Task(StrEnum):
    # signals(a, b): `a` is 1, 0, -1 and `b` is `a` for the positive class ("positively
    # correlated"), `-a` for the negative one ("negatively correlated").
    signed = auto()
    # signals(a, b): `a` is 0, 1, 2 and `b` ascends with it for the positive
    # class ("positively correlated"), descends for the negative one.
    ordered = auto()


@dataclass
class Frames:
    entities: pd.DataFrame
    signals: pd.DataFrame
    task: pd.DataFrame


########################################################################################
# Adjust this. #########################################################################
########################################################################################

# Adjust the defaults here to make changes to the task.


@dataclass
class Config:
    # The TASK env var overrides this, so run_all.sh can sweep both variants
    # without rewriting this file.
    task: Task = Task(os.environ.get("TASK", Task.ordered))
    n_train: int = 1000
    n_test: int = 300
    rows_per_key: int = 3
    # The SEED env var overrides this. It seeds the data and both models.
    seed: int = env_int("SEED", 0)


########################################################################################
########################################################################################
########################################################################################

CONFIG = Config()
