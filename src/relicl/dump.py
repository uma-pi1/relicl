import logging
import os
import os.path as osp
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from relbench.base import TaskType

from relicl.config import RelICLConfig, resolve_tabicl_checkpoint
from relicl.model.predictor import PredictionTask
from relicl.typing import EvalMode

logger = logging.getLogger(__name__)

# Columns added to a dumped frame so that both splits and their targets survive the
# round trip through parquet.
SPLIT_COL_NAME = "__split"
TARGET_COL_NAME = "__target"

# Values of the split column.
TRAIN_SPLIT = "train"
EVAL_SPLIT = "eval"

# Where dumps are written to, relative to the run directory.
DUMP_DIR_NAME = "dumps"

# File listing all dumps of a run, one YAML flow mapping per line (as in `Trace`).
INDEX_FILENAME = "index.yaml"

# Characters that are unsafe in file names.
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._=-]+")


# Metadata. ############################################################################


@dataclass
class DumpMetadata:
    file: str

    # RelBench task the dump belongs to.
    db: str
    task: str
    task_type: TaskType

    # Which evaluation split and which (timestep, test batch) produced it.
    eval_mode: EvalMode
    label: str

    # Shape of the dumped frame, excluding the split and target columns.
    n_train: int
    n_eval: int
    n_features: int

    # Full config of the producing run. Repeated per dump so that a dump stays
    # self-describing when it is moved away from its run directory.
    config: dict[str, Any] = field(default_factory=dict)


# Public interface. ####################################################################


def dump_prediction_frames(
    task: PredictionTask,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> None:
    """Dumps the final feature tables of one prediction task."""
    config = RelICLConfig.instance()

    # Only validation timesteps are dumped. At cutoff t the training rows already cover
    # everything up to t, so fine-tuning on a test dump would leak the future into
    # every earlier test timestep.
    if not config.output.dump_final_table or task.eval_mode is not EvalMode.val:
        return

    # Both splits go into one file, tagged by `SPLIT_COL_NAME`.
    df = _combine(train_df, task.train_targets, eval_df, task.test_targets)

    outdir = osp.join(HydraConfig.get().runtime.output_dir, DUMP_DIR_NAME)
    os.makedirs(outdir, exist_ok=True)

    # `task.label` already identifies both the cutoff timestamp and the test batch.
    filename = f"{task.eval_mode}-{_sanitize(task.label)}.parquet"
    df.to_parquet(osp.join(outdir, filename), index=False)

    # A dump records the checkpoint that actually ran, so readers do not have to
    # resolve an unset `checkpoint_version` themselves.
    config_dump = cast(
        dict,
        OmegaConf.to_container(config, resolve=True, enum_to_str=True),  # type: ignore[call-overload]
    )
    config_dump["tabicl"]["checkpoint_version"] = resolve_tabicl_checkpoint(
        task.task_type
    )

    metadata = DumpMetadata(
        file=filename,
        db=config.relbench.db,
        task=config.relbench.task,
        task_type=task.task_type,
        eval_mode=task.eval_mode,
        label=task.label,
        n_train=int((df[SPLIT_COL_NAME] == TRAIN_SPLIT).sum()),
        n_eval=int((df[SPLIT_COL_NAME] == EVAL_SPLIT).sum()),
        n_features=len(df.columns) - 2,
        config=config_dump,
    )
    _append_to_index(outdir, metadata)

    logger.info(
        f"Dumped {filename}: {metadata.n_train} train + {metadata.n_eval} eval rows, "
        f"{metadata.n_features} features."
    )


# Utilities. ###########################################################################


def _sanitize(label: str) -> str:
    """Turns a task label into a file name component."""
    return _UNSAFE_CHARS.sub("_", label).strip("_")


def _combine(
    train_df: pd.DataFrame,
    train_targets: np.ndarray,
    eval_df: pd.DataFrame,
    eval_targets: np.ndarray,
) -> pd.DataFrame:
    """
    Combines the training and evaluation frames into a single dumpable frame, tagging
    each row with the split it came from and its target.
    """

    # Sanity check: Both frames feed the same model, so their columns must agree.
    assert list(train_df.columns) == list(eval_df.columns), (
        "Training and evaluation frames must have identical columns to be dumped."
    )

    parts: list[pd.DataFrame] = []
    for split, df, targets in (
        (TRAIN_SPLIT, train_df, train_targets),
        (EVAL_SPLIT, eval_df, eval_targets),
    ):
        # Sanity check: A length mismatch here would silently mislabel rows.
        assert len(df) == len(targets), (
            f"The {split} frame has {len(df)} rows but {len(targets)} targets."
        )

        part = df.reset_index(drop=True).copy()
        part[SPLIT_COL_NAME] = split
        part[TARGET_COL_NAME] = targets
        parts.append(part)

    combined = pd.concat(parts, ignore_index=True)

    # Parquet requires unique string column names.
    combined.columns = combined.columns.astype(str)
    duplicates = combined.columns[combined.columns.duplicated()].unique().tolist()
    assert len(duplicates) == 0, (
        f"Cannot dump frame with duplicate columns: {duplicates}"
    )

    return combined


def _plain_values(items: list[tuple[str, Any]]) -> dict[str, Any]:
    """Unwraps enum members so that the index stays readable with `yaml.safe_load`;
    otherwise they are written as Python object tags."""
    return {k: v.value if isinstance(v, Enum) else v for k, v in items}


def _append_to_index(outdir: str, metadata: DumpMetadata) -> None:
    """Appends one line describing a dump to the index file."""
    # One line per dump, so timesteps can be appended as the run proceeds.
    with open(osp.join(outdir, INDEX_FILENAME), "a+") as f:
        line = yaml.dump(
            asdict(metadata, dict_factory=_plain_values),
            width=float("inf"),
            default_flow_style=True,
            sort_keys=False,
        ).strip()
        f.write(line + "\n")
