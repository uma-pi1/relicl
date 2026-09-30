import numpy as np
from relbench.base import TaskType

from relicl.typing import TargetTransform


def apply_target_transform(
    targets: np.ndarray, transform: TargetTransform, task_type: TaskType
) -> np.ndarray:
    """Transforms regression targets before the backbone is fitted on them."""
    if transform is TargetTransform.none:
        return targets

    # Sanity check: only regression has a continuous target to reshape.
    assert task_type is TaskType.REGRESSION, (
        f"method.target_transform={transform} is regression-only, but the task is "
        f"{task_type}."
    )

    # Sanity check: log1p is undefined below -1.
    assert targets.min() >= 0.0, (
        f"method.target_transform={transform} needs non-negative targets, but the "
        f"smallest one is {targets.min()}."
    )

    return np.log1p(targets)


def invert_target_transform(
    preds: np.ndarray, transform: TargetTransform
) -> np.ndarray:
    """Maps predictions back into the original target space."""
    if transform is TargetTransform.none:
        return preds

    # The tabular model may predict slightly below zero in log space. Hence, clip the
    # output to keep the inverse from emitting a negative target.
    return np.clip(np.expm1(preds), 0.0, None)
