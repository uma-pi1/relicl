import os.path as osp
from typing import cast

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from relbench.base import TaskType
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.compose import make_column_selector
from tabfm import TabFMClassifier, TabFMRegressor
from tabicl import TabICLClassifier, TabICLRegressor
from tabpfn import TabPFNClassifier, TabPFNRegressor

from relicl.config import resolve_tabicl_checkpoint
from relicl.model import TabModel
from relicl.rng import random_state
from relicl.timing import timed
from relicl.typing import TimingEvent

########################################################################################
# TabICL. ##############################################################################
########################################################################################


class TabICLModel(TabModel):
    def new_classification_model(self, finetuned: bool = False) -> TabICLClassifier:
        return TabICLClassifier(
            random_state=random_state(),
            **self._kwargs(finetuned, TaskType.BINARY_CLASSIFICATION),
        )

    def new_regression_model(self, finetuned: bool = False) -> TabICLRegressor:
        return TabICLRegressor(
            random_state=random_state(),
            **self._kwargs(finetuned, TaskType.REGRESSION),
        )

    def _kwargs(self, finetuned: bool, task_type: TaskType) -> dict:
        """
        Turns the TabICL config into constructor arguments.

        A fine-tuned checkpoint is only ever handed to the predictor, because currently,
        only the predictor is fine-tuned on fixed outputs from the extractors.
        """
        assert self.config.tabicl is not None, "tabicl is unset"
        kwargs = cast(dict, OmegaConf.to_container(self.config.tabicl))

        # `regression_output` is a `predict()` argument, not a constructor one.
        del kwargs["regression_output"]

        # Get either configured or pre-defined checkpoint for the current task type.
        kwargs["checkpoint_version"] = resolve_tabicl_checkpoint(task_type)

        model_path = kwargs["model_path"]

        if not finetuned or model_path is None:
            kwargs["model_path"] = None
            return kwargs

        # Sanity check: When the file is missing, TabICL quietly downloads a pretrained
        # checkpoint to that path instead of failing, which looks like a working
        # fine-tuned run.
        assert osp.isfile(model_path), (
            f"No checkpoint at tabicl.model_path: {model_path}"
        )

        return kwargs


# Feature table check. #################################################################

# Dtypes that TabICL's `TransformToNumerical` selects columns by.
_CATEGORICAL_DTYPES = ["string", "object", "category", "boolean"]


def assert_tabicl_dtypes(df: pd.DataFrame) -> None:
    """
    Sanity check: every column of the final feature table has a dtype TabICL keeps.
    """
    kept = set(make_column_selector(dtype_include=_CATEGORICAL_DTYPES)(df))
    kept |= set(make_column_selector(dtype_include="number")(df))

    dropped = [str(col) for col in df.columns if col not in kept]
    assert not dropped, (
        f"TabICL would drop {len(dropped)} of {len(df.columns)} columns because their "
        f"dtype is neither categorical nor numeric: {dropped[:10]}"
    )


# Normalize function. ##################################################################


@timed(event_type=TimingEvent.tabicl)
def tabicl_normalize(
    tabicl: TabICLClassifier | TabICLRegressor, X: np.ndarray
) -> np.ndarray:
    layer_norm = tabicl.model_.row_interactor.out_ln
    with torch.no_grad():
        # Convert to tensor and match device and dtype of LN.
        device = next(tabicl.model_.row_interactor.out_ln.parameters()).device
        dtype = next(tabicl.model_.row_interactor.out_ln.parameters()).dtype
        X_t = torch.from_numpy(X).to(device=device, dtype=dtype)

        if X_t.shape[-1] == 512:
            # Unflatten to 4 x 128 -> LN -> flatten to 512.
            X_t = X_t.unflatten(-1, (4, 128))
            out = layer_norm(X_t)
            out = out.flatten(start_dim=-2)

        elif X_t.shape[-1] == 128:
            # Apply layer norm directly.
            out = layer_norm(X_t)

        else:
            raise ValueError(f"Unexpected input shape: {X_t.shape}")

        return out.cpu().numpy()


########################################################################################
# TabPFN. ##############################################################################
########################################################################################


class TabPFNModel(TabModel):
    def new_classification_model(self, finetuned: bool = False) -> ClassifierMixin:
        assert self.config.tabpfn is not None, "tabpfn is unset"
        tabpfn_config = cast(dict, OmegaConf.to_container(self.config.tabpfn))
        del tabpfn_config["data_source"]
        return TabPFNClassifier(random_state=random_state(), **tabpfn_config)

    def new_regression_model(self, finetuned: bool = False) -> RegressorMixin:
        assert self.config.tabpfn is not None, "tabpfn is unset"
        tabpfn_config = cast(dict, OmegaConf.to_container(self.config.tabpfn))
        del tabpfn_config["data_source"]
        return TabPFNRegressor(random_state=random_state(), **tabpfn_config)


########################################################################################
# TabFM. ###############################################################################
########################################################################################


class TabFMModel(TabModel):
    def __init__(self):
        super().__init__()
        from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0

        self.model = tabfm_v1_0_0.load(device="cuda")

    def new_classification_model(self, finetuned: bool = False) -> ClassifierMixin:
        assert self.config.tabfm is not None, "tabfm is unset"
        tabfm_config = cast(dict, OmegaConf.to_container(self.config.tabfm))
        return TabFMClassifier(
            model=self.model, random_state=random_state(), **tabfm_config
        )

    def new_regression_model(self, finetuned: bool = False) -> RegressorMixin:
        assert self.config.tabfm is not None, "tabfm is unset"
        tabfm_config = cast(dict, OmegaConf.to_container(self.config.tabfm))
        return TabFMRegressor(
            model=self.model, random_state=random_state(), **tabfm_config
        )
