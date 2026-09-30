import logging

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from numpy import ndarray
from relbench.base import TaskType
from sklearn.base import RegressorMixin
from tabicl import TabICLClassifier
from torch import Tensor
from torch.nn import Module

from relicl.config import RelICLConfig, WithConfig
from relicl.dump import dump_prediction_frames
from relicl.model import (
    ContextRowEmbeddings,
    PredictionTask,
    Predictor,
)
from relicl.model.early_fusion import fuse
from relicl.model.embeddings import pool_context_embs_list
from relicl.model.target_transform import (
    apply_target_transform,
    invert_target_transform,
)
from relicl.timing import timed, timed_step
from relicl.typing import FusionType, ModelType, PoolingMethod, TimingEvent
from relicl.utils import attached_hook

from .tab_model import (
    TabFMModel,
    TabICLModel,
    TabPFNModel,
    assert_tabicl_dtypes,
    tabicl_normalize,
)

logger = logging.getLogger(__name__)


def _predict_regression(
    model: RegressorMixin, X: npt.NDArray | pd.DataFrame, config: RelICLConfig
) -> ndarray:
    if config.method.model is ModelType.tabicl:
        assert config.tabicl is not None, "tabicl is unset"
        preds = model.predict(X, output_type=config.tabicl.regression_output.value)
    else:
        preds = model.predict(X)

    return invert_target_transform(preds, config.method.target_transform)


########################################################################################
# TabICLPredictor. #####################################################################
########################################################################################


class RelICLPredictor(Predictor):
    def __init__(self, tab_model: TabICLModel | TabPFNModel | TabFMModel):
        super().__init__()
        self._tab_model = tab_model

    def _fit_targets(self, targets: ndarray, task_type: TaskType) -> ndarray:
        """The targets the backbone is actually fitted on."""
        return apply_target_transform(
            targets, self.config.method.target_transform, task_type
        )

    # Public methods. ##################################################################

    @timed()
    def predict(
        self, task: PredictionTask, context_embs_list: list[ContextRowEmbeddings]
    ) -> ndarray:
        train_df = task.train_df_vec
        test_df = task.test_df_vec

        if len(context_embs_list) == 0:
            return self._predict_without_context(
                train_df, task.train_targets, test_df, task.task_type
            )

        # add count columns when desired
        # TODO we many not need this anymore, really. The main difference is that this
        #  are sample counts, whereas the global count features are global counts.
        if self.config.rewrite.count_features:
            _, context_counts_df = pool_context_embs_list(
                None, context_embs_list, None, only_counts=True
            )
            assert context_counts_df.shape[0] == train_df.shape[0] + test_df.shape[0]
            train_counts_df = context_counts_df.iloc[: len(task.train_df_vec), :]
            test_counts_df = context_counts_df.iloc[len(task.train_df_vec) :, :]
            train_df = pd.concat(
                [
                    train_df.reset_index(drop=True),
                    train_counts_df.reset_index(drop=True),
                ],
                axis=1,
            )
            test_df = pd.concat(
                [test_df.reset_index(drop=True), test_counts_df.reset_index(drop=True)],
                axis=1,
            )

        match self.config.fusion.type:
            case FusionType.late:
                # Late fusion. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                assert self.config.method.model is ModelType.tabicl

                # Check that the config is set.
                late_config = self.config.fusion.late
                assert late_config is not None, "fusion.late is unset under late fusion"

                # Get model.
                match task.task_type:
                    case TaskType.BINARY_CLASSIFICATION:
                        tab_model = self._tab_model.new_classification_model(
                            finetuned=True
                        )
                    case TaskType.REGRESSION:
                        tab_model = self._tab_model.new_regression_model(finetuned=True)
                    case _:
                        raise ValueError(f"Unsupported task type: {task.task_type}")

                # Fit on training data.
                assert_tabicl_dtypes(train_df)
                tab_model.fit(
                    train_df, self._fit_targets(task.train_targets, task.task_type)
                )

                # Pool the context row embeddings.
                pooled_context_embs, _ = pool_context_embs_list(
                    late_config.task_pooling.to_numpy(),
                    context_embs_list,
                    (lambda X: tabicl_normalize(tab_model, X))
                    if late_config.task_normalize
                    else lambda X: X,
                )
                assert pooled_context_embs is not None

                # And scale them.
                pooled_context_embs.embs *= late_config.task_successor_weight

                # Then, predict on the test data, injecting all embeddings from the
                # context tables.
                return _TabICLEmbeddingInjector(
                    tab_model, pooled_context_embs.embs, task.task_type
                ).predict_with_injection_row_interactor(test_df)

            case FusionType.early:
                # Early fusion. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                return self._predict_with_context(
                    task,
                    train_df,
                    test_df,
                    context_embs_list,
                )

            case FusionType.none:
                raise ValueError(
                    "No fusion should have no context tables so this "
                    "error should not be reached (see above)."
                )

    # Protected methods. ###############################################################

    @timed(event_type=TimingEvent.tabicl)
    def _predict_without_context(
        self,
        train_df: pd.DataFrame,
        train_targets: ndarray,
        test_df: pd.DataFrame,
        task_type: TaskType,
    ) -> npt.NDArray:
        # Get model.
        match task_type:
            case TaskType.BINARY_CLASSIFICATION:
                model = self._tab_model.new_classification_model(finetuned=True)
            case TaskType.REGRESSION:
                model = self._tab_model.new_regression_model(finetuned=True)
            case _:
                raise ValueError(f"Unsupported task type: {task_type}")

        # Fit.
        if self.config.method.model is ModelType.tabicl:
            assert_tabicl_dtypes(train_df)
        model.fit(train_df, self._fit_targets(train_targets, task_type))

        # Predict.
        match task_type:
            case TaskType.BINARY_CLASSIFICATION:
                return model.predict_proba(test_df)
            case TaskType.REGRESSION:
                # noinspection bad-return
                return _predict_regression(model, test_df, self.config)
            case _:
                raise ValueError(f"Unsupported task type: {task_type}")

    @timed()
    def _predict_with_context(
        self,
        task: PredictionTask,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        context_embs_list: list[ContextRowEmbeddings],
    ) -> ndarray:
        task_type = task.task_type
        n_train = len(train_df)

        # Split context embeddings between train and test.
        train_context_embeddings_list: list[ContextRowEmbeddings]
        test_context_embeddings_list: list[ContextRowEmbeddings]
        split_context_embeddings = [ce.split_rows(n_train) for ce in context_embs_list]
        train_context_embeddings_list, test_context_embeddings_list = zip(  # type: ignore[assignment]
            *split_context_embeddings
        )

        # Early fusion on the training data.
        train_df = fuse(train_df, train_context_embeddings_list)

        # Early fusion on the test data.
        test_df = fuse(test_df, test_context_embeddings_list)

        # Remove columns if they are all-NaN in both train and test.
        all_nan_in_train = train_df.isna().all(axis=0)
        all_nan_in_test = test_df.isna().all(axis=0)
        cols_to_drop = all_nan_in_train & all_nan_in_test
        train_df = train_df.loc[:, ~cols_to_drop]
        test_df = test_df.loc[:, ~cols_to_drop]

        # This is the final feature table; dump it here so that fine-tuning can run
        # offline on exactly what the backbone sees.
        dump_prediction_frames(task, train_df, test_df)

        # All backbones are given the data frame itself. Converting to an array first
        # collapses the column dtypes into one, and a single string column then makes
        # TabICL treat every numeric column as categorical too.
        if self.config.method.model is ModelType.tabicl:
            assert_tabicl_dtypes(train_df)

        # Get model.
        match task_type:
            case TaskType.BINARY_CLASSIFICATION:
                model = self._tab_model.new_classification_model(finetuned=True)
            case TaskType.REGRESSION:
                model = self._tab_model.new_regression_model(finetuned=True)
            case _:
                raise ValueError(f"Unsupported task type: {task_type}")

        # Fit the TabICL or TabPFN model on the training data.
        model.fit(train_df, self._fit_targets(task.train_targets, task_type))

        # Predict.
        with timed_step("tabicl-in-predictor", event_type=TimingEvent.tabicl):
            match task_type:
                case TaskType.BINARY_CLASSIFICATION:
                    return model.predict_proba(test_df)
                case TaskType.REGRESSION:
                    # noinspection bad-return
                    return _predict_regression(model, test_df, self.config)
                case _:
                    raise ValueError(f"Unsupported task type: {task_type}")


########################################################################################
# EmbeddingInjector. ###################################################################
########################################################################################


class _TabICLEmbeddingInjector(WithConfig):
    def __init__(
        self,
        clf: TabICLClassifier,
        context_embeddings,
        task_type: TaskType,
    ) -> None:
        """
        Injects precomputed row embeddings into a specific module in TabICL
        (typically `col_embedder.in_linear`) during a forward pass.
        """
        super().__init__()

        # Store classifier and index in state.
        self.clf = clf
        self.context_embeddings = context_embeddings

        # Task type.
        self.task_type = task_type

        # Keep track of which tables (corresponding to ensemble members) have been used.
        self.ensemble_ptr = 0

        # Keep track of which rows have been used.
        self.row_ptr = 0

    @timed(event_type=TimingEvent.tabicl)
    def predict_with_injection_row_interactor(self, data) -> np.ndarray:
        """
        Perform a forward pass where the selected module's output
        is replaced with the injected embeddings.
        """
        with attached_hook(
            self.clf.model_.row_interactor.tf_row.blocks[-1],
            self._inject_hook_row_interactor,
        ):
            match self.task_type:
                case TaskType.BINARY_CLASSIFICATION:
                    return self.clf.predict_proba(data)
                case TaskType.REGRESSION:
                    return _predict_regression(self.clf, data, self.config)
                case _:
                    raise ValueError(f"Unsupported task type: {self.task_type}")

    def _inject_hook_row_interactor(
        self, _: Module, __: tuple[Tensor, ...], output: Tensor
    ) -> Tensor:
        """
        Replace the output with the precomputed embeddings.

        Args:
            output: Tensor of shape [B, T, H + C, E] where:
                    B is the number of tables,
                    T is the number of samples (rows), H is the number of features,
                    C is the number of class tokens, and E is the embedding dimension.

        """
        cloned_output = output.clone()

        # Validate data shapes.
        assert cloned_output.dim() == 4

        # Verify that the classifier is fitted (otherwise no `model_`).
        assert hasattr(self.clf, "model_")

        # Get another table's row embeddings (injection) sorted by foreign key.
        # The row embeddings are already flattened to shape [B, T, 4 * E]; this
        # needs to be undone so that the shapes match.
        injection = (
            torch.from_numpy(self.context_embeddings)
            .unflatten(-1, (4, 128))  # [B, T, 4 * E] -> [B, T, 4, E]
            .to(cloned_output.device, dtype=cloned_output.dtype)
        )

        # Get injection dimensions: This always corresponds to the full data
        # available.
        injection_n_rows = injection.size(1)

        # Get target dimensions: This corresponds to the current batch size,
        # determined # by the TabICL inference manager.
        target_n_ensemble_members = cloned_output.size(0)
        target_n_rows = cloned_output.size(1)

        # Check whether there are sufficiently many unused rows available.
        if self.row_ptr + target_n_rows <= injection_n_rows:
            row_start = self.row_ptr
            row_end = row_start + target_n_rows
        else:
            # If this is not the case, advance to the next ensemble member and reset
            # the row pointer.
            row_start = 0
            row_end = target_n_rows
            # TabICL may process multiple ensemble members in parallel, so advance
            # by that many.
            self.ensemble_ptr += target_n_ensemble_members
            logger.debug(
                f"Advancing to next ensemble member(s). New ensemble ptr: {self.ensemble_ptr}"
            )

        # Slicing injection to match target dimensions.
        injection = injection[
            self.ensemble_ptr : self.ensemble_ptr + target_n_ensemble_members,
            row_start:row_end,
        ]

        # Update row pointer for the next batch.
        self.row_ptr = row_end

        # Log injection details.
        logger.debug(
            f"Selected injection slice with ensemble indices "
            f"[{self.ensemble_ptr}:{self.ensemble_ptr + target_n_ensemble_members}] "
            f"and row indices [{row_start}:{row_end}]."
        )

        # Check that dimension match.
        injection_target = cloned_output[..., :4, :]
        logger.debug(
            f"Injecting row embeddings of shape {injection.shape} into output slice of shape {injection_target.shape}"
            f" (checksum: {torch.nansum(injection):.2f})"
        )
        assert injection_target.size() == injection.size(), (
            "An error here may also indicate that TabICL's inference manager uses "
            "batching, i.e., splitting the inference data into smaller tables "
            "(enable see debug logs for more details)."
        )

        if torch.isnan(injection).any():
            nan_percentage = torch.isnan(injection).float().mean().item() * 100
            logger.debug(
                f"Encountered NaN embeddings during injection. NaN percentage: {nan_percentage:.2f}%"
            )

        # Embedding injection is late fusion only.
        assert self.config.fusion.late is not None

        match self.config.fusion.late.task_pooling:
            case PoolingMethod.sum:
                # Handle NaNs in injection using zeros (no effect on the sum).
                if torch.isnan(injection).any():
                    injection = torch.where(
                        torch.isnan(injection),
                        torch.zeros_like(injection),
                        injection,
                    )

                # Replace the first C columns in the third dimension ("H+C") with the
                # sum of the current values and the injected embeddings.
                cloned_output[..., :4, :] = torch.sum(
                    torch.stack(
                        (
                            cloned_output[..., :4, :],  # [B, T, 4, E]
                            injection,
                        )
                    ),
                    dim=0,
                )
            case PoolingMethod.mean:
                # Handle NaNs by replacing them with the same value as the target,
                # hence the mean will not be affected.
                if torch.isnan(injection).any():
                    injection = torch.where(
                        torch.isnan(injection),
                        cloned_output[..., :4, :],
                        injection,
                    )

                cloned_output[..., :4, :] = torch.mean(
                    torch.stack(
                        (
                            cloned_output[..., :4, :],  # [B, T, 4, E]
                            injection,
                        )
                    ),
                    dim=0,
                )
            case PoolingMethod.max:
                # Handle NaNs by replacing them with very small values, so they
                # do not affect the max operation.
                if torch.isnan(injection).any():
                    injection = torch.where(
                        torch.isnan(injection),
                        torch.full_like(injection, float("-inf")),
                        injection,
                    )

                cloned_output[..., :4, :] = torch.maximum(
                    cloned_output[..., :4, :],  # [B, T, 4, E]
                    injection,
                )

        return cloned_output
