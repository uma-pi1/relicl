import logging
from abc import ABC, abstractmethod

import numpy as np
from relbench.base import TaskType

from relicl.config import WithConfig
from relicl.model.extractor import Extractor
from relicl.model.prediction_data import PredictionData, PredictionDataGNN
from relicl.model.predictor import (
    PredictionTask,
    Predictor,
)
from relicl.progress import tqdm
from relicl.timing import timed
from relicl.trace import Trace
from relicl.typing import FusionType, InferenceType
from relicl.utils import (
    compute_binary_classification_metrics,
    compute_regression_metrics,
    compute_target_coverage,
)

logger = logging.getLogger(__name__)

# RelModel. ############################################################################


class RelModel(WithConfig, ABC):
    """Model to perform relational inference."""

    @property
    @abstractmethod
    def extractor(self) -> Extractor:
        """Returns an extractor that allows to extract row embeddings from a context
        table.

        """

    @property
    @abstractmethod
    def predictor(self) -> Predictor:
        """Returns a predictor that performs predictions on the task table using the
        row embeddings extracted from the context tables.

        """

    def predict(
        self,
        task: PredictionTask,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        match self.config.fusion.inference:
            case InferenceType.one_pass:
                return self._predict_one_pass(task)
            case InferenceType.tabular_gnn:
                return self._predict_tabular_gnn(task)

    @timed()
    def _predict_one_pass(
        self,
        task: PredictionTask,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Runs relational inference on the provided tables.

        The target column in the test table is ignored used during prediction (so can
        be arbitrary), but whatever values are in there are returned to facilitate
        evaluation.

        Returns:
        - preds_proba: [n_test, n_classes] = predicted class probabilities
        - targets_test: [n_test] = actual test targets originally in test_table
        - keys: [n_test] = corresponding keys

        """
        trace = Trace.instance()

        # precompute the tables
        task.db.compute()

        batched_preds_proba: list[np.ndarray] = []
        for i, batched_task in enumerate(
            task.iter_test_batches(self.config.method.test_batch_size)
        ):
            # Sample task.
            batched_task = batched_task.sample_train()

            data = PredictionData(batched_task)

            # Collect relevant rows for each table by traversing the inference graph.
            data.compute_relevant_rows()

            # on dry runs, don't run extractors and predictors
            if self.config.run.dry:
                continue

            if self.config.fusion.type is FusionType.none:
                # Predict from the rewritten, count-augmented task table alone.
                predictions = self.predictor.predict(batched_task, [])
            else:
                # Now compute row/key embeddings of all tables but the task table.
                data.extract_all_context_embeddings(self.extractor)

                # Predict and store results.
                predictions = self.predictor.predict(
                    batched_task,
                    data.get_context_embeddings(data.task_table_name, self.extractor),
                )
            batched_preds_proba.append(predictions)

            # Compute metrics of this batch.
            match task.task_type:
                case TaskType.BINARY_CLASSIFICATION:
                    batch_metrics = compute_binary_classification_metrics(
                        batched_task.test_targets, predictions, print_metrics=False
                    )
                case TaskType.REGRESSION:
                    batch_metrics = compute_regression_metrics(
                        batched_task.test_targets, predictions, print_metrics=False
                    )
                case _:
                    raise ValueError(f"Unsupported task type: {task.task_type}")

            # Trace results.
            trace.trace("batch_results", **batch_metrics)

            # Regression only: trace how well the training rows sampled for this batch
            # cover its test targets. Sampling is random and refit per batch, so the
            # coverage (and with it TabICL's target scaling) varies between batches.
            if task.task_type is TaskType.REGRESSION:
                trace.trace(
                    "batch_target_coverage",
                    **compute_target_coverage(
                        batched_task.train_targets,
                        batched_task.test_targets,
                        predictions,
                    ),
                )

        return (
            np.concat(batched_preds_proba) if not self.config.run.dry else np.array([]),
            task.test_targets,
            task.test_keys,
        )

    def _predict_tabular_gnn(
        self,
        task: PredictionTask,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # check if config looks ok
        if self.config.fusion.type is not FusionType.early:
            raise ValueError("Tabular-GNN inference requires fusion.type=early")
        if not self.config.fusion.allow_context_from_future:
            # not clear how to handle this
            raise ValueError(
                "Tabular-GNN inference requires fusion.allow_context_from_future=true"
            )

        logger.debug(
            "Tabular-GNN inference currently ignores all sampling and batching options."
        )

        # precompute the tables
        task.db.compute()
        data = PredictionDataGNN(task)

        if not self.config.run.dry:
            assert self.config.fusion.tabular_gnn is not None, (
                "fusion.tabular_gnn is unset"
            )
            num_layers = self.config.fusion.tabular_gnn.n_layers
            it = tqdm(range(num_layers), desc="Layer", position=1, leave=False)
            for layer in it:
                it.set_description(f"Layer {layer}")
                logger.debug(f"Layer {layer} of tabular GNN")

                data.update_row_embeddings(
                    self.extractor, remaining_layers=num_layers - (layer + 1)
                )

            # Predict and store results.
            preds_proba = self.predictor.predict(
                task,
                data.get_context_embeddings(data.task_table_name, self.extractor)
                if num_layers > 0
                else [],
            )

        return (
            preds_proba if not self.config.run.dry else np.array([]),
            task.test_targets,
            task.test_keys,
        )
