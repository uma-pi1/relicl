import logging
import os
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd
import yaml
from hydra.core.hydra_config import HydraConfig
from matplotlib.backends.backend_pdf import PdfPages
from relbench.base import TaskType

from relicl import rng
from relicl.config import WithConfig
from relicl.ensemble import Predictions
from relicl.model import PredictionTask
from relicl.model.relicl import RelICLModel
from relicl.progress import tqdm
from relicl.schema import RDLTask
from relicl.timing import timed, timed_step
from relicl.trace import Trace
from relicl.typing import EvalMode
from relicl.utils import (
    compute_binary_classification_metrics,
    compute_regression_metrics,
    store_graph_to_disk,
)

########################################################################################
# Utilities. ###########################################################################
########################################################################################

logger = logging.getLogger(__name__)
yaml.add_representer(
    OrderedDict,
    lambda dumper, data: dumper.represent_mapping(
        "tag:yaml.org,2002:map", data.items()
    ),
)


########################################################################################
# Class: RDLRunner. ####################################################################
########################################################################################


# One independent RNG stream per evaluation split, so that running one split does not
# shift the random draws of the other.
_RNG_STREAM_KEY = {EvalMode.val: 0, EvalMode.test: 1}


class RDLRunner(WithConfig):
    """Runs and evaluates an RDLTask."""

    def __init__(self) -> None:
        super().__init__()
        self._relicl_model = RelICLModel()

    def run(self, rdl_task: RDLTask) -> dict[str, dict[str, float]]:
        if not self.config.run.val and not self.config.run.test:
            raise ValueError("Neither run.val nor run.test set")

        val_metrics = (
            self._run_eval(rdl_task, EvalMode.val) if self.config.run.val else {}
        )
        test_metrics = (
            self._run_eval(rdl_task, EvalMode.test) if self.config.run.test else {}
        )
        return dict(val=val_metrics, test=test_metrics)

    @timed()
    def _run_eval(self, rdl_task: RDLTask, mode: EvalMode) -> dict[str, float]:
        """Run an RDL task and log evaluation metrics."""

        # Start this split on its own RNG stream.
        rng.set_stream(_RNG_STREAM_KEY[mode])

        # Get tracing instance.
        trace = Trace.instance()

        # Collect results per batch.
        collected_preds_proba: list[np.ndarray] = []
        collected_labels: list[np.ndarray] = []
        collected_keys: list[np.ndarray] = []
        collected_cutoffs: list[np.ndarray] = []

        # Determine timestamps to use.
        match mode:
            case EvalMode.val:
                timestamps = rdl_task.val_timestamps
            case EvalMode.test:
                timestamps = rdl_task.test_timestamps

        timestep_iterator = tqdm(timestamps, desc="Iterating over test times")
        for cutoff_timestamp in timestep_iterator:
            timestep_iterator.set_description_str(f"Time ({mode}) {cutoff_timestamp}")

            # filter DB of task
            cutoff_rdl_task = rdl_task.cutoff(cutoff_timestamp)

            # Filter eval items.
            match mode:
                case EvalMode.val:
                    eval_table = cutoff_rdl_task.val_table(cutoff_timestamp)
                case EvalMode.test:
                    eval_table = cutoff_rdl_task.test_table(cutoff_timestamp)

            # And train items
            train_table = cutoff_rdl_task.train_table(include_val=mode == EvalMode.test)

            # export query graphs
            if self.config.output.graphs:

                def quote(_name: str) -> str:
                    return _name.replace("_", r"\_").replace("#", r"\#")

                outdir = os.path.join(
                    HydraConfig.get().runtime.output_dir, "graphs/query"
                )
                os.makedirs(outdir, exist_ok=True)
                with PdfPages(  # type: ignore
                    os.path.join(outdir, f"{cutoff_timestamp}.pdf"), f""
                ) as pdf:
                    for name, table in cutoff_rdl_task.db.tables.items():
                        graph, root = table.query_graph(
                            shapes=self.config.output.all_shapes
                        )
                        store_graph_to_disk(
                            graph,
                            pdf,
                            f"graphs/{cutoff_timestamp}",
                            highlight_nodes=[root],
                            prog="dot",
                            bend_edges=False,
                            title=f"Query graph for $\\bf{{{quote(name)}}}$ at {cutoff_timestamp}",
                        )

            # Run prediction.
            pred_task = PredictionTask.create(
                target_col_name=cutoff_rdl_task.target_col_name,
                id_col_name=cutoff_rdl_task.task_table_id_col_name,
                train_table=train_table,
                test_table=eval_table,
                entity_key_col_name=cutoff_rdl_task.task_table_entity_key_col_name,
                db=cutoff_rdl_task.db,
                eval_mode=mode,
                label=f"{cutoff_timestamp}",
                task_type=cutoff_rdl_task.task_type,
            )
            predictions, labels, keys = self._relicl_model.predict(pred_task)

            # on dry runs, compute no metrics
            if self.config.run.dry:
                continue

            # Compute metrics of this batch.
            match rdl_task.task_type:
                case TaskType.BINARY_CLASSIFICATION:
                    step_metrics = compute_binary_classification_metrics(
                        labels, predictions, print_metrics=False
                    )
                    timestep_iterator.set_postfix(dict(auroc=step_metrics["auroc"]))
                case TaskType.REGRESSION:
                    step_metrics = compute_regression_metrics(
                        labels, predictions, print_metrics=False
                    )
                    timestep_iterator.set_postfix(dict(mae=step_metrics["mae"]))
                case _:
                    raise ValueError(f"Unsupported task type: {rdl_task.task_type}")

            # Trace results.
            trace.trace(
                "timestep_results",
                eval_mode=mode,
                cutoff_timestamp=cutoff_timestamp,
                **step_metrics,
            )

            # Store raw results.
            collected_preds_proba.append(predictions)
            collected_labels.append(labels)
            collected_keys.append(np.asarray(keys))
            collected_cutoffs.append(
                np.full(len(keys), pd.Timestamp(cutoff_timestamp).isoformat())
            )

        if self.config.run.dry:
            logger.warning(f"Metrics ({mode}): skipped (dry run)")
            return {}

        # Compute overall metrics.
        match rdl_task.task_type:
            case TaskType.BINARY_CLASSIFICATION:
                overall_metrics = compute_binary_classification_metrics(
                    np.concatenate(collected_labels),
                    np.concatenate(collected_preds_proba),
                    print_metrics=False,
                )
            case TaskType.REGRESSION:
                overall_metrics = compute_regression_metrics(
                    np.concatenate(collected_labels),
                    np.concatenate(collected_preds_proba),
                    print_metrics=False,
                )
            case _:
                raise ValueError(f"Unsupported task type: {rdl_task.task_type}")
        logger.info(f"Metrics ({mode}): {overall_metrics}")
        trace.trace("final_results", eval_mode=mode, **overall_metrics)

        if self.config.output.predictions:
            self._save_predictions(
                task_type=rdl_task.task_type,
                mode=mode,
                keys=np.concatenate(collected_keys),
                cutoffs=np.concatenate(collected_cutoffs),
                labels=np.concatenate(collected_labels),
                predictions=np.concatenate(collected_preds_proba),
            )

        return overall_metrics

    def _save_predictions(
        self,
        task_type: TaskType,
        mode: EvalMode,
        keys: np.ndarray,
        cutoffs: np.ndarray,
        labels: np.ndarray,
        predictions: np.ndarray,
    ) -> None:
        """Writes all predictions of one split as a single parquet file."""

        n = len(keys)
        hydra = HydraConfig.get()

        # Columns shared by both task types.
        data: dict[str, Any] = dict(
            key=keys,
            cutoff_timestamp=cutoffs,
            y_true=labels,
            task_type=np.full(n, task_type.value),
        )

        # Classification keeps the full probability vector, regression the estimate.
        match task_type:
            case TaskType.BINARY_CLASSIFICATION:
                data["y_pred"] = np.argmax(predictions, axis=1)
                data["p_pred"] = np.max(predictions, axis=1)
                data["proba"] = list(predictions)
            case TaskType.REGRESSION:
                data["y_pred"] = predictions
            case _:
                raise ValueError(f"Unsupported task type: {task_type}")

        filename = os.path.join(hydra.runtime.output_dir, f"predictions-{mode}.parquet")
        with timed_step("dump_predictions", eval_mode=str(mode), n_rows=n):
            Predictions(hydra.job.name, hydra.job.id, mode, pd.DataFrame(data)).save(
                filename
            )
