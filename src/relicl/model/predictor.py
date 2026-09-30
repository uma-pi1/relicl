import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator

import numpy as np
import pandas as pd
from relbench.base import TaskType

from relicl.config import RelICLConfig, WithConfig
from relicl.effective_config import EffectiveConfig
from relicl.model.embeddings import ContextRowEmbeddings
from relicl.model.vectorizer import Vectorizer
from relicl.progress import trange
from relicl.rng import get_rng
from relicl.schema import Database, Table
from relicl.timing import timed
from relicl.typing import EvalMode, TimingEvent

logger = logging.getLogger(__name__)


# Sampling helpers. ####################################################################


@timed(event_type=TimingEvent.preprocessing)
def _score_rows(train_table: Table) -> pd.Series:
    """
    Scores training rows for sampling. The score itself carries no meaning beyond its
    order (see _sample_indices).
    """
    train_conf = RelICLConfig.instance().sampling.train
    index = np.arange(len(train_table))

    # No half-life set: sample weights randomly (leads to uniform sampling).
    if train_conf.recency_half_life_steps is None:
        return pd.Series(get_rng().random(len(train_table)), index=index)

    # Sanity check: recency weighting needs a time column to measure age against.
    assert train_table.time_col_names, (
        "sampling.train.recency_half_life_steps needs a time column on the train table"
    )

    times = train_table.df[list(train_table.time_col_names)].max(axis=1).to_numpy()

    # RelBench spaces task timestamps by the task's prediction horizon, so the median
    # gap between them is the unit that makes a half-life comparable across tasks.
    stamps = np.unique(times)

    # A single timestamp leaves no gap to measure, but it also means every row is
    # equally recent, so the uniform score already is the correctly weighted one.
    if len(stamps) <= 1:
        logger.warning(
            "The train table has at most one distinct timestamp, so every row is equally "
            "recent and sampling.train.recency_half_life_steps cannot change anything."
        )
        EffectiveConfig.count("recency_half_life_inert")
        return pd.Series(get_rng().random(len(train_table)), index=index)

    age_steps = (stamps[-1] - times) / np.median(np.diff(stamps))

    # Gumbel-top-k: taking the largest `gumbel + log(weight)` draws a weighted sample
    # without replacement, with weights `0.5 ** (age / half_life)`.
    log_weight = -(age_steps / train_conf.recency_half_life_steps) * np.log(2.0)
    scores = get_rng().gumbel(size=len(train_table)) + log_weight
    return pd.Series(scores, index=index)


@timed(event_type=TimingEvent.preprocessing)
def _sample_indices(
    train_table: Table,
    test_table: Table,
    entity_key_col_name: str,
    scores: pd.Series,
) -> np.ndarray:
    conf = RelICLConfig.instance().sampling.train

    if not conf.restrict_to_test_keys:
        # No restriction: Sample from all rows.
        return scores.nlargest(conf.sample_size).index.to_numpy()

    EffectiveConfig.count("restrict_to_test_keys_batches")

    # Work with a positional RangeIndex to keep indices compatible with iloc.
    train_df = train_table.df.reset_index(drop=True)

    # Phase 1: Take k rows per test entity ("anchor").
    test_keys = test_table.df[entity_key_col_name]
    n_test_keys = int(test_keys.nunique())
    n_per_test_key = 0
    if conf.anchor_budget_fraction > 0.0 and n_test_keys > 0:
        n_per_test_key = max(
            1, round(conf.sample_size * conf.anchor_budget_fraction / n_test_keys)
        )

    anchor_mask = train_df[entity_key_col_name].isin(test_keys)
    if n_per_test_key > 0:
        anchor_indices = (
            scores[anchor_mask]
            .groupby(train_df.loc[anchor_mask, entity_key_col_name], sort=False)
            .nlargest(n_per_test_key)
            .index.get_level_values(-1)
            .to_numpy(dtype=np.int64)
        )
    else:
        anchor_indices = np.array([], dtype=int)

    # Report which fixed depth the configured fraction works out to, so that the
    # reparameterization can be read against the runs that used a fixed count.
    logger.debug(
        f"Anchors: fraction {conf.anchor_budget_fraction:g} of budget {conf.sample_size} "
        f"over {n_test_keys} test key(s) -> up to {n_per_test_key} per key, "
        f"{len(anchor_indices)} selected."
    )

    # On a task whose entities never repeat, no train row matches a test key and the
    # restriction selects nothing.
    if n_per_test_key > 0 and len(anchor_indices) == 0:
        logger.debug(
            "No train row shares an entity key with this test batch, so "
            "sampling.train.restrict_to_test_keys has no effect here."
        )
        EffectiveConfig.count("restrict_to_test_keys_inert")

    # Phase 2: Fill remaining budget (either from all rows or only from anchors).
    budget = conf.sample_size - len(anchor_indices)
    if budget <= 0:
        # The cap is soft here: dropping anchors would leave those test entities without
        # their own history in the context window, which is the point of the
        # restriction, so the budget is raised to fit them instead of subsampling.
        if budget < 0:
            logger.warning(
                f"Sampling budget ({conf.sample_size}) is too small to hold the "
                f"per-test-key training rows ({len(anchor_indices)} anchors). The "
                f"budget is increased to {len(anchor_indices)}."
            )
            EffectiveConfig.count("train_sample_size_raised")
            EffectiveConfig.maximum("train_sample_size", len(anchor_indices))
        return anchor_indices

    # Restricting the fill pool to test entities leaves nothing behind when those
    # entities have no further rows beyond the anchors. Widen the pool rather than
    # train on tiny sample.
    only_from_test_keys = conf.only_fill_from_test_keys
    if only_from_test_keys and anchor_mask.sum() <= len(anchor_indices):
        logger.warning(
            f"Nothing left to fill the training sample from: {len(anchor_indices)} "
            f"anchor(s) exhaust every train row whose key appears in this test "
            f"batch. Continuing as if "
            f"sampling.train.only_fill_from_test_keys=false."
        )
        EffectiveConfig.count("only_fill_from_test_keys_disabled")
        only_from_test_keys = False

    # Either fill up only from test entities or from all rows.
    if only_from_test_keys:
        fill_pool_mask = anchor_mask.copy()
    else:
        fill_pool_mask = pd.Series(True, index=train_df.index)

    # Remove already selected anchors.
    fill_pool_mask.loc[anchor_indices] = False

    # Take rows with largest scores.
    fill_indices = scores[fill_pool_mask].nlargest(budget).index.to_numpy()

    # Return all sampled indices.
    return np.concatenate([anchor_indices, fill_indices])


# Dataclass: PredictionTask. ###########################################################


@dataclass
class PredictionTask(WithConfig):
    """Describes a prediction task."""

    # Training data.
    train_table: Table
    train_df_vec: pd.DataFrame
    train_targets: np.ndarray

    # Test data.
    test_table: Table
    test_df_vec: pd.DataFrame
    test_targets: np.ndarray

    # Task type.
    task_type: TaskType

    # Context information.
    entity_key_col_name: str
    db: Database

    # Evaluation split this task belongs to.
    eval_mode: EvalMode = EvalMode.val

    label: str = ""

    def df(self, table_name: str) -> pd.DataFrame:
        if table_name == self.train_table.name:
            # the task table
            return pd.concat([self.train_table.df, self.test_table.df])
        else:
            # all other tables
            return self.db.tables[table_name].df

    def df_vec(self, table_name: str) -> pd.DataFrame:
        if table_name == self.train_table.name:
            # the task table
            return pd.concat([self.train_df_vec, self.test_df_vec])
        else:
            # all other tables
            df = self.db.tables[table_name].df
            return Vectorizer.new_instance().fit_transform(df)

    @property
    def test_keys(self):
        # TODO HACK just use first column
        return self.test_table.df.iloc[:, 0].to_numpy()

    def iter_test_batches(
        self,
        batch_size: int,
    ) -> Generator["PredictionTask", None, None]:
        for i, start in enumerate(
            trange(
                0,
                len(self.test_table),
                batch_size,
                position=1,
                leave=False,
                desc="Batches",
            )
        ):
            # Batching. ################################################################
            end = start + batch_size

            logger.debug(
                f"Test batching: rows {start} - {end} (total: {len(self.test_table)})"
            )

            yield PredictionTask(
                # Train (unchanged).
                train_table=self.train_table,
                train_df_vec=self.train_df_vec,
                train_targets=self.train_targets,
                # Test (batched).
                test_table=self.test_table[start:end],
                test_df_vec=self.test_df_vec[start:end],
                test_targets=self.test_targets[start:end],
                # Database (unchanged).
                entity_key_col_name=self.entity_key_col_name,
                db=self.db,
                eval_mode=self.eval_mode,
                label=f"{self.label}-{i}",
                # Task type (unchanged).
                task_type=self.task_type,
            )

    @timed(event_type=TimingEvent.preprocessing)
    def sample_train(
        self,
    ) -> "PredictionTask":
        # Score each row.
        scores = _score_rows(self.train_table)

        # Sample indices given the scores.
        sample_indices = _sample_indices(
            train_table=self.train_table,
            test_table=self.test_table,
            entity_key_col_name=self.entity_key_col_name,
            scores=scores,
        )

        return PredictionTask(
            train_table=self.train_table[sample_indices],
            train_df_vec=self.train_df_vec.iloc[sample_indices],
            train_targets=self.train_targets[sample_indices],
            test_table=self.test_table,
            test_df_vec=self.test_df_vec,
            test_targets=self.test_targets,
            entity_key_col_name=self.entity_key_col_name,
            db=self.db,
            eval_mode=self.eval_mode,
            label=self.label,
            task_type=self.task_type,
        )

    @classmethod
    @timed(event_type=TimingEvent.preprocessing)
    def create(
        cls,
        target_col_name: str,
        id_col_name: str,
        train_table: Table,
        test_table: Table,
        task_type: TaskType,
        entity_key_col_name: str,
        db: Database,
        eval_mode: EvalMode = EvalMode.val,
        label: str = "",
    ) -> "PredictionTask":
        """
        Splits data into items and labels and vectorizes tables.
        """
        # TODO: Drop FK/PK-ID cols.
        # TODO: Incorporate count column as additional feature.
        # TODO: not needed, as this table is never directly used
        # assert train_table.df.equals(db.tables[train_table.name].df)

        # Extract data frames and targets
        df_train = train_table.df.copy()
        df_test = test_table.df.copy()
        train_targets = df_train.pop(target_col_name).astype(np.float32).to_numpy()
        test_targets = df_test.pop(target_col_name).astype(np.float32).to_numpy()

        # Remove ID column (no longer needed).
        df_train.pop(id_col_name)
        df_test.pop(id_col_name)

        # Vectorize
        vectorizer = Vectorizer.new_instance()
        df_vec_combined = vectorizer.fit_transform(
            pd.concat([df_train, df_test], ignore_index=True)
        )
        train_df_vec = df_vec_combined.iloc[: len(df_train)]
        test_df_vec = df_vec_combined.iloc[len(df_train) :]

        # noinspection PyCallingNonCallable
        return cls(
            train_table=train_table,
            train_df_vec=train_df_vec,
            train_targets=train_targets,
            test_table=test_table,
            test_df_vec=test_df_vec,
            test_targets=test_targets,
            entity_key_col_name=entity_key_col_name,
            db=db,
            eval_mode=eval_mode,
            label=label,
            task_type=task_type,
        )


# Predictor. ###########################################################################


class Predictor(WithConfig, ABC):
    """Performs the final prediction on the task table given row embeddings obtained
    from the context tables"""

    def __init__(self):
        WithConfig.__init__(self)

    @abstractmethod
    def predict(
        self, task: PredictionTask, context_embs_list: list[ContextRowEmbeddings]
    ) -> np.ndarray:
        """Performs the final prediction on the task table given row embeddings obtained
        from the context tables.

        Returns preds_proba.

        """
