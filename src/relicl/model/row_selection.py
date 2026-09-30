import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from ordered_set import OrderedSet

from relicl.config import WithConfig
from relicl.effective_config import EffectiveConfig
from relicl.rng import get_rng
from relicl.timing import timed
from relicl.typing import TimingEvent

logger = logging.getLogger(__name__)

########################################################################################
# Abstract base class. #################################################################
########################################################################################


class RelevantRowSelector(WithConfig, ABC):
    """Base class for selecting relevant rows in a table."""

    def __init__(self, table_name: str, df: pd.DataFrame) -> None:
        WithConfig.__init__(self)
        self._table_name = table_name
        self._df = df

    @abstractmethod
    def mark_keys_relevant(self, key_col_name: str, keys: pd.Series) -> None:
        """Mark the specified keys as relevant."""

    @property
    @abstractmethod
    def df(self) -> pd.DataFrame:
        """Return the relevant rows."""


########################################################################################
# No-op sub-class. #####################################################################
########################################################################################


class AllRowSelector(RelevantRowSelector):
    """Selects all rows. All rows are considered relevant, no sampling is applied."""

    def mark_keys_relevant(self, key_col_name: str, keys: pd.Series) -> None:
        raise NotImplementedError("This implementation considers all rows as relevant.")

    @property
    def df(self) -> pd.DataFrame:
        return self._df


########################################################################################
# Sampling sub-class. ##################################################################
########################################################################################


class SamplingRelevantRowSelector(RelevantRowSelector):
    """
    Selects a set of relevant rows. Applies downsampling if too many relevant rows
    exist.
    """

    def __init__(self, table_name: str, df: pd.DataFrame) -> None:
        super().__init__(table_name, df)

        self._relevant_row_mask = np.zeros(len(df), dtype=bool)
        self._relevant_rows_per_key: OrderedSet[frozenset[int]] = OrderedSet()

    def mark_keys_relevant(self, key_col_name: str, keys: pd.Series) -> None:
        # Compute relevant row masks for t.
        row_mask = self._df[key_col_name].isin(keys).to_numpy()
        self._relevant_row_mask |= row_mask

        # Compute relevant rows by key value (used for sampling). First filter on the
        # rows that have a relevant key, then collect the set of row numbers for each
        # key.
        key_pos_df = self._df.assign(__pos__=np.arange(len(self._df)))[
            [key_col_name, "__pos__"]
        ]  # position in full table

        # Compute relevant rows of this key.
        rows_per_key: list[frozenset[int]] = (
            key_pos_df.iloc[row_mask]  # only relevant keys
            .groupby(key_col_name)["__pos__"]
            .agg(frozenset)  # type: ignore[arg-type]
            .to_list()
        )

        # Store new row number sets.
        self._relevant_rows_per_key.update(rows_per_key)

    @property
    def _requires_sampling(self) -> bool:
        """Indicates whether (down-)sampling is required."""
        return (
            self._relevant_row_mask.sum()
            > self.config.sampling.context_tables_sample_size
        )

    @timed(event_type=TimingEvent.preprocessing)
    def _sample_if_needed(self) -> None:
        if self._requires_sampling:
            sample_size = self.config.sampling.context_tables_sample_size

            # Sampling required (always done from the full set of relevant rows). Make sure
            # that at least one instance of each relevant key is retained.
            n_keys = len(self._relevant_rows_per_key)
            assert n_keys > 0

            # Check whether sample size permits at least one row per key.
            per_key_budget = sample_size // n_keys
            if per_key_budget < 1:
                logger.warning(
                    f"Sampling budget ({sample_size}) is too small to retain at "
                    f"least one row per key ({n_keys} keys) from table '{self._table_name}'. "
                    f"The budget is increased to {n_keys}."
                )
                EffectiveConfig.count("context_sample_size_raised")
                EffectiveConfig.maximum("context_sample_size", n_keys)
                per_key_budget = 1

            # Sample from rows per key. The key is irrelevant, it only matters that at least
            # one row per key is retained.
            sampled_rows: set[int] = set()
            for row_set in self._relevant_rows_per_key:
                row_list = sorted(row_set)  # (sorting for reproducibility)
                chosen = get_rng().choice(
                    row_list,
                    size=(min(per_key_budget, len(row_list))),
                    replace=False,
                )
                sampled_rows.update(int(i) for i in chosen)

            # Apply sample to row mask.
            sampled_row_mask = np.zeros(len(self._df), dtype=bool)
            sampled_row_mask[list(sampled_rows)] = True
            self._relevant_row_mask = sampled_row_mask

    @property
    def df(self) -> pd.DataFrame:
        # (Down-)sample the data frame. The result is effectively cached (since the
        # sample method modifies the state, i.e., this class is mutable).
        if self._requires_sampling:
            self._sample_if_needed()

        return self._df.iloc[self._relevant_row_mask, :]
