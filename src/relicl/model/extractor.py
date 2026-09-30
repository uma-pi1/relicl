from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from relbench.base import TaskType

from relicl.config import WithConfig
from relicl.model.early_fusion import fuse
from relicl.model.embeddings import (
    ContextRowEmbeddings,
    RowEmbeddings,
    pool_context_embs_list,
)
from relicl.timing import timed
from relicl.typing import FusionType

########################################################################################
# Class: Extractor. ####################################################################
########################################################################################


class Extractor(WithConfig, ABC):
    """Extracts row embeddings for the rows of a given table"""

    def __init__(self) -> None:
        WithConfig.__init__(self)

    @abstractmethod
    def extract_without_context(
        self, df: pd.DataFrame, task_type: TaskType
    ) -> RowEmbeddings:
        """Extracts row embeddings of the rows in the given data frame."""

    def extract_with_context(
        self,
        df: pd.DataFrame,
        context_embs_list: list[ContextRowEmbeddings],
        task_type: TaskType,
    ) -> RowEmbeddings:
        """
        Adds new columns to the data frame from the context embeddings, then runs
        extraction.
        """

        # Run extraction.
        fused_df = fuse(df, context_embs_list)
        fused_df.dropna(axis=1, inplace=True, how="all")
        return self.extract_without_context(fused_df, task_type)

    @abstractmethod
    def normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize embeddings."""

    @timed()
    def extract(
        self,
        df: pd.DataFrame,
        context_embs_list: list[ContextRowEmbeddings],
        task_type: TaskType,
    ) -> RowEmbeddings:
        """Extracts row embeddings of the rows in the given table, incorporating the
        information from embeddings of successor context tables.
        """
        if len(context_embs_list) == 0:
            return self.extract_without_context(df, task_type)

        # Pool the context row embeddings across successor tables. Only late fusion
        # uses the pooled embeddings; early fusion needs the counts alone.
        if self.config.fusion.type is FusionType.late:
            late_config = self.config.fusion.late
            assert late_config is not None, "fusion.late is unset"

            pooled_context_embs, context_counts_df = pool_context_embs_list(
                late_config.successor_pooling.to_numpy(),
                context_embs_list,
                self.normalize if late_config.successor_normalize else lambda X: X,
            )
        else:
            pooled_context_embs, context_counts_df = pool_context_embs_list(
                None, context_embs_list, None, only_counts=True
            )

        # Add counts to data frame
        # TODO we many not need this anymore, really. The main difference is that this
        #  are sample counts, whereas the global count features are global counts.
        if self.config.rewrite.count_features:
            df = pd.concat([df.reset_index(drop=True), context_counts_df], axis=1)

        match self.config.fusion.type:
            case FusionType.early:
                return self.extract_with_context(df, context_embs_list, task_type)
            case FusionType.late:
                # Run tabular model.
                extracted_row_embs = self.extract_without_context(df, task_type)

                # Pool the output with the pooled context row embeddings.
                assert pooled_context_embs is not None
                return self.pool_extracted_and_context_embs(
                    extracted_row_embs, pooled_context_embs
                )
            case FusionType.none:
                raise ValueError(
                    "'No fusion' should not reach this; nothing should be extracted."
                )

    def pool_extracted_and_context_embs(
        self, extracted_row_embs: RowEmbeddings, pooled_context_embs: RowEmbeddings
    ) -> RowEmbeddings:
        """Pools extracted embeddings and pooled context embeddings.

        Extractors can use this to propagate context embeddings, esp. when the context
        embeddings do not influence the extracted row embeddings (in which case the
        information in the context embeddings would get lost otherwise).
        """
        # Sanity check: reached from the late branch of `extract` alone.
        assert self.config.fusion.type is FusionType.late
        late_config = self.config.fusion.late
        assert late_config is not None, "fusion.late is unset"

        pooling_fn = late_config.combine_pooling.to_numpy()
        pooled_row_embs = pooling_fn(
            np.array(
                [
                    extracted_row_embs.embs,
                    pooled_context_embs.embs * late_config.combine_successor_weight,
                ]
            ),
            axis=0,
        )
        if late_config.combine_normalize:
            pooled_row_embs = self.normalize(pooled_row_embs)

        return RowEmbeddings(pooled_row_embs)
