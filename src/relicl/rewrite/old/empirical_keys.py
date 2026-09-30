import logging
from itertools import chain, combinations
from typing import Iterator, Sequence, TypeVar

import networkx as nx
import pandas as pd

from relicl.schema import Ref

from .joiner import Joiner

logger = logging.getLogger(__name__)


########################################################################################
# Main class. ##########################################################################
########################################################################################


class EmpiricalKeysJoiner(Joiner):
    def _build_join_graph(self) -> nx.DiGraph:
        # Initialize empty graph.
        graph: nx.DiGraph = nx.DiGraph()

        # Add all tables as nodes.
        graph.add_nodes_from(self._db.schema.nodes)

        # Identify empirical keys.
        table_to_keys: dict[str, list[tuple[Ref, ...]]] = {}
        for table_name, table in self._db.tables.items():
            fkey_refs = self._db.get_fkey_refs(table_name)
            table_to_keys[table_name] = _identify_empirical_keys(table.df, fkey_refs)

        # Find matches of empirical keys between tables.
        matching_keys = _find_empirical_key_matches(table_to_keys)

        # Add edges based on matching empirical keys.
        for (source, target), (matching_key1, matching_key2) in matching_keys.items():
            # This mapping is used to identify which columns in the 2nd table correspond
            # to the columns in the 1st table. Two columns correspond to each other if
            # they point to the same target.
            right_by_target = {ref.to_col_name: ref for ref in matching_key2}

            left_on: list[str] = []
            right_on: list[str] = []

            for ref1 in matching_key1:
                ref2 = right_by_target[ref1.to_col_name]
                left_on.append(ref1.from_col_name)
                right_on.append(ref2.from_col_name)

            graph.add_edge(source, target, left_on=left_on, right_on=right_on)

        return graph


########################################################################################
# Helper functions. ####################################################################
########################################################################################

T = TypeVar("T")


def _identify_empirical_keys(
    df: pd.DataFrame, fkey_refs: list[Ref]
) -> list[tuple[Ref, ...]]:
    """
    For each subset of the foreign keys, check if the subset uniquely identifies rows
    in the table.

    This function checks this empirically, i.e., by comparing the number of total rows
    to the number of unique occurrences of the foreign key columns in the subset.
    """

    def non_empty_subsets(ls: Sequence[T]) -> Iterator[tuple[T, ...]]:
        return chain.from_iterable(combinations(ls, n) for n in range(1, len(ls) + 1))

    empirical_keys: list[tuple[Ref, ...]] = []
    for subset in non_empty_subsets(fkey_refs):
        cols = [ref.from_col_name for ref in subset]
        if df[cols].dropna().drop_duplicates().shape[0] == df[cols].dropna().shape[0]:
            empirical_keys.append(subset)

    return empirical_keys


def _find_empirical_key_matches(
    tables_to_keys: dict[str, list[tuple[Ref, ...]]],
) -> dict[tuple[str, str], tuple[tuple[Ref, ...], tuple[Ref, ...]]]:
    matches: dict[tuple[str, str], tuple[tuple[Ref, ...], tuple[Ref, ...]]] = {}

    for table_name_1, keys1 in tables_to_keys.items():
        for table_name_2, keys2 in tables_to_keys.items():
            if table_name_1 != table_name_2 and len(keys1) > 0 and len(keys2) > 0:
                match = _find_empirical_key_match(keys1, keys2)
                if match is not None:
                    matches[(table_name_1, table_name_2)] = match

    return matches


def _find_empirical_key_match(
    keys1: list[tuple[Ref, ...]], keys2: list[tuple[Ref, ...]]
) -> tuple[tuple[Ref, ...], tuple[Ref, ...]] | None:
    """
    A match between two tables exists if there is a key in one table that is a
    (non-empty) subset of a key in the other table.

    Note that there can be more than one match between two tables. This method returns
    upon the first match.
    """

    def sig(key: tuple[Ref, ...]) -> frozenset[str]:
        """
        A key is identified by the targets of the refs it contains.

        This means that one key is a subset of another if the targets of the first key
        are a subset of the targets of the second key.
        """
        return frozenset(ref.to_col_name for ref in key)

    for key1 in keys1:
        sig1 = sig(key1)
        for key2 in keys2:
            sig2 = sig(key2)
            if sig1 <= sig2:
                return key1, key2

    return None
