import logging
import os
from typing import Any

import yaml

logger = logging.getLogger(__name__)

EFFECTIVE_FILENAME = "_EFFECTIVE.yaml"


class EffectiveConfig:
    """Records where a run's behavior departed from the config it was given.

    This may be relevant for several hyperparameters, for instance: a sampling budget
    is raised to fit its anchors, recency weighting finds a single timestamp to act on.

    Counters are cumulative over a run, since most of these are decided per test
    batch rather than once.
    """

    _facts: dict[str, Any] = {}

    @classmethod
    def reset(cls) -> None:
        cls._facts = {}

    @classmethod
    def count(cls, key: str, n: int = 1) -> None:
        """Counts an occurrence. A key that is absent never occurred."""
        cls._facts[key] = cls._facts.get(key, 0) + n

    @classmethod
    def maximum(cls, key: str, value: float) -> None:
        """Keeps the largest value seen, e.g.,the largest a raised budget reached."""
        cls._facts[key] = max(cls._facts.get(key, value), value)

    @classmethod
    def add(cls, key: str, item: str) -> None:
        """Collects distinct names, e.g., the context tables that were dropped."""
        items = cls._facts.setdefault(key, [])
        if item not in items:
            items.append(item)

    @classmethod
    def facts(cls) -> dict[str, Any]:
        return dict(cls._facts)

    @classmethod
    def dump(cls, dirname: str) -> None:
        """Writes the facts next to the run's results, or nothing if there are none."""
        path = os.path.join(dirname, EFFECTIVE_FILENAME)
        with open(path, "w") as f:
            yaml.safe_dump(cls.facts(), f, sort_keys=True)
        if cls._facts:
            logger.info(f"Effective-config differences: {cls.facts()}")
