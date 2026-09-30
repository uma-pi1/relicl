import logging
import sys
from collections.abc import Callable

import yaml
from hydra.core.config_store import ConfigStore
from relbench.base import TaskType

from relicl import rng
from relicl.config import TASK_TABLE_ID_COL_NAME, TASK_TABLE_NAME, RelICLConfig
from relicl.effective_config import EffectiveConfig
from relicl.rdl import RDLRunner
from relicl.rewrite import apply_rewrites
from relicl.schema import Database, RDLTask

logger = logging.getLogger(__name__)

# Hydra composes `config/relicl.yaml` on top of the structured config.
ConfigStore.instance().store(name="base_config", node=RelICLConfig)

# Overrides shared by all examples.
DEFAULT_OVERRIDES = (
    "run.val=false",
    "run.test=true",
    "rewrite.count_features=false",
    "fusion.key_pooling=max",
    # Needs PYTHONHASHSEED and CUBLAS_WORKSPACE_CONFIG in the environment.
    "method.use_max_reproducibility=true",
)


def insert_default_overrides(*overrides: str) -> None:
    """Add overrides to the command line unless it already sets them.

    Hydra resolves hydra.job.name from relbench.db/task, so call this before main().
    """
    argv = " ".join(sys.argv)
    for override in (*DEFAULT_OVERRIDES, *overrides):
        if override.split("=", 1)[0] not in argv:
            sys.argv.insert(1, override)


def finalize_task(db: Database, n_train: int, n_test: int) -> RDLTask:
    """Validate the database and wrap it in a binary classification task."""
    db.assert_is_valid()

    # The RelBench path stamps a no-op cutoff on every base table, and
    # db.cutoff() relies on it being there.
    for name, table in list(db.tables.items()):
        db.tables[name] = table.filter_by_time(None)

    return RDLTask(
        db=db,
        task_table_name=TASK_TABLE_NAME,
        task_table_time_col_name="date",
        task_table_id_col_name=TASK_TABLE_ID_COL_NAME,
        task_table_entity_key_col_name="entity_id",
        target_col_name="label",
        train_ids=list(range(n_train)),
        val_ids=[],
        test_ids=list(range(n_train, n_train + n_test)),
        task_type=TaskType.BINARY_CLASSIFICATION,
    )


def run(config: RelICLConfig, build_task: Callable[[], RDLTask]) -> dict:
    """Run RelICL on the built task and return its test metrics."""
    # Config setup.
    RelICLConfig.set(config)
    EffectiveConfig.reset()
    rng.seed(config.method.seed, config.method.use_max_reproducibility)

    rdl_task = build_task()
    logger.info("Schema:\n" + yaml.dump(rdl_task.description(), width=float("inf")))

    # As in RelICL's own pipeline, which simplifies the schema by default.
    apply_rewrites(rdl_task)

    results = RDLRunner().run(rdl_task)
    print("\n=== metrics ===")
    print(yaml.dump(results, default_flow_style=False))
    return results.get("test", {})
