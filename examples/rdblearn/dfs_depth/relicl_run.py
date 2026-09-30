import hydra
from ordered_set import OrderedSet

from relicl.config import REPO_ROOT, TASK_TABLE_NAME, BackendType, RelICLConfig
from relicl.schema import Database, RDLTask, Table

from .. import relicl_common
from ..common import write_json
from .config import CONFIG, RESULTS_DIR
from .data import frames_from_config


def _build_rdl_task() -> RDLTask:
    """Wrap the generated frames in RelICL's database schema types."""
    backend = BackendType.pandas
    frames = frames_from_config(CONFIG)

    db = Database()
    for table in frames.chain:
        db.add_table(
            Table.create(
                table.name, OrderedSet([table.pkey]), OrderedSet(), table.df, backend
            )
        )
    db.add_table(
        Table.create(
            TASK_TABLE_NAME, OrderedSet(), OrderedSet(["date"]), frames.task, backend
        )
    )
    db.add_many_to_one(TASK_TABLE_NAME, "entity_id", "entities", "entity_id")
    for parent, child in zip(frames.chain, frames.chain[1:]):
        db.add_many_to_one(child.name, parent.pkey, parent.name, parent.pkey)
    return relicl_common.finalize_task(db, CONFIG.n_train, CONFIG.n_test)


@hydra.main(
    version_base="1.3",
    config_path=str(REPO_ROOT / "config"),
    config_name="relicl",
)
def main(config: RelICLConfig) -> None:
    """Run one evaluation and record its test metrics."""
    metrics = relicl_common.run(config, _build_rdl_task)

    model = str(config.method.model)
    write_json(
        RESULTS_DIR / f"relicl_depth{CONFIG.depth}_{model}_seed{CONFIG.seed}.json",
        dict(
            system="relicl",
            depth=CONFIG.depth,
            model=model,
            seed=CONFIG.seed,
            **metrics,
        ),
    )


if __name__ == "__main__":
    relicl_common.insert_default_overrides(
        "relbench.db=depth-toy",
        "relbench.task=chain",
        # The same seed as the generated data.
        f"method.seed={CONFIG.seed}",
    )
    main()
