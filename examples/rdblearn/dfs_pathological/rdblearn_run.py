from .. import rdblearn_common
from ..common import write_json
from .config import CONFIG, RESULTS_DIR
from .data import frames_from_config


# noinspection unresolved-references
def main() -> None:
    # Imports that are not available in the RelICL environment.
    from fastdfs.api import create_rdb

    frames = frames_from_config(CONFIG)
    rdb = create_rdb(
        name="corr-toy",
        tables={"entities": frames.entities, "signals": frames.signals},
        primary_keys={"entities": "entity_id", "signals": "signal_id"},
        foreign_keys=[("signals", "entity_id", "entities", "entity_id")],
    )

    # RDBLearn at its defaults. Every aggregate DFS generates here is constant across
    # entities, so all of them are dropped and only the join key reaches TabPFN.
    clf = rdblearn_common.make_classifier(CONFIG.seed)
    metrics = rdblearn_common.fit_and_score(clf, rdb, frames.task, CONFIG.n_train)

    write_json(
        RESULTS_DIR / f"rdblearn_{CONFIG.task}_seed{CONFIG.seed}.json",
        dict(system="rdblearn", task=str(CONFIG.task), seed=CONFIG.seed, **metrics),
    )


if __name__ == "__main__":
    main()
