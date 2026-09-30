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
        name="depth-toy",
        tables={table.name: table.df for table in frames.chain},
        primary_keys={table.name: table.pkey for table in frames.chain},
        foreign_keys=[
            (child.name, parent.pkey, parent.name, parent.pkey)
            for parent, child in zip(frames.chain, frames.chain[1:])
        ],
    )

    # RDBLearn at its defaults apart from the depth bound.
    clf = rdblearn_common.make_classifier(CONFIG.seed, max_depth=CONFIG.dfs_max_depth)
    metrics = rdblearn_common.fit_and_score(clf, rdb, frames.task, CONFIG.n_train)

    write_json(
        RESULTS_DIR
        / f"rdblearn_depth{CONFIG.depth}_dfs{CONFIG.dfs_max_depth}_seed{CONFIG.seed}.json",
        dict(
            system="rdblearn",
            depth=CONFIG.depth,
            dfs_max_depth=CONFIG.dfs_max_depth,
            seed=CONFIG.seed,
            **metrics,
        ),
    )


if __name__ == "__main__":
    main()
