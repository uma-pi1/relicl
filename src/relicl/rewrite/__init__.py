from relicl.config import RelICLConfig
from relicl.rewrite.contract_edge_from_task_table import (
    rewrite_contract_edge_from_task_table,
)
from relicl.rewrite.count_features import rewrite_add_count_features
from relicl.rewrite.drop_self_references import rewrite_drop_self_references
from relicl.rewrite.eliminate_junction_tables import rewrite_eliminate_junction_tables
from relicl.rewrite.normalize_column_names import rewrite_normalize_column_names
from relicl.rewrite.pull_fk_rows import rewrite_pull_fk_rows
from relicl.rewrite.utils import save_step_fn
from relicl.schema import RDLTask
from relicl.timing import timed
from relicl.typing import SimplifyMethod, TimingEvent


@timed(event_type=TimingEvent.rewrite)
def apply_rewrites(rdl_task: RDLTask) -> None:
    config = RelICLConfig.instance()
    save_step = save_step_fn(rdl_task)
    save_step("Original schema")

    rewrite_normalize_column_names(rdl_task, save_step)

    add_counts = config.rewrite.count_features
    if add_counts and not config.rewrite.count_features_after_simplify:
        rewrite_add_count_features(rdl_task.db, save_step)

    # Keep this after the count features above, which are the only users of a
    # self-reference.
    rewrite_drop_self_references(rdl_task, save_step)

    if config.rewrite.simplify is not None:
        match config.rewrite.simplify:
            case SimplifyMethod.default:
                rewrite_pull_fk_rows(rdl_task, save_step)
                rewrite_eliminate_junction_tables(rdl_task, save_step)
                rewrite_contract_edge_from_task_table(rdl_task, save_step)
            case SimplifyMethod.old:
                from .old.simplify import simplify_task

                simplify_task(rdl_task)

    if add_counts and config.rewrite.count_features_after_simplify:
        rewrite_add_count_features(rdl_task.db, save_step)

    save_step(close=True)
