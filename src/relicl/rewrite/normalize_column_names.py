from relicl.schema import RDLTask
from relicl.schema.database import normalized_column_name
from relicl.timing import timed
from relicl.typing import SaveStepFn, TimingEvent


@timed(event_type=TimingEvent.rewrite)
def rewrite_normalize_column_names(
    rdl_task: RDLTask, save_step: SaveStepFn = SaveStepFn.default()
) -> None:
    # Normalize column names.
    rdl_task.db.normalize_column_names()
    rdl_task.target_col_name = normalized_column_name(
        rdl_task.task_table_name, rdl_task.target_col_name
    )
    rdl_task.task_table_time_col_name = normalized_column_name(
        rdl_task.task_table_name, rdl_task.task_table_time_col_name
    )
    rdl_task.task_table_id_col_name = normalized_column_name(
        rdl_task.task_table_name, rdl_task.task_table_id_col_name
    )
    rdl_task.task_table_entity_key_col_name = normalized_column_name(
        rdl_task.task_table_name, rdl_task.task_table_entity_key_col_name
    )

    save_step("normalize_column_names")
