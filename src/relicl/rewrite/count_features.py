import logging

from relicl.schema import Database, Ref, Table
from relicl.schema.table_dask import CountFeatureDaskTable
from relicl.schema.table_pandas import CountFeaturePandasTable
from relicl.timing import timed
from relicl.typing import BackendType, SaveStepFn, TimingEvent

logger = logging.getLogger(__name__)


@timed(event_type=TimingEvent.rewrite)
def rewrite_add_count_features(
    db: Database, save_step: SaveStepFn = SaveStepFn.default()
) -> None:
    new_tables: dict[str, Table] = {}

    # we collect the new_tables first
    for u, v, data in db.schema.edges(data=True):
        ref: Ref = data["ref"]
        if ref.is_to_many:
            new_tables[u] = _count_feature_table(
                new_tables.get(u, db.tables[u]),  # in case multiple counts are added
                db.tables[v],
                ref.from_col_name,
                ref.to_col_name,
            )
        if ref.is_from_many:
            new_tables[v] = _count_feature_table(
                new_tables.get(v, db.tables[v]),  # in case multiple counts are added
                db.tables[u],
                ref.to_col_name,
                ref.from_col_name,
            )

    # and then put them into the db
    db.tables.update(new_tables)

    # and save schema graph
    save_step("add_count_features")


def _count_feature_table(
    one_table: Table,
    many_table: Table,
    one_col_name: str,
    many_col_name: str,
) -> Table:
    match one_table.backend:
        case BackendType.dask:
            return CountFeatureDaskTable(
                one_table, many_table, one_col_name, many_col_name
            )
        case BackendType.pandas:
            return CountFeaturePandasTable(
                one_table, many_table, one_col_name, many_col_name
            )
