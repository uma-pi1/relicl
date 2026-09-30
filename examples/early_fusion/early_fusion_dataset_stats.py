import os.path as osp
import sys

import pandas as pd

from paper.config import EARLY_FUSION_DATASETS

# Reuse the task registry of the sibling `early_fusion_sandbox.py`.
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from early_fusion_sandbox import _REGISTRY, load_task  # noqa: E402
from early_fusion_seeds import DATASETS  # noqa: E402


def main() -> None:
    """Collect the table statistics `app:early-fusion` reports.

    Only OpenML is touched, so this is cheap to rerun and independent of the runs in
    `early_fusion_seeds.py`.
    """
    rows = []
    for dataset in DATASETS:
        data, target_col, _ = load_task(dataset)
        rows.append(
            dict(
                dataset=dataset,
                openml_id=_REGISTRY[dataset].openml_id,
                rows=len(data),
                features=data.shape[1],
                pos_rate=float(target_col.mean()),
            )
        )
        print(f"{dataset}: {rows[-1]}")

    # Written straight to where `poetry run paper` reads it.
    out_path = EARLY_FUSION_DATASETS
    pd.DataFrame(rows).sort_values("dataset").to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
