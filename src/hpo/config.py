import os.path as osp
from pathlib import Path

# Paths. ###############################################################################

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = Path(
    osp.abspath(
        next(
            parent
            for parent in CURRENT_FILE.parents
            if (parent / "pyproject.toml").exists()
        )
    )
)
