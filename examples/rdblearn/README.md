# RelICL vs RDBLearn on constructed tasks

Re-runs the two constructed tasks reported in the paper: the cross-column
interaction task (`dfs_pathological`) and the bounded depth task (`dfs_depth`).

## Install

Requires Python 3.12 and a CUDA GPU. From the repository root:

```bash
# RelICL.
poetry install
cp -n config/relicl-example.yaml config/relicl.yaml

# RDBLearn, which needs its own environment.
python3.12 -m venv examples/rdblearn/.venv
examples/rdblearn/.venv/bin/python -m pip install \
    -r examples/rdblearn/requirements.txt
```

## Run

From the repository root, with the three seeds reported in the paper:

```bash
SEEDS="0 1 2" examples/rdblearn/dfs_pathological/run_all.sh
SEEDS="0 1 2" examples/rdblearn/dfs_depth/run_all.sh
```

Per-run results land in `examples/rdblearn/*/results/`, and the tables in the paper
are written to `paper/tex/generated/table-dfs-pathological.tex` and
`paper/tex/generated/table-dfs-depth.tex`.
