#!/usr/bin/env bash
set -euo pipefail

source "$( dirname -- "${BASH_SOURCE[0]}" )"/../common.sh
PKG=examples.rdblearn.dfs_depth

# Seeds to run, e.g. SEEDS="0 1 2". Each seeds the data and both models.
for seed in ${SEEDS:-0}; do
    export SEED="$seed"
    for model in tabicl tabpfn; do
        poetry run python -m "$PKG".relicl_run model="$model"
    done

    # RDBLearn at its default depth bound, and at the bound that reaches the signal.
    for dfs_max_depth in 2 3; do
        DFS_MAX_DEPTH="$dfs_max_depth" "$VENV"/bin/python -m "$PKG".rdblearn_run
    done
done

poetry run python -m "$PKG".collect_results
