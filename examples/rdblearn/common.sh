# Sourced by the examples' run_all.sh.

RDBLEARN_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
VENV="$RDBLEARN_DIR"/.venv

# Install the RDBLearn venv shared by all examples.
[ -d "$VENV" ] || python3.12 -m venv "$VENV"
"$VENV"/bin/python -m pip install -r "$RDBLEARN_DIR"/requirements.txt

# RelICL runs with method.use_max_reproducibility=true.
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=":4096:8"

# The examples are modules under `examples.rdblearn`.
cd "$RDBLEARN_DIR"/../..
