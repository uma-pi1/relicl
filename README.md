# RelICL

Code for the paper
[RelICL: Training-free Relational Learning with Tabular Foundation Models](https://arxiv.org/abs/2610.01725).
The results behind the paper are in
[uma-pi1/relicl-results](https://github.com/uma-pi1/relicl-results).

## Installation

Requires Ubuntu 22.04 or later, Python 3.12, a CUDA GPU and
[Poetry](https://python-poetry.org/).

```bash
git clone https://github.com/uma-pi1/relicl
cd relicl
poetry install
cp config/relicl-example.yaml config/relicl.yaml
```

## Run

Run RelICL on a [RelBench](https://relbench.stanford.edu/) dataset and task:

```bash
poetry run relicl run relbench.db=<dataset> relbench.task=<task>
```

Any setting in `config/relicl.yaml` can be overridden the same way.

## Layout

- `src/relicl/`: RelICL itself (`poetry run relicl`).
- `src/hpo/`: hyperparameter search with Optuna (`poetry run hpo`).
- `src/paper/`: builds the paper's tables from the results (`poetry run paper`).
- `config/`: run configurations.
- `examples/rdblearn/`: the comparison with RDBLearn on constructed tasks, see
  its README.
- `examples/early_fusion/`: the OpenML experiment on row embeddings as input
  columns.

## Paper tables

From this repository's root, copy a clone of the results repository over it,
collect all runs into `paper/tex/generated/results.csv`, then build the tables
into `paper/tex/generated/`:

```bash
git clone https://github.com/uma-pi1/relicl-results ../relicl-results
rsync -a --exclude=.git --exclude=/README.md ../relicl-results/ .
poetry run paper collect
poetry run paper tables
poetry run python -m examples.rdblearn.dfs_depth.collect_results
poetry run python -m examples.rdblearn.dfs_pathological.collect_results
poetry run python -m examples.rdblearn.collect_synthetic
```

## Citation

```bibtex
@misc{forbat2026relicltrainingfreerelationallearning,
      title={RelICL: Training-free Relational Learning with Tabular Foundation Models},
      author={Simon Forbat and Rainer Gemulla},
      year={2026},
      eprint={2610.01725},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2610.01725},
}
```
