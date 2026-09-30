import re

from relicl.config import REPO_ROOT

# Inputs. ##############################################################################

# Data dirs.
EXPERIMENTS_DIR = REPO_ROOT / "paper" / "experiments"
OUTPUTS_SUBDIR = "outputs"

# RelBench dataset/task stats, produced by
# `paper/relbench-stats/relbench_task_stats.py` and checked in as CSV.
DATASET_STATS_DIR = REPO_ROOT / "paper" / "relbench-stats"

# Early fusion. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# The OpenML experiment behind `app:early-fusion`, which replaces a table's feature
# columns by a row embedding. Both files are produced in `examples/early_fusion/` and
# checked in: the runs by `early_fusion_seeds.py`, the table statistics by
# `early_fusion_dataset_stats.py`.
EARLY_FUSION_RESULTS = EXPERIMENTS_DIR / "early-fusion-seeds-reduced-vs-unreduced.tsv"
EARLY_FUSION_DATASETS = EXPERIMENTS_DIR / "early-fusion-datasets.csv"

# HPO. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HPO_OUTPUTS_SUBDIR = "hpo-outputs"
HPO_STAGE = "hpo"  # Used to denote HPO runs in output CSV.
SORTED_RESULTS_GLOB = "*/_ANALYSIS/*-results_sorted.csv"  # Used to collect results.
TEST_DIR_SUFFIX = "-test"  # Contains the test runs.

# Baselines. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Published numbers, maintained by hand in CSV.
BASELINES_FILE = REPO_ROOT / "paper" / "baselines.csv"
BASELINE_COLUMNS = [
    "model",
    "kind",
    "db",
    "task",
    "split",
    "metric",
    "value",
    "source",
    "note",
]
MODEL_ORDER = [
    # "GraphSAGE",
    # "RelGNN",
    # "RelGT",
    # "DS+LightGBM",
    # "KumoRFM-2",
    "RT",
    "PluRel",
    "RDBLearn",
    "TabPFN-REL-Local",
]
# Every published model, for the appendix table that compares all of them.
ALL_MODEL_ORDER = [
    "GraphSAGE",
    "RelGNN",
    "RelGT",
    "DS+LightGBM",
    "RT",
    "RT-pretrained-on-target-db",
    "PluRel",
    "KumoRFM-2",
    "RDBLearn",
    "TabPFN-REL-Client",
    "TabPFN-REL-Local",
]
# How a model is headed in a table, as against the name `baselines.csv` stores it under.
MODEL_LABEL = {
    "RT-pretrained-on-target-db": "RT (target DB)",
    "TabPFN-REL-Client": "TabPFN-Rel (API)",
    "TabPFN-REL-Local": "TabPFN-Rel",
}

# Outputs. #############################################################################

GENERATED_DIR = REPO_ROOT / "paper" / "tex" / "generated"
RESULTS_FILE = GENERATED_DIR / "results.csv"

# Reading one run directory. ###########################################################

START_EVENT_LINE = 0  # Which line of the trace.yaml contains the start event.
DURATION_RE = re.compile(
    r"DURATION-full-program.*\bduration_s:\s*([0-9.]+)"
)  # RE to obtain a run's duration.
CONFIG_PREFIX = "config."  # How to prefix config variables in the output.
NON_SCALAR_METRICS = ("confusion_matrix",)  # Metrics excluded from the output CSV.

# Metrics. #############################################################################

# Headline metric = main metric to report
HEADLINE_METRIC = dict(regression="mae", binary_classification="auroc")
LOWER_IS_BETTER = dict(mae=True, auroc=False, r2=False)
HEADLINE_COLUMNS = (
    "mae",
    "auroc",
    "mae_member_mean",
    "auroc_member_mean",
    "mae_member_std",
    "auroc_member_std",
    "mae_member_sem",
    "auroc_member_sem",
)
SPLITS = ("val", "test")
RESULT_KEYS = ["stage", "db", "task", "variant"]  # Identifying columns of a single run.

# Metrics the main results table reports, in the order they appear inside a cell. The
# headline metric of the task type comes first. Where there are several, every cell of
# the table carries them all, the summary rows included.
REPORTED_METRICS = dict(regression=("mae", "r2"), binary_classification=("auroc",))

# Separates the metrics inside one cell.
CELL_SEPARATOR = " / "

# Tables. ##############################################################################

# Our models. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Our models, appended after MODEL_ORDER in this order.
SEED_ENSEMBLE = "RelICL (TabICL, seed ens.)"
TABPFN = "RelICL (TabPFN)"
TABPFN_SEED_ENSEMBLE = "RelICL (TabPFN, seed ens.)"
# `backbone` names the tabular model behind the run and goes into file names of the
# per-backbone tables.
OURS = {
    "RelICL (TabICL)": dict(
        stage="final-multiple-configs", kind="pretrained-tabular", backbone="tabicl"
    ),
    TABPFN: dict(
        stage="final-multiple-configs-tabpfn",
        kind="pretrained-tabular",
        backbone="tabpfn",
    ),
    SEED_ENSEMBLE: dict(stage="final", kind="pretrained-tabular", backbone="tabicl"),
    TABPFN_SEED_ENSEMBLE: dict(
        stage="final-tabpfn", kind="pretrained-tabular", backbone="tabpfn"
    ),
}

# The model the main tables report (e.g., the HPO-per-task reference).
MAIN_MODEL = "RelICL (TabICL)"
# What a table calls our model when it holds a single variant of it, whose backbone
# then need not be named.
OURS_LABEL = "RelICL"
# The columns the main results table reports. Only the tuned backbone competes against
# the baselines there; the other backbone gets a table of its own (BACKBONE_MODELS).
MAIN_MODELS = [MAIN_MODEL]
# The columns of the backbone table, which compares our own backbones against each
# other and so reports no baseline.
BACKBONE_MODELS = [MAIN_MODEL, TABPFN]
# The columns of the appendix's seed-ensemble table (i.e., pre-config-ensembling),
# same backbone order as BACKBONE_MODELS.
SEED_ENSEMBLE_MODELS = [SEED_ENSEMBLE, TABPFN_SEED_ENSEMBLE]

# Which performance value to report:
# - "value" (ensembled prediction)
# - "seed_mean" (mean over seeds)
OURS_VALUE = "value"

# Config ensembling. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# The member in every recipe: `member_00` of the three-seed run.
ENSEMBLE_BASELINE = "Full"

# Candidates on top of the baseline. Keep in step with `scripts/run-paper*.sh`.
ENSEMBLE_MEMBERS = (
    "no-count-features",
    "count-windows",
    "no-simplify",
    "reduce-per-estimator",
)

# Where a candidate run is looked up per backbone, first hit wins.
ENSEMBLE_MEMBER_STAGES = dict(
    tabicl=("ablations/{task}/{member}/member_00", "config-ensemble/{task}/{member}"),
    tabpfn=("config-ensemble-tabpfn/{task}/{member}",),
)

# The recipes `paper select` picked, read by `paper assemble`.
RECIPES_FILE = GENERATED_DIR / "config-ensemble-recipes.yaml"

# Formatting. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Used for handling name collisions of tasks: If no collision, use only the task name.
# If there is a collision: Append DB_LABEL_CHARS of the DB to the task name.
DB_LABEL_PREFIX = "rel-"
DB_LABEL_CHARS = 3
DB_LABEL_FORMAT = "{task} ({db})"

# What to put for missing values.
MISSING_CELL = "--"

# LaTeX.
BOLD = "\\textbf{{{text}}}"
UNDERLINE = "\\underline{{{text}}}"
TASK_HEADER = "\\rotatebox[origin=l]{{90}}{{{task}}}"  # requires `graphicx`

# How a kind is headed in a table, as against the key `baselines.csv` stores it under.
KIND_LABEL = {
    "trained-gnn": "Trained",
    "trained-gt": "Trained",
    "trained-gbdt": "Trained",
    "pretrained-relational": "Pretrained (rel.)",
    "pretrained-tabular": "Pretrained (tab.)",
}

# The column the tasks, the kinds and the models are named in.
TASK_COLUMN = "Task"
KIND_COLUMN = "Kind"
MODEL_HEADER = "Model"

# Column alignment.
LABEL_SPEC = "l"
NUMBER_SPEC = "c"
# A column whose cells carry several metrics ("0.04 / 18.3") becomes two table columns
# joined by the separator, so that the metrics line up on either side of it.
PAIRED_SPEC = r"r@{\;/\;}l"

RANK_COLUMN = "$\\emptyset$ Rank"
WINS_COLUMN = "\\# Wins"
MEAN_COLUMN = "$\\emptyset$"
RANK_DECIMALS = 2
SUMMARY_COLUMNS = (WINS_COLUMN, RANK_COLUMN, MEAN_COLUMN)

# Summary rows/columns.
SUMMARY_HEADERS = {
    WINS_COLUMN: f"{WINS_COLUMN} $(\\uparrow)$",
    RANK_COLUMN: f"{RANK_COLUMN} $(\\downarrow)$",
}
SUMMARY_RULE = "\\midrule"  # only for rows

# Rules the tables are cut with. `GROUP_RULE` underlines one kind of the header row,
# `TOP_RULE` is what `to_latex` writes above that header row.
TOP_RULE = "\\toprule"
MID_RULE = "\\midrule"  # What `to_latex` writes below the header rows.
GROUP_RULE = "\\cmidrule(lr){{{first}-{last}}}"  # requires `booktabs`

# Number formatting.
PERCENT_METRICS = ("auroc", "r2")
PERCENT_DECIMALS = 1
VALUE_DECIMALS = 2

# Seed spread. One table per model in `SEED_ENSEMBLE_MODELS`.
SEED_VALUE_COLUMNS = dict(value="Ensemble", seed_mean="Mean")
SEED_SPREAD_COLUMNS = dict(seed_std="SD", seed_sem="SEM")
SEED_SPREAD_OF = "seed_mean"
SPREAD_FORMAT = "{value} $\\pm$ {spread}"

# Per-task HPO winners. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HPO_BEST_LABEL = "Per-task HPO"  # Heads the winning trial's columns.
HPO_REFERENCE_STAGE = str(OURS[MAIN_MODEL]["stage"])  # What the paper reports.
HPO_SPLIT = "test"  # The table reports test only, so the caption names the split.
HPO_GAIN_COLUMN = "$\\Delta$"  # Positive: tuning per task won.

# Wall clock cost.
TIME_COLUMN = "$\\emptyset$ Time (geom.~mean)"
TIME_FORMAT = "{ratio:.2f}$\\times$"

# Win matrices. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# One entry per matrix, keyed by the name its file carries. `published` names the
# baselines taken from `baselines.csv`, `models` our own runs keyed in `OURS`.
WIN_MATRIX_TABLES = {
    "win-matrix": dict(published=MODEL_ORDER, models=MAIN_MODELS),
    "win-matrix-all": dict(published=ALL_MODEL_ORDER, models=MAIN_MODELS),
}

# The split the matrices report: the published baselines mostly report test only.
WIN_MATRIX_SPLIT = "test"

# A cell reads "wins/shared": the tasks the row model won, of the tasks the pair shares.
WIN_MATRIX_CELL = r"{wins}\,/\,{shared}"

# Heads the last column, which counts the pairs a model wins out of those it is in.
# Only the full matrices carry it, and only with `--pairs-won`.
WIN_MATRIX_SUMMARY = "Pairs won $(\\uparrow)$"

# Marks a cell that a fallback metric decided, i.e. not the first entry of
# `REPORTED_METRICS`. Regression falls back from MAE to $R^2$ for the models that
# report only the latter.
# It takes no width, so that it does not push its cell off center.
WIN_MATRIX_FALLBACK = r"\rlap{$^{*}$}"

# Early fusion. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# How the runs name themselves in `EARLY_FUSION_RESULTS`.
EARLY_FUSION_BASELINE = "BASELINE"
# The row-embedding settings, in table order, and how each is headed.
EARLY_FUSION_SETTINGS = {
    "un-reduced": "$\\tfm(\\tenc(R))$",
    "reduced": "$\\tfm(\\operatorname{reduce}(\\tenc(R)))$",
}
# The settings of the main-text table; the appendix table reports all of them.
EARLY_FUSION_MAIN_SETTINGS = ("un-reduced",)
EARLY_FUSION_BASELINE_COLUMN = "$\\tfm(R)$"
EARLY_FUSION_METRIC = "auroc"
EARLY_FUSION_METRIC_COLUMN = "AUROC"
EARLY_FUSION_DELTA_COLUMN = "$\\Delta$"
# The round a run belongs to.
EARLY_FUSION_ROUND = "round"
EARLY_FUSION_SUMMARY = MEAN_COLUMN
EARLY_FUSION_SPREAD_FORMAT = SPREAD_FORMAT

# Columns of the dataset-statistics table, in table order.
EARLY_FUSION_DATASET_COLUMNS = {
    "dataset": "Dataset",
    "openml_id": "OpenML ID",
    "rows": "Rows",
    "features": "Features",
    "pos_rate": "Positive class (\\%)",
}
EARLY_FUSION_DATASET_SPEC = "lrrrr"

# Preamble for generated outputs.
TABLE_PREAMBLE = "% Generated by `poetry run paper tables`. Do not edit.\n"
