import argparse
import logging
import os
import os.path as osp
import shutil
import subprocess
import sys
from collections import OrderedDict
from typing import Any, Mapping, MutableMapping, cast

import hydra
import matplotlib
import numpy as np
import pandas as pd
import yaml
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from termcolor import colored

import relicl.ensemble
from relicl import rng
from relicl.config import (
    MEMBER_OVERRIDES_FILENAME,
    NOT_EXISTENT,
    REPO_ROOT,
    RESULTS_FILENAME,
    RelICLConfig,
    flatten_and_diff,
)
from relicl.effective_config import EffectiveConfig
from relicl.ensemble import Predictions, diff_pred
from relicl.rdl import RDLRunner
from relicl.rewrite import apply_rewrites
from relicl.schema import RDLTask
from relicl.timing import timed_step
from relicl.trace import Trace
from relicl.utils import (
    compute_binary_classification_metrics,
    compute_member_spread,
    compute_regression_metrics,
    get_git_revision_short_hash,
    store_rdl_task_graph_to_disk,
)

matplotlib.use("Agg")  # for graph plotting; speeds things up

logger = logging.getLogger(__name__)

# ######################################################################################
# Main function. #######################################################################
# ######################################################################################


def main() -> None:
    """Dispatcher to main RelICL functionalities"""

    # all arguments should be passed on to the actual command
    # hack: prefix_chars='ä' to turn off prefix_chars for optional arguments
    parser = argparse.ArgumentParser("relicl", add_help=False, prefix_chars="ä")
    parser.add_argument(
        "command",
        choices=[
            "run",
            "create",
            "resume",
            "val",
            "test",
            "valtest",
            "diff",
            "ensemble",
        ],
    )
    parser.add_argument("args", nargs="*")
    parser.exit_on_error = False
    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        parser.print_usage()
        print(e)
        print("""
run       Create an RelICL experiment folder and run RelICL
create    Create a new RelICL experiment folder, but do not run RelICL
          (To override default folder, use hydra.run.dir=<folder>.)
resume    Run or rerun configured splits in a RelICL experimental folder
val       Run or rerun validation split in a RelICL experiment folder
test      Run or rerun test split in a RelICL experiment folder
valtest   Run or rerun validation and test split in a RelICL experiment folder
diff      Compare the results in two (or more) RelICL experiment folders
ensemble  Ensemble the results of two (or more) RelICL experiment folders
        """)
        exit(1)

    match args.command:
        case "run":
            package, method = "relicl.cli", "run"
        case "create":
            package, method = "relicl.cli", "create"
        case "resume":
            package, method = "relicl.cli", "resume"
        case "val":
            package, method = "relicl.cli", "val"
        case "test":
            package, method = "relicl.cli", "test"
        case "valtest":
            package, method = "relicl.cli", "valtest"
        case "diff":
            package, method = "relicl.cli", "diff"
        case "ensemble":
            package, method = "relicl.cli", "ensemble"
        case _:
            raise ValueError(f"Unknown command {args.command}")

    sys.argv = [sys.argv[0], *args.args]
    exec(f"from {package} import {method}")
    return eval(f"{method}()")


# ######################################################################################
# Run command. #########################################################################
# ######################################################################################

cs = ConfigStore.instance()
cs.store(name="base_config", node=RelICLConfig)


def _run_relicl(args, **kwargs):
    # The flag `check=True` makes a run fail when it encounters a non-zero exit code.
    subprocess.run(
        ["poetry", "-P", REPO_ROOT, "run", "relicl", *args], check=True, **kwargs
    )


@hydra.main(
    version_base="1.3", config_path=str(REPO_ROOT / "config"), config_name="relicl"
)
def run(config: RelICLConfig) -> None:
    if config.ensemble.enable:
        run_ensemble(config)
    else:
        run_member(config)


def run_ensemble(config: RelICLConfig) -> None:
    RelICLConfig.set(config)
    assert config.ensemble.enable
    assert len(config.ensemble.members) > 0
    trace = Trace.instance()
    trace.trace(
        "start",
        revision=get_git_revision_short_hash(),
        db=config.relbench.db,
        task=config.relbench.task,
        config=OmegaConf.to_container(config, resolve=True),
    )

    ensemble_overrides = list(HydraConfig.get().overrides.task)

    # create the ensemble members
    d = osp.abspath(HydraConfig.get().runtime.output_dir)
    member_dirs = []
    for i, member in enumerate(config.ensemble.members):
        print(f"Member {i:02}: {member}")

        # Members are Hydra overrides.
        overrides = [*ensemble_overrides, "ensemble.enable=false", *member]

        # create the ensemble member
        member_dir = osp.join(d, f"member_{i:02}")
        member_dirs.append(member_dir)
        overrides_file = osp.join(member_dir, MEMBER_OVERRIDES_FILENAME)
        if not osp.exists(member_dir):
            _run_relicl(
                [
                    "create",
                    *overrides,
                    f"hydra.run.dir={member_dir}",
                    f"hydra.job.id={HydraConfig.get().job.id}/member_{i:02}",
                ],
                cwd=d,
            )
            with open(overrides_file, "w") as f:
                yaml.safe_dump(overrides, f)
        elif osp.isfile(overrides_file):
            # If already present, verify it was created from the same overrides.
            old_overrides = yaml.safe_load(open(overrides_file))
            if old_overrides != overrides:
                raise ValueError(
                    f"Prior and current member_{i:02} overrides differ:\n"
                    f"  prior:   {old_overrides}\n"
                    f"  current: {overrides}"
                )
        else:
            # Members created before the file existed are reused unverified.
            logger.warning(
                f"member_{i:02} has no {MEMBER_OVERRIDES_FILENAME}; cannot verify that "
                f"it matches the current ensemble config."
            )

    # run the ensemble members
    for i, member_dir in enumerate(member_dirs):
        # Hydra composed the member, so read back what it resolved to.
        member_config = OmegaConf.load(osp.join(member_dir, ".hydra/config.yaml"))
        mode = "val" if member_config.run.val else ""
        mode += "test" if member_config.run.test else ""
        assert mode != ""
        _heading(f"Running member {i:02} ({mode})")
        _run_relicl([mode, member_dir], cwd=d)

    # produce the result
    try:
        _ensemble_collect_sources(member_dirs, "val")
        run_val = True
    except:
        run_val = False

    try:
        _ensemble_collect_sources(member_dirs, "test")
        run_test = True
    except:
        run_test = False

    mode = "val" if run_val else ""
    mode += "test" if run_test else ""
    assert mode != ""
    _heading(f"Ensembling ({mode})")
    _run_relicl(
        [
            "ensemble",
            *member_dirs,
            "--overwrite",
            f"--average-logits={config.ensemble.average_logits}",
            f"--mode={mode}",
        ],
        cwd=d,
    )

    trace.trace("stop")


def run_member(config: RelICLConfig) -> None:
    try:
        # Initialize.
        RelICLConfig.set(config)
        assert not config.ensemble.enable
        EffectiveConfig.reset()

        with timed_step("full-program"):
            # Log start.
            trace = Trace.instance()
            trace.trace(
                "start",
                revision=get_git_revision_short_hash(),
                db=config.relbench.db,
                task=config.relbench.task,
                config=OmegaConf.to_container(config, resolve=True),
            )

            # Init randomness.
            rng.seed(config.method.seed, config.method.use_max_reproducibility)

            # Load RDL task from relbench.
            rdl_task = RDLTask.from_relbench()
            trace.trace("schema", **rdl_task.description())
            logger.info(
                "Schema:\n"
                + yaml.dump(
                    rdl_task.description(eager=config.output.all_shapes),
                    width=float("inf"),
                    default_flow_style=False,
                ),
            )
            if config.output.graphs:
                store_rdl_task_graph_to_disk(
                    rdl_task, "schema-original.pdf", shapes=config.output.all_shapes
                )

            # Apply all rewrites and potentially simplify
            apply_rewrites(rdl_task)
            trace.trace("rewritten_schema", **rdl_task.description())
            logger.info(
                f"Rewritten schema (simplify={config.rewrite.simplify}):\n"
                + yaml.dump(
                    rdl_task.description(eager=config.output.all_shapes),
                    width=float("inf"),
                    default_flow_style=False,
                )
            )
            if config.output.graphs:
                store_rdl_task_graph_to_disk(
                    rdl_task, "schema-rewritten.pdf", shapes=config.output.all_shapes
                )

            # Run RDL task.
            rdl_runner = RDLRunner()
            results = rdl_runner.run(rdl_task)
            trace.trace("stop")

            # Write final value to disk.
            with open(
                osp.join(HydraConfig.get().runtime.output_dir, RESULTS_FILENAME), "w"
            ) as f:
                yaml.dump(results, f)

            # Dump effective config ("which HPs are different to the ones configured?")
            # to disk.
            EffectiveConfig.dump(HydraConfig.get().runtime.output_dir)

    except Exception as e:
        logger.exception(e)
        raise


# ######################################################################################
# Create command. ######################################################################
# ######################################################################################


@hydra.main(
    version_base="1.3", config_path=str(REPO_ROOT / "config"), config_name="relicl"
)
def create(_: RelICLConfig) -> None:
    print(f"Created folder {HydraConfig.get().runtime.output_dir}")


# ######################################################################################
# Val command. #########################################################################
# ######################################################################################


def _is_relicl_directory(d: str) -> bool:
    if not osp.isdir(d):
        return False
    if not osp.isfile(osp.join(d, ".hydra/config.yaml")):
        return False
    return True


def _metrics_from_predictions(pred: Predictions) -> dict[str, Any]:
    if pred.is_regression:
        return compute_regression_metrics(
            pred.df["y_true"].to_numpy(),
            pred.df["y_pred"].to_numpy(),
            print_metrics=False,
        )
    # TODO: No overload variant of "stack" matches argument types "Series[Any]", "int"
    return compute_binary_classification_metrics(
        pred.df["y_true"].to_numpy(),
        np.stack(pred.df["proba"], axis=0),  # type: ignore
        print_metrics=False,
    )


def _recompute_results_from_file(f: str) -> dict[str, Any]:
    return _metrics_from_predictions(Predictions.load(f))


def _recompute_results(d: str) -> None:
    val_file = osp.join(d, "predictions-val.parquet")
    val_metrics = _recompute_results_from_file(val_file) if osp.exists(val_file) else {}
    test_file = osp.join(d, "predictions-test.parquet")
    test_metrics = (
        _recompute_results_from_file(test_file) if osp.exists(test_file) else {}
    )
    results = dict(val=val_metrics, test=test_metrics)
    with open(osp.join(d, RESULTS_FILENAME), "w") as f:
        yaml.dump(results, f)


def _backup(d: str) -> str:
    n = 1
    while osp.exists(osp.join(d, f".backup/{n:03}")):
        n += 1
    backup_dir = osp.join(d, f".backup/{n:03}")
    shutil.copytree(d, backup_dir, ignore=lambda _, __: ".backup")
    return backup_dir


def _is_completed(d: str, mode: str):
    # No predictions -> not completed
    if not osp.exists(osp.join(d, f"predictions-{mode}.parquet")):
        return False

    # Predictions are there, but are they complete?
    results_file = osp.join(d, RESULTS_FILENAME)
    if not osp.exists(results_file):
        return False

    result = yaml.load(open(osp.join(d, RESULTS_FILENAME)), Loader=yaml.SafeLoader)
    # "n" is present in both classification and regression metrics.
    if mode in result and "n" in result[mode]:
        return True
    else:
        return False


def rerun(mode: str | None) -> None:
    parser = argparse.ArgumentParser(f"relicl {mode}")
    parser.add_argument("dir", nargs="?", default=os.getcwd())
    parser.add_argument("--no-skip-existing", action="store_true", default=False)
    args = parser.parse_args()
    args.dir = str(osp.abspath(args.dir))

    val_file = osp.join(args.dir, "predictions-val.parquet")
    test_file = osp.join(args.dir, "predictions-test.parquet")
    results_file = osp.join(args.dir, RESULTS_FILENAME)
    if mode is not None:
        run_val = "val" in mode
        run_test = "test" in mode
    else:
        config = OmegaConf.load(osp.join(args.dir, ".hydra/config.yaml"))
        run_val = config.run.val
        run_test = config.run.test
        mode = "val" if run_val else ""
        mode += "test" if run_test else ""

    print(f"Rerunning {mode} for {args.dir}")
    if not args.no_skip_existing:
        if run_val and _is_completed(args.dir, "val"):
            print(f"Skipping val as it's already completed.")
            run_val = False
        if run_test and _is_completed(args.dir, "test"):
            print(f"Skipping test as it's already completed.")
            run_test = False
    if not (run_val or run_test):
        print("Nothing to do.")
        return

    # backup and remove files that will be recreated
    _backup(args.dir)
    if run_val and osp.exists(val_file):
        print(f"Removing prior {val_file}.")
        os.remove(val_file)
    if run_test and osp.exists(test_file):
        print(f"Removing prior {test_file}.")
        os.remove(test_file)

    # append defaults list
    job_id = yaml.load(
        open(osp.join(args.dir, ".hydra/hydra.yaml")), Loader=yaml.SafeLoader
    )["hydra"]["job"]["id"]
    shutil.copy(
        osp.join(args.dir, ".hydra/config.yaml"),
        osp.join(args.dir, ".hydra/config-with-defaults.yaml"),
    )
    with open(osp.join(args.dir, ".hydra/config-with-defaults.yaml"), "a") as f:
        f.write("\ndefaults: [base_config,_self_]\n")
    sys.argv = [
        sys.argv[0],
        f"--config-path={args.dir}/.hydra",
        "--config-name=config-with-defaults.yaml",
        f"hydra.job.id={job_id}",
        f"hydra.run.dir={args.dir}",
        f"run.val={run_val}",
        f"run.test={run_test}",
    ]
    # noinspection PyArgumentList
    run()

    # recompute results
    if osp.exists(osp.join(args.dir, RESULTS_FILENAME)):
        print(f"Updating prior results file {results_file}.")
        os.remove(results_file)
    _recompute_results(args.dir)


def resume() -> None:
    rerun(None)


def val() -> None:
    rerun("val")


def test() -> None:
    rerun("test")


def valtest() -> None:
    rerun("valtest")


# ######################################################################################
# diff command. ########################################################################
# ######################################################################################


def _heading(s: str) -> None:
    print(colored(f"\n{s}", attrs=["bold"]))


def _collect_files(dirs: list[str], filename: str) -> list[str]:
    result: list[str] = []
    for d in dirs:
        if not osp.isdir(d):
            raise IOError(f"{d} is not a directory")

        f = osp.join(d, filename)
        if not osp.isfile(f):
            raise IOError(f"{f} not found")

        result.append(f)

    return result


def _print_diff(diff: Mapping[str, list[Any]]) -> None:
    for k, v in diff.items():
        print(f"{k}: ", end="")
        yaml.dump(v, sys.stdout, width=float("inf"), default_flow_style=True)


def _diff_configs(inputs: list[str], is_dirs: bool = False) -> None:
    if is_dirs:
        inputs = _collect_files(inputs, ".hydra/config.yaml")
    _diff = flatten_and_diff([cast(DictConfig, OmegaConf.load(f)) for f in inputs])

    # Keys every input has are a genuine difference in settings; keys only some inputs
    # have say that the configs differ in shape, which is far less interesting and would
    # otherwise bury the former.
    _heading("Configuration differences")
    _print_diff({k: v for k, v in _diff.items() if NOT_EXISTENT not in v})

    absent = {k: v for k, v in _diff.items() if NOT_EXISTENT in v}
    if absent:
        _heading("Configuration keys absent from some inputs")
        _print_diff(absent)


def _diff_results(inputs: list[str], is_dirs: bool = False) -> None:
    if is_dirs:
        inputs = _collect_files(inputs, RESULTS_FILENAME)
    _diff = flatten_and_diff(
        [yaml.load(open(f), Loader=yaml.SafeLoader) for f in inputs]
    )
    _heading("Result differences")
    for k, v in _diff.items():
        print(f"{k}: ", end="")
        yaml.dump(v, sys.stdout, width=float("inf"), default_flow_style=True)


def _diff_predictions(
    inputs: list[str], is_dirs: bool = False, mode: str = "test"
) -> None:
    if is_dirs:
        inputs = _collect_files(inputs, f"predictions-{mode}.parquet")
    preds = [Predictions.load(f) for f in inputs]
    _diff = diff_pred(preds)
    _heading(f"Predictions ({mode})")
    for k, v in _diff.items():
        print(f"{k}: ", end="")
        if k != "predictions":
            yaml.dump(v, sys.stdout, width=float("inf"), default_flow_style=True)
        else:
            print()
            for row in v:
                print("- ", end="")
                yaml.dump(row, sys.stdout, width=float("inf"), default_flow_style=True)


def diff() -> None:
    parser = argparse.ArgumentParser("relicl diff")
    parser.add_argument("input", nargs="+")
    parser.add_argument("--with-predictions", action="store_true", default=False)
    args = parser.parse_args()
    if len(args.input) < 2:
        print("Error: only one input provided", file=sys.stderr)
        exit(1)

    _heading("Inputs")
    yaml.dump(args.input, sys.stdout)

    _diff_configs(args.input, is_dirs=True)
    _diff_results(args.input, is_dirs=True)
    if args.with_predictions:
        # TODO: This should use typing.EvalMode.
        for mode in ["val", "test"]:
            try:
                _diff_predictions(args.input, is_dirs=True, mode=mode)
            except IOError as e:
                _heading(f"Predictions ({mode})")
                print(f"Cannot diff predictions: {e}")


# ######################################################################################
# ensemble command. ####################################################################
# ######################################################################################


def _ensemble_collect_sources(
    names: list[str], mode: str, require_dir: bool = True
) -> list[str]:
    sources = []
    for name in names:
        if osp.isdir(name):
            name = osp.join(name, f"predictions-{mode}.parquet")
        elif require_dir:
            raise ValueError(
                f"--mode=valtest but an input file (instead of a results folder) is specified"
            )

        if osp.isfile(name):
            if osp.splitext(name)[1] not in (".parquet", ".yaml"):
                raise ValueError(f"{name} is not a .parquet (or legacy .yaml) file")

            sources.append(name)
        else:
            raise ValueError(f"{name} not found")

    return sources


def ensemble() -> None:
    parser = argparse.ArgumentParser("relicl ensemble")
    parser.add_argument("input", nargs="+")
    parser.add_argument(
        "--mode", nargs="?", choices=["val", "test", "valtest"], default="test"
    )
    parser.add_argument("--average-logits", nargs="?", type=str2bool, default=True)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--no-output", action="store_true", default=False)
    args = parser.parse_args()
    if len(args.input) < 2:
        print(
            "Warning: only one input provided, no ensembling is done.", file=sys.stderr
        )

    # collect sources
    # TODO: This should use typing.EvalMode.
    modes = ["val", "test"] if args.mode == "valtest" else [args.mode]
    sources: MutableMapping[str, list[str]] = OrderedDict()
    for mode in modes:
        sources[mode] = _ensemble_collect_sources(args.input, mode, len(modes) > 1)
    dirs = [osp.dirname(source) for source in sources[modes[0]]]

    _heading("Inputs")
    yaml.dump(sources, sys.stdout)

    # some diffs
    if len(dirs) > 1:
        try:
            _diff_configs(dirs, is_dirs=True)
        except IOError:
            logger.exception("IOError on the directories: %s", dirs)

        try:
            _diff_results(dirs, is_dirs=True)
        except IOError:
            logger.exception("IOError on the directories: %s", dirs)

    # check whether it's safe to write
    outputs: MutableMapping[str, str] = OrderedDict()
    if not args.no_output:
        for mode in modes:
            outfile = f"predictions-{mode}.parquet"
            if not args.overwrite and osp.exists(outfile):
                print(
                    f"Error: output file {outfile} exists and "
                    "neither --overwrite nor --no-output is not specified",
                    file=sys.stderr,
                )
                exit(1)
            outputs[mode] = outfile

        if not args.overwrite and osp.exists(RESULTS_FILENAME):
            print(
                f"Error: results file {RESULTS_FILENAME} exists and "
                "neither --overwrite nor --no-output is not specified",
                file=sys.stderr,
            )
            exit(1)
        outputs["results"] = RESULTS_FILENAME

    _heading("Outputs")
    yaml.dump(outputs, sys.stdout)

    # do the ensembling
    # Both splits are always written, empty for the one that was not ensembled, so the
    # result file has the same shape as the one a single member writes.
    metrics: MutableMapping[str, dict[str, Any]] = OrderedDict(val={}, test={})
    for mode in modes:
        # load traces
        preds: list[Predictions] = []
        for source in sources[mode]:
            preds.append(Predictions.load(source))

        # ensemble them
        if len(preds) > 1:
            out = relicl.ensemble.ensemble_predictions(
                preds, average_logits=args.average_logits
            )
        else:
            out = preds[0]
        assert out.eval_mode == mode, (
            f"Files specified for {mode} are actually {out.eval_mode}"
        )

        # compute output statistics
        if out.is_regression:
            metrics[mode] = compute_regression_metrics(
                out.df["y_true"].to_numpy(),
                out.df["y_pred"].to_numpy(),
                print_metrics=False,
            )
        else:
            # TODO: No overload variant of "stack" matches argument types
            #  "Series[Any]", "int" [call-overload]
            metrics[mode] = compute_binary_classification_metrics(
                out.df["y_true"].to_numpy(),
                np.stack(out.df["proba"], axis=0),  # type: ignore
                print_metrics=False,
            )

        # How far the members disagree.
        metrics[mode].update(
            compute_member_spread([_metrics_from_predictions(p) for p in preds])
        )

        # store the results
        if not args.no_output:
            out.save(outputs[mode])

    # output the final results
    _heading("Results")
    yaml.dump(metrics, sys.stdout)
    if not args.no_output:
        with open(outputs["results"], "w") as f:
            yaml.dump(metrics, f)


# ######################################################################################
# yaml2csv script ######################################################################
# ######################################################################################


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def yaml2csv() -> None:
    parser = argparse.ArgumentParser("yaml2csv")
    parser.add_argument("filename", nargs="?", default=None)
    parser.add_argument(
        "--expand",
        help="expand all array-valued fields to individual fields (e.g., proba TO proba_0, proba_1)",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()

    f = open(args.filename, "rt") if args.filename is not None else sys.stdin

    records: list[MutableMapping[str, Any]] = []
    for line in f:
        record = yaml.load(line, Loader=yaml.SafeLoader)
        new_record = OrderedDict()
        for key, value in record.items():
            if args.expand and isinstance(value, list):
                for i, entry in enumerate(value):
                    new_record[key + f"_{i}"] = value[i]
            else:
                new_record[key] = value

        records.append(new_record)

    try:
        pd.DataFrame.from_records(records).to_csv(sys.stdout, index=False)
    except BrokenPipeError:
        # when output is used in pipe and pipe is closed (e.g., "json2csv | head"), do
        # not throw an error
        sys.exit(0)


# Entry-point. #########################################################################

if __name__ == "__main__":
    main()
