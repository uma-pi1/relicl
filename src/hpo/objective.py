import logging
import os
import subprocess
import time
from os import path as osp
from typing import Any

import numpy as np
from hydra.core.hydra_config import HydraConfig
from optuna import Trial, TrialPruned
from optuna.trial import TrialState
from relbench.base import TaskType

from hpo.params import ConditionalGroup, Param
from hpo.utils_hpo import (
    RelICLSubprocessError,
    _format_override,
    _get_effective,
    _get_results,
)

logger = logging.getLogger(__name__)


def _as_text(stream: bytes | str | None) -> str:
    """`TimeoutExpired` is typed for both byte and text mode; the run uses text=True."""
    if stream is None:
        return ""
    return stream if isinstance(stream, str) else stream.decode(errors="replace")


def _sample_trial_seeds(study_seed: int, trial_number: int, n: int) -> list[int]:
    """Return trial-number ``trial_number`` from the original sequential seed stream."""

    rng = np.random.default_rng(study_seed)
    for _ in range(trial_number):
        rng.integers(0, 2**31 - 1, size=n)
    return rng.integers(0, 2**31 - 1, size=n).tolist()


class Objective:
    def __init__(
        self,
        relbench_cfg,
        param_definitions: dict[str, Param],
        conditional_groups: list[ConditionalGroup],
        study_seed: int,
        trial_seeds: Any,
        task_type: TaskType,
        run_test: bool,
        deadline: float | None = None,
        timeout_multiplier: float | None = None,
        timeout_warmup: int = 3,
    ) -> None:
        self._relbench_cfg = relbench_cfg
        self._param_definitions = param_definitions
        self._conditional_groups = conditional_groups
        self._task_type = task_type
        self._run_test = run_test
        self._deadline = deadline
        self._timeout_multiplier = timeout_multiplier
        self._timeout_warmup = timeout_warmup

        self._study_seed = study_seed
        self._trial_seeds = trial_seeds

    def _trial_timeout(self, trial: Trial) -> float | None:
        """Wall-clock cap for one trial, or None when nothing bounds it.

        Two independent bounds, whichever is tighter: the study deadline, so no trial
        can overrun `timeout_h`, and a multiple of the median duration of the trials
        that already completed.
        """

        caps: list[float] = []

        # Deadline bound. Applies from the very first trial, no observations needed.
        if self._deadline is not None:
            caps.append(max(self._deadline - time.monotonic(), 0.0))

        # Outlier bound. Needs enough completed trials to have a stable median, so the
        # warmup trials run bounded only by the deadline.
        if self._timeout_multiplier is not None:
            durations = [
                t.duration.total_seconds()
                for t in trial.study.get_trials(
                    deepcopy=False, states=(TrialState.COMPLETE,)
                )
                if t.duration is not None
            ]
            if len(durations) >= self._timeout_warmup:
                caps.append(self._timeout_multiplier * float(np.median(durations)))

        return min(caps) if caps else None

    def __call__(self, trial: Trial) -> float:
        # Unconditional parameters.
        override_dict = {
            k: param.suggest(trial) for k, param in self._param_definitions.items()
        }

        # Conditional parameters. Conditions match the values suggested above, plus
        # run-level facts like `task_type` that are fixed for the whole study.
        context = dict(task_type=self._task_type.value) | override_dict
        for group in self._conditional_groups:
            if all(context.get(p) == v for p, v in group.conditions):
                for name, param in group.params.items():
                    override_dict[name] = param.suggest(trial)

        # Add RelBench database & task.
        override_dict["relbench.db"] = self._relbench_cfg.db
        override_dict["relbench.task"] = self._relbench_cfg.task

        # Adjust output dir.
        out_path = osp.join(
            HydraConfig.get().runtime.output_dir, f"trial-{trial.number}"
        )
        override_dict["hydra.run.dir"] = out_path

        # Compute validation metrics, and test metrics only if asked for.
        override_dict["run.val"] = True
        override_dict["run.test"] = self._run_test
        override_dict["run.dry"] = False

        # Disable plots.
        override_dict["output.graphs"] = False
        override_dict["output.all_shapes"] = False

        # Keep the trial logs free of progress bars.
        override_dict["output.progress"] = False

        # Reproducibility.
        override_dict["method.use_max_reproducibility"] = True
        os.environ["PYTHONHASHSEED"] = "0"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # Seeds.
        if self._trial_seeds.sample:
            seeds = _sample_trial_seeds(
                self._study_seed, trial.number, self._trial_seeds.n
            )

            if len(seeds) == 1:
                # A single member is not an ensemble.
                override_dict["method.seed"] = seeds[0]
            else:
                # Create ensemble config.
                override_dict["ensemble.enable"] = True
                override_dict["ensemble.average_logits"] = True
                override_dict["ensemble.members"] = [
                    [f"method.seed={seed}"] for seed in seeds
                ]

        # Build hydra override args.
        overrides = [_format_override(k, v) for k, v in override_dict.items()]

        # Run RelICL under a wall-clock cap.
        timeout_s = self._trial_timeout(trial)
        trial.set_user_attr("trial_timeout_s", timeout_s)
        timed_out = False
        try:
            result = subprocess.run(
                ["poetry", "run", "relicl", "run"] + overrides,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            stdout, stderr = result.stdout, result.stderr
        except subprocess.TimeoutExpired as e:
            # The exception carries whatever the run had produced before the kill.
            timed_out = True
            stdout = _as_text(e.stdout)
            stderr = _as_text(e.stderr)

        # Keep the subprocess output. It used to be logged only on failure and
        # discarded otherwise, which left no record of the warnings a successful trial
        # emitted (a raised sampling budget, a dropped context table, a pooling
        # fallback), for exactly the trials worth inspecting.
        os.makedirs(out_path, exist_ok=True)
        with open(osp.join(out_path, "trial.log"), "w") as f:
            f.write(stdout)
            if stderr:
                f.write("\n=== stderr ===\n")
                f.write(stderr)

        # Catch overrun. Pruning rather than failing keeps the trial in the study as a
        # record while excluding it from best-trial selection.
        if timed_out:
            logger.warning(
                "RelICL exceeded its %.0fs budget and was killed; pruning trial %d",
                timeout_s,
                trial.number,
            )
            raise TrialPruned

        # Catch failure.
        if result.returncode != 0:
            logger.error("RelICL failed\nstdout:\n%s\n\nstderr:\n%s", stdout, stderr)
            raise RelICLSubprocessError

        # Parse return values.
        results = _get_results(out_path)

        # Log all metrics. The test split is empty unless it was run.
        for k, v in results["val"].items():
            trial.set_user_attr(f"val_{k}", v)
        for k, v in results["test"].items():
            trial.set_user_attr(f"test_{k}", v)

        # Log where the run departed from the config this trial asked for. Without
        # these, a parameter that was inert on a task is indistinguishable from one
        # that was applied and did not help.
        for k, v in _get_effective(out_path).items():
            trial.set_user_attr(f"effective_{k}", v)

        # Log test performance (duplicated; for ease-of-use).
        if self._run_test:
            test_performance = results["test"][self._relbench_cfg.hp_metric]
            logger.info(f"test performance: {test_performance}")
            trial.set_user_attr("test_value", test_performance)

        # Optimize based on validation performance.
        val_performance = results["val"][self._relbench_cfg.hp_metric]
        logger.info(f"val performance: {val_performance}")
        return val_performance
