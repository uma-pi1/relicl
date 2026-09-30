from typing import Any

from optuna.distributions import BaseDistribution
from optuna.samplers import RandomSampler
from optuna.study import Study
from optuna.trial import FrozenTrial


class ResumableRandomSampler(RandomSampler):
    """Optuna's random sampler with its RNG fast-forwarded after a restart.

    Optuna's ``RandomSampler`` keeps its RNG state in memory. Recreating it with the
    same seed therefore starts at the beginning of the sequence. Before this sampler's
    first new draw, it replays one draw for every parameter distribution already stored
    in the study. The restored RNG is then in exactly the state an uninterrupted
    ``RandomSampler`` would have reached.

    This relies only on data Optuna persists for every suggested parameter, including
    parameters of failed, pruned, and abandoned running trials. The HPO entry point is
    sequential, so the persisted trial/parameter order is also the original draw order.
    """

    def __init__(self, seed: int) -> None:
        # Keep RandomSampler's public behavior and distribution handling; only its lost
        # state is reconstructed in sample_independent below.
        super().__init__(seed=seed)
        self._restored = False

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        if not self._restored:
            previous_trials = sorted(
                (
                    previous
                    for previous in study.get_trials(deepcopy=False)
                    if previous.number < trial.number
                ),
                key=lambda previous: previous.number,
            )
            for previous in previous_trials:
                fixed_params = previous.system_attrs.get("fixed_params", {})
                for (
                    previous_name,
                    previous_distribution,
                ) in previous.distributions.items():
                    # Optuna records both of these as trial parameters without asking
                    # the sampler for a value, so neither consumed an RNG draw.
                    if previous_name in fixed_params or previous_distribution.single():
                        continue
                    super().sample_independent(
                        study,
                        previous,
                        previous_name,
                        previous_distribution,
                    )
            self._restored = True

        return super().sample_independent(study, trial, param_name, param_distribution)
