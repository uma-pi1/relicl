import os

import numpy as np
import torch

_rng: np.random.Generator | None = None
_base_seed: int | None = None


def seed(s: int, use_max_reproducibility: bool) -> None:
    global _base_seed
    assert _base_seed is None, "Random number generator already seeded."

    _base_seed = s
    set_stream()  # sets _rng, uses _base_seed

    if use_max_reproducibility:
        # Check env vars.
        if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            raise DeterministicCUDAError()
        if "PYTHONHASHSEED" not in os.environ:
            raise MissingHashSeedError()

        # Set torch to maximum reproducibility.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


def set_stream(*key: int) -> None:
    """Restart the global RNG on the stream identified by `key`.

    Streams with different keys are independent, so draws taken on one of them do
    not shift any of the others. This keeps a part of a run (e.g., one evaluation
    split) reproducible no matter which other parts ran before it.
    """
    global _rng
    assert _base_seed is not None, "Random number generator not seeded."

    _rng = np.random.default_rng(np.random.SeedSequence(_base_seed, spawn_key=key))
    _seed_all()  # uses _rng


def get_rng() -> np.random.Generator:
    global _rng
    assert _rng is not None, "Random number generator not seeded. Call `seed` first."
    return _rng


def random_state() -> int:
    return int(get_rng().integers(2**32 - 1))


def _seed_all():
    _seed_python(random_state())
    _seed_torch(random_state())
    _seed_torch_cuda(random_state())
    _seed_numpy(random_state())
    # _seed_numba(random_state())  # not used


def _seed_python(seed):
    import random

    random.seed(seed)


def _seed_torch(seed):
    import torch

    torch.manual_seed(seed)


def _seed_torch_cuda(seed):
    import torch.cuda

    torch.cuda.manual_seed_all(seed)


def _seed_numpy(seed):
    import numpy.random

    numpy.random.seed(seed)


def _seed_numba(seed):
    import numba
    import numpy as np

    @numba.njit
    def seed_numba_(seed_):
        np.random.seed(seed_)

    seed_numba_(seed)


class DeterministicCUDAError(Exception):
    def __init__(self) -> None:
        super().__init__(
            "The env variable CUBLAS_WORKSPACE_CONFIG must be set "
            "when using deterministic CUDA (e.g., to ':4096:8')."
        )


class MissingHashSeedError(Exception):
    def __init__(self) -> None:
        super().__init__("The env variable PYTHONHASHSEED must be set.")
