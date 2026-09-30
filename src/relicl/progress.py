from typing import Any

import tqdm as tqdm_lib

from relicl.config import RelICLConfig


def _disable() -> bool:
    """Progress bars are silenced by `output.progress=false`."""
    return not RelICLConfig.instance().output.progress


def tqdm(*args: Any, **kwargs: Any) -> Any:
    """`tqdm.tqdm`, disabled unless `output.progress` is set."""
    return tqdm_lib.tqdm(*args, disable=_disable(), **kwargs)


def trange(*args: Any, **kwargs: Any) -> Any:
    """`tqdm.trange`, disabled unless `output.progress` is set."""
    return tqdm_lib.trange(*args, disable=_disable(), **kwargs)
