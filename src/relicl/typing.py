from enum import StrEnum, auto
from typing import Any, Callable, Literal, Protocol, TypeVar

import numpy as np
import pandas as pd
from dask import dataframe as dd

T_SHAPE = tuple[int | Any, int]


T_DF = TypeVar("T_DF", pd.DataFrame, dd.DataFrame)

T_JOIN_HOW = Literal["left", "right", "inner", "outer"]


class SaveStepFn(Protocol):
    def __call__(self, title: str = "", close: bool = False) -> None: ...

    @classmethod
    def default(cls) -> "SaveStepFn":
        def g(title: str = "", close: bool = False):
            pass

        return g


class PoolingMethod(StrEnum):
    mean = auto()
    max = auto()
    sum = auto()

    def to_numpy(self) -> Callable:
        match self:
            case PoolingMethod.sum:
                return np.nansum
            case PoolingMethod.max:
                return np.nanmax
            case PoolingMethod.mean:
                return np.nanmean


class DimReductionMethod(StrEnum):
    # Random projections.
    gaussian_random_projection = auto()
    sparse_random_projection = auto()
    sign_random_projection = auto()
    # PCA.
    pca = auto()
    pca_whiten = auto()
    # SVD.
    svd = auto()
    svd_whiten = auto()
    # Entries selection.
    random_k = auto()
    first_k = auto()


class FusionType(StrEnum):
    early = auto()
    late = auto()
    none = auto()


class SimplifyMethod(StrEnum):
    old = auto()
    default = auto()


class VectorizerMethod(StrEnum):
    default = auto()
    skrub = auto()
    none = auto()


class BackendType(StrEnum):
    pandas = auto()
    dask = auto()


class EvalMode(StrEnum):
    val = auto()
    test = auto()


class ModelType(StrEnum):
    tabicl = auto()
    tabpfn = auto()
    tabfm = auto()


class InferenceType(StrEnum):
    one_pass = auto()
    tabular_gnn = auto()


class TimingEvent(StrEnum):
    any = auto()
    tabicl = auto()
    rewrite = auto()
    fusion = auto()
    preprocessing = auto()


class DataSourceType(StrEnum):
    train = auto()
    test = auto()


class RegressionOutput(StrEnum):
    """Point estimate taken from TabICL's predicted quantiles."""

    mean = auto()
    median = auto()


class TargetTransform(StrEnum):
    """Transform applied to regression targets before fitting the backbone."""

    none = auto()
    log1p = auto()
