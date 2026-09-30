import logging
from typing import TypeVar

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_datetime64_any_dtype
from skrub import DatetimeEncoder, TableVectorizer

from relicl.config import RelICLConfig, WithConfig
from relicl.model.text import embed_column, is_prose_column
from relicl.timing import timed
from relicl.typing import TimingEvent, VectorizerMethod

logger = logging.getLogger(__name__)

T = TypeVar("T")


# Array column utilities. ##############################################################


def _is_ndarray_column(series: pd.Series) -> bool:
    # Drop nulls for inspection.
    sample = series.dropna()
    if sample.empty:
        return False

    first = sample.iloc[0]

    if isinstance(first, np.ndarray):
        return True
    elif isinstance(first, list):
        raise NotImplementedError()
    elif isinstance(first, tuple):
        raise NotImplementedError()

    return False


def _to_fixed_width(x, width: int) -> list:
    """Truncate or pad with NaN to exactly `width` elements."""
    if x is None:
        return [np.nan] * width
    arr = list(x)  # works for both list and ndarray
    if len(arr) >= width:
        return arr[:width]  # truncate
    return arr + [np.nan] * (width - len(arr))  # pad


def _encode_ndarray_series(series: pd.Series, width: int) -> pd.DataFrame:
    col = series.name
    col_names = [f"{col}__{i}" for i in range(width)]

    return pd.DataFrame(
        series.map(lambda x: _to_fixed_width(x, width)).tolist(),
        columns=col_names,
        index=series.index,
    )


# Classes. #############################################################################


class Vectorizer(WithConfig):
    """Vectorizes a table before processing by a tabular model"""

    def __init__(self) -> None:
        super().__init__()
        self._n_jobs = self.config.vectorizer.n_jobs

    def fit(self, df: T) -> "Vectorizer":
        raise NotImplementedError()

    def transform(self, df: T) -> T:
        raise NotImplementedError()

    def fit_transform(self, df: T) -> T:
        return self.fit(df).transform(df)

    @staticmethod
    def new_instance() -> "Vectorizer":
        config = RelICLConfig.instance()
        match config.vectorizer.type:
            case VectorizerMethod.default:
                return DefaultVectorizer()
            case VectorizerMethod.skrub:
                return SkrubVectorizer()
            case VectorizerMethod.none:
                return IdentityVectorizer()


class IdentityVectorizer(Vectorizer):
    def fit(self, df) -> "IdentityVectorizer":
        return self

    def transform(self, df: T) -> T:
        return df


class SkrubVectorizer(Vectorizer):
    def __init__(self) -> None:
        super().__init__()
        self._table_vectorizer = TableVectorizer(n_jobs=self._n_jobs)

    def fit(self, df) -> "SkrubVectorizer":
        self._table_vectorizer.fit(df)
        return self

    def transform(self, df: T) -> T:
        return self._table_vectorizer.transform(df)

    def fit_transform(self, df: T) -> T:
        return self._table_vectorizer.fit_transform(df)


class DefaultVectorizer(Vectorizer):
    @timed(event_type=TimingEvent.preprocessing)
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Container for columns that are expanded into multiple.
        expanded_cols: dict[str, pd.DataFrame] = {}

        # Pandas nullable columns. #####################################################

        # Convert pandas-nullable integer dtypes to float so missing values become
        # np.nan and the column stays uniformly numeric (#39). Likewise, convert pd.NA
        # in strings to np.nan.
        for col in df.columns:
            s = df[col]

            if pd.api.types.is_integer_dtype(
                s.dtype
            ) and pd.api.types.is_extension_array_dtype(s.dtype):
                df[col] = s.astype("float32")

            # dask will use pd.NA for strings with null values after joins, whereas
            # pandas will use np.nan. Convert it to np.nan as expected by downstream
            # models. Tested with `isinstance` rather than `== "string"`, which is False
            # for the `str` variant on pandas 3.0.3 and True on 3.0.5, and which pandas
            # itself marks as undecided (`pandas/core/arrays/string_.py:239`).
            if isinstance(s.dtype, pd.StringDtype):
                df[col] = s.astype(object).where(~df[col].isna(), np.nan)

        # Time columns. ################################################################

        # Identify time cols.
        time_col_names = [
            col1 for col1 in df.columns if is_datetime64_any_dtype(df[col1])
        ]

        # `DatetimeEncoder` emits calendar components plus, by default, raw epoch
        # seconds.
        relative_time = self.config.vectorizer.relative_time
        enc = DatetimeEncoder(add_total_seconds=not relative_time)

        # The newest timestamp in this frame is the prediction time: the task table is
        # vectorized once per evaluation cutoff with train and test rows together, train
        # rows lie at or before that cutoff, and every test row lies exactly on it.
        reference = (
            pd.Series([df[col].max() for col in time_col_names]).max()
            if relative_time and time_col_names
            else None
        )

        # Encode time columns.
        for col in time_col_names:
            encoded = enc.fit_transform(
                df[col]
            )  # Expects a Series, returns a DataFrame.

            # Age in days, positive into the past.
            if reference is not None:
                age = (reference - df[col]).dt.total_seconds() / 86_400.0
                encoded[f"{col}_rel_days"] = age.astype("float32").to_numpy()

            expanded_cols[col] = encoded

        # Array columns. ###############################################################

        # Identify array columns.
        ndarray_col_names = [col for col in df.columns if _is_ndarray_column(df[col])]

        # Encode array columns.
        for col in ndarray_col_names:
            encoded = _encode_ndarray_series(
                df[col], width=self.config.vectorizer.array_col_width
            )
            expanded_cols[col] = encoded

        # Text columns. ################################################################

        if self.config.vectorizer.text.enable:
            # Free text is otherwise ordinal-encoded by the backbone, which turns every
            # distinct string into an arbitrary unordered integer.
            text_col_names = [
                col
                for col in df.columns
                if col not in expanded_cols and is_prose_column(df[col])
            ]

            # Encode text columns.
            for col in text_col_names:
                expanded_cols[col] = embed_column(df[col])

        # Other columns. ###############################################################

        # Pass other columns through unchanged.
        other_cols = [col for col in df.columns if col not in expanded_cols]
        other_cols_df = df[other_cols]

        # Combine the result into single DF.
        result: pd.DataFrame = pd.concat(
            [other_cols_df] + list(expanded_cols.values()), axis=1
        )

        # Drop NaN / constant columns. #################################################

        # Drop all-NaN and constant columns.
        all_nan = [c for c in result.columns if result[c].isna().all()]
        constant = [c for c in result.columns if result[c].nunique(dropna=True) <= 1]
        logger.debug(f"Dropping constant cols: {constant}")
        logger.debug(f"Dropping all-NaN cols: {all_nan}")
        result = result.drop(columns=list(set(all_nan) | set(constant)))

        return result
