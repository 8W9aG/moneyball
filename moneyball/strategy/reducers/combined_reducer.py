"""A feature reducer combining many sub feature reducers."""

import pandas as pd

from .constant_reducer import ConstantReducer
from .correlation_reducer import CorrelationReducer
from .duplicate_reducer import DuplicateReducer
from .reducer import Reducer


class CombinedReducer(Reducer):
    """Combined reducer extractor class."""

    # pylint: disable=too-few-public-methods

    def __init__(
        self, keep_features: list[str], reducers: list[Reducer] | None = None
    ) -> None:
        super().__init__()
        if reducers is None:
            reducers = [
                ConstantReducer(),
                DuplicateReducer(),
                CorrelationReducer(),
            ]
        self._reducers = reducers
        self._keep_features = keep_features

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        keep_df = df[self._keep_features].copy()
        for reducer in self._reducers:
            df = reducer.process(df)
        for column in keep_df.columns.values:
            if column not in df.columns.values:
                df[column] = keep_df[column]
        return df
