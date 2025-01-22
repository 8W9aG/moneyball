"""The constant feature reducer."""

import pandas as pd
from feature_engine.selection import DropConstantFeatures

from .reducer import Reducer


class ConstantReducer(Reducer):
    """The constant reducer class."""

    # pylint: disable=too-few-public-methods

    def __init__(self) -> None:
        super().__init__()
        self._dcf = DropConstantFeatures(missing_values="ignore")

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the dataframe and remove the necessary features."""
        return self._dcf.fit_transform(df)
