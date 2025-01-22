"""The duplicate feature reducer."""

import pandas as pd
from feature_engine.selection import DropDuplicateFeatures

from .reducer import Reducer


class DuplicateReducer(Reducer):
    """The duplicate reducer class."""

    # pylint: disable=too-few-public-methods

    def __init__(self) -> None:
        super().__init__()
        self._ddf = DropDuplicateFeatures(missing_values="ignore")

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the dataframe and remove the necessary features."""
        return self._ddf.fit_transform(df)
