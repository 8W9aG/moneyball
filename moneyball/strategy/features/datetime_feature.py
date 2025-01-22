"""The datetime feature extractor."""

import pandas as pd
from feature_engine.datetime import DatetimeFeatures

from .feature import Feature


class DatetimeFeature(Feature):
    """The datatime feature extractor class."""

    # pylint: disable=too-few-public-methods

    def __init__(self) -> None:
        super().__init__()
        self._dtf = DatetimeFeatures(
            features_to_extract="all", missing_values="ignore", drop_original=False
        )

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the dataframe and add the necessary features."""
        return self._dtf.fit_transform(df)
