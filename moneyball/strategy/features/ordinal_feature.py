"""The ordinal feature extractor."""

import pandas as pd
from feature_engine.encoding import OrdinalEncoder
from sportsball.data.field_type import FieldType  # type: ignore

from .feature import Feature


class OrdinalFeature(Feature):
    """The ordinal feature extractor class."""

    # pylint: disable=too-few-public-methods

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self._od = OrdinalEncoder(missing_values="ignore", encoding_method="arbitrary")

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the dataframe and add the necessary features."""
        df = self._od.fit_transform(df)
        for categorical_column in df.attrs[str(FieldType.CATEGORICAL)]:
            if categorical_column not in df.columns.values:
                continue
            df[categorical_column] = (
                df[categorical_column].fillna(0).astype(int).astype("category")
            )
        return df
