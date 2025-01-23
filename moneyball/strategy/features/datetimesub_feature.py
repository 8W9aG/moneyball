"""The datetime sub feature extractor."""

import pandas as pd
from feature_engine.datetime import DatetimeSubtraction
from sportsball.data.field_type import FieldType  # type: ignore

from .feature import Feature


class DatetimeSubFeature(Feature):
    """The datetime sub feature extractor class."""

    # pylint: disable=too-few-public-methods

    def __init__(self, target: str, reference: str, output_unit: str) -> None:
        super().__init__()
        self.column_name = f"{target}_sub_{reference}_{output_unit}"
        self._dtsf = DatetimeSubtraction(
            variables=target,
            reference=reference,
            output_unit=output_unit,
            new_variables_names=[self.column_name],
            missing_values="ignore",
        )

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the dataframe and add the necessary features."""
        attrs = df.attrs
        df = self._dtsf.fit_transform(df)
        df.attrs = attrs
        df.attrs[str(FieldType.LOOKAHEAD)].append(self.column_name)
        return df
