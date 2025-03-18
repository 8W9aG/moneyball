"""The datetime feature extractor."""

import datetime
import random

import pandas as pd
from feature_engine.datetime import DatetimeFeatures
from pandas.api.types import is_datetime64_any_dtype

from .feature import Feature


# https://gist.github.com/rg3915/db907d7455a4949dbe69
def gen_datetime(
    min_year=1900, max_year=datetime.datetime.now().year
) -> datetime.datetime:
    """Generate a datetime in format yyyy-mm-dd hh:mm:ss.000000."""
    start = datetime.datetime(min_year, 1, 1, 00, 00, 00)
    years = max_year - min_year + 1
    end = start + datetime.timedelta(days=365 * years)
    return start + (end - start) * random.random()


class DatetimeFeature(Feature):
    """The datetime feature extractor class."""

    # pylint: disable=too-few-public-methods

    def __init__(self) -> None:
        super().__init__()
        self._dtf = DatetimeFeatures(
            features_to_extract="all", missing_values="ignore", drop_original=False, utc=True
        )

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the dataframe and add the necessary features."""
        dt_columns = [x for x in df.columns if is_datetime64_any_dtype(x)]

        replace_dt = None
        while replace_dt is None:
            random_dt = gen_datetime()
            found = False
            for column in dt_columns:
                unique_dts = df[column].unique()
                if random_dt in unique_dts:
                    found = True
                    break
            if not found:
                replace_dt = random_dt
        df = df.replace({pd.NaT: replace_dt})
        for dt_column in dt_columns:
            df[dt_column] = pd.to_datetime(dt_column, utc=True, errors='coerce')
        df = self._dtf.fit_transform(df)

        def fix_dummy_date(row: pd.Series) -> pd.Series:
            for dt_column in dt_columns:
                if row[dt_column] != replace_dt:
                    continue
                row[dt_column] = pd.NaT
                for column in df.columns.values:
                    if column.startswith(str(dt_column)):
                        row[column] = None
            return row

        return df.apply(fix_dummy_date, axis=1)
