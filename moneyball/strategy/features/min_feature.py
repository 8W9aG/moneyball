"""The min feature extractor."""

import pandas as pd
from sportsball.data.game_model import SEASON_TYPE_COLUMN  # type: ignore
from sportsball.data.game_model import GAME_DT_COLUMN, SEASON_YEAR_COLUMN
from sportsball.data.league_model import DELIMITER  # type: ignore

from .feature import Feature

MIN_COLUMN_PREFIX = "min"


def _process_season_start_dt(df: pd.DataFrame) -> pd.DataFrame:
    season_start_dt = {}
    season_start_dt_col = DELIMITER.join(["season", "startdt"])
    df[season_start_dt_col] = None

    def record_season_start_dt(row: pd.Series) -> pd.Series:
        nonlocal season_start_dt
        nonlocal season_start_dt_col
        season_key = "-".join([str(row[SEASON_YEAR_COLUMN]), row[SEASON_TYPE_COLUMN]])
        if season_key not in season_start_dt:
            season_start_dt[season_key] = row[GAME_DT_COLUMN]
        row[season_start_dt_col] = season_start_dt[season_key]
        return row

    return df.apply(record_season_start_dt, axis=1)


class MinFeature(Feature):
    """The min feature extractor class."""

    # pylint: disable=too-few-public-methods

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the dataframe and add the necessary features."""
        df = _process_season_start_dt(df)
        return df.copy()
