"""The simple moving average feature extractor."""

import statistics

import pandas as pd
from sportsball.data.league_model import DELIMITER  # type: ignore

from .columns import (find_team_count, team_column_prefix,
                      team_identifier_column, team_points_column)
from .feature import Feature


def _process_team_games(df: pd.DataFrame, sma: int) -> pd.DataFrame:
    sma_last_col = f"sma_{sma}"
    team_count = find_team_count(df)
    for i in range(team_count):
        sma_col = DELIMITER.join([team_column_prefix(i), sma_last_col])
        df[sma_col] = None

    smas: dict[str, list[float]] = {}

    def record_sma(row: pd.Series) -> pd.Series:
        nonlocal smas
        nonlocal sma
        teams_points = []
        for i in range(team_count):
            points = float(row[team_points_column(i)])
            teams_points.append(points)
        teams_points = [float(x == max(teams_points)) for x in teams_points]
        for i in range(team_count):
            team_idx = row[team_identifier_column(i)]
            sma_points = smas.get(team_idx, [])
            sma_avg = 0.0
            if sma_points:
                sma_avg = statistics.fmean(sma_points)
            sma_col = DELIMITER.join([team_column_prefix(i), sma_last_col])
            row[sma_col] = sma_avg
            if len(sma_points) >= sma:
                sma_points = sma_points[1:]
            smas[team_idx] = sma_points + [teams_points[i]]
        return row

    return df.apply(record_sma, axis=1)


class SMAFeature(Feature):
    """The simple moving average feature extractor class."""

    # pylint: disable=too-few-public-methods

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the dataframe and add the necessary features."""
        df = _process_team_games(df, 5)
        return df.copy()
