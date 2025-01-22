"""The total feature extractor."""

# pylint: disable=comparison-with-itself
import pandas as pd
import tqdm
from sportsball.data.game_model import VENUE_COLUMN_PREFIX  # type: ignore
from sportsball.data.league_model import DELIMITER  # type: ignore

from .columns import (find_player_count, find_team_count, player_column_prefix,
                      player_identifier_column, team_column_prefix,
                      team_identifier_column, venue_identifier_column)
from .feature import Feature

TOTAL_COLUMN_PREFIX = "total"
TOTAL_GAMES_COLUMN = "games"
TOTAL_KICKS_COLUMN = "kicks"


def _process_player_team_games(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    team_count = find_team_count(df)
    player_count = find_player_count(df, team_count)
    player_team_games: dict[str, int] = {}
    for i in range(team_count):
        for j in range(player_count):
            total_attendance_col = DELIMITER.join(
                [TOTAL_COLUMN_PREFIX, player_column_prefix(i, j), TOTAL_GAMES_COLUMN]
            )
            df[total_attendance_col] = None
    for row in tqdm.tqdm(df.itertuples(), desc="Processing player team games total"):
        for i in range(team_count):
            team_idx = row[cols.index(team_identifier_column(i)) + 1]
            # Check for NaNs
            if team_idx != team_idx:
                continue
            for j in range(player_count):
                player_col = player_identifier_column(i, j)
                if player_col not in cols:
                    continue
                player_idx = row[cols.index(player_col) + 1]
                # Check for NaNs
                if player_idx != player_idx:
                    continue
                key = "-".join([team_idx, player_idx])
                count = player_team_games.get(key, 0)
                total_attendance_col = DELIMITER.join(
                    [
                        TOTAL_COLUMN_PREFIX,
                        player_column_prefix(i, j),
                        TOTAL_GAMES_COLUMN,
                    ]
                )
                df.loc[row.Index, total_attendance_col] = count
                player_team_games[key] = count + 1
    return df


def _process_team_venue_games(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    venue_col = venue_identifier_column()
    if venue_col not in cols:
        return df
    team_count = find_team_count(df)
    for i in range(team_count):
        total_venue_col = DELIMITER.join(
            [
                TOTAL_COLUMN_PREFIX,
                team_column_prefix(i),
                VENUE_COLUMN_PREFIX,
                TOTAL_GAMES_COLUMN,
            ]
        )
        df[total_venue_col] = None
    venue_team_games: dict[str, int] = {}
    for row in tqdm.tqdm(df.itertuples(), desc="Processing team venue games total"):
        for i in range(team_count):
            team_idx = row[cols.index(team_identifier_column(i)) + 1]
            # Check for NaNs
            if team_idx != team_idx:
                continue
            venue_idx = row[cols.index(venue_identifier_column()) + 1]
            # Check for NaNs
            if venue_idx != venue_idx:
                continue
            key = "-".join([team_idx, venue_idx])
            count = venue_team_games.get(key, 0)
            total_venue_col = DELIMITER.join(
                [
                    TOTAL_COLUMN_PREFIX,
                    team_column_prefix(i),
                    VENUE_COLUMN_PREFIX,
                    TOTAL_GAMES_COLUMN,
                ]
            )
            df.loc[row.Index, total_venue_col] = count
            venue_team_games[key] = count + 1
    return df


class TotalFeature(Feature):
    """The total feature extractor class."""

    # pylint: disable=too-few-public-methods

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the dataframe and add the necessary features."""
        cols = df.columns.values.tolist()
        df = _process_player_team_games(df, cols)
        df = _process_team_venue_games(df, cols)
        return df.copy()
