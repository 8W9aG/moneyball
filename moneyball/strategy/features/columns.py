"""Helper functions for columns."""

import pandas as pd
from sportsball.data.coach_model import COACH_IDENTIFIER_COLUMN
from sportsball.data.game_model import GAME_ATTENDANCE_COLUMN  # type: ignore
from sportsball.data.game_model import (GAME_WEEK_COLUMN, TEAM_COLUMN_PREFIX,
                                        VENUE_COLUMN_PREFIX)
from sportsball.data.league_model import DELIMITER  # type: ignore
from sportsball.data.news_model import NEWS_SUMMARY_COLUMN
from sportsball.data.odds_model import ODDS_ODDS_COLUMN
from sportsball.data.player_model import PLAYER_KICKS_COLUMN  # type: ignore
from sportsball.data.player_model import (PLAYER_FUMBLES_LOST_COLUMN,
                                          PLAYER_IDENTIFIER_COLUMN)
from sportsball.data.team_model import PLAYER_COLUMN_PREFIX  # type: ignore
from sportsball.data.team_model import (NAME_COLUMN, TEAM_COACHES_COLUMN,
                                        TEAM_IDENTIFIER_COLUMN,
                                        TEAM_NEWS_COLUMN, TEAM_ODDS_COLUMN,
                                        TEAM_POINTS_COLUMN)
from sportsball.data.venue_model import VENUE_IDENTIFIER_COLUMN  # type: ignore


def team_column_prefix(team_idx: int) -> str:
    """Generate a prefix for a team column at a given index."""
    return DELIMITER.join(
        [
            TEAM_COLUMN_PREFIX,
            str(team_idx),
        ]
    )


def team_identifier_column(team_idx: int) -> str:
    """Generate a team identifier column at a given index."""
    return DELIMITER.join([team_column_prefix(team_idx), TEAM_IDENTIFIER_COLUMN])


def team_points_column(team_idx: int) -> str:
    """Generate a team points column at a given index."""
    return DELIMITER.join([team_column_prefix(team_idx), TEAM_POINTS_COLUMN])


def team_name_column(team_idx: int) -> str:
    """Generate a team name column at a given index."""
    return DELIMITER.join([team_column_prefix(team_idx), NAME_COLUMN])


def player_column_prefix(team_idx: int, player_idx: int | None) -> str:
    """Generate a prefix for a player column at a given index."""
    if player_idx is None:
        return DELIMITER.join(
            [
                team_column_prefix(team_idx),
                PLAYER_COLUMN_PREFIX,
            ]
        )
    return DELIMITER.join(
        [
            team_column_prefix(team_idx),
            PLAYER_COLUMN_PREFIX,
            str(player_idx),
        ]
    )


def player_identifier_column(team_idx: int, player_idx: int) -> str:
    """Generate a player identifier column at a given index."""
    return DELIMITER.join(
        [player_column_prefix(team_idx, player_idx), PLAYER_IDENTIFIER_COLUMN]
    )


def coach_column_prefix(team_idx: int, coach_idx: int) -> str:
    """Generate the coach column prefix."""
    return DELIMITER.join(
        [
            team_column_prefix(team_idx),
            TEAM_COACHES_COLUMN,
            str(coach_idx),
        ]
    )


def coach_identifier_column(team_idx: int, coach_idx: int) -> str:
    """Generate a coach identifier column."""
    return DELIMITER.join(
        [coach_column_prefix(team_idx, coach_idx), COACH_IDENTIFIER_COLUMN]
    )


def attendance_column() -> str:
    """Generate an attendance column."""
    return DELIMITER.join([GAME_ATTENDANCE_COLUMN])


def find_team_count(df: pd.DataFrame) -> int:
    """Find the number of teams in the dataframe."""
    team_count = 0
    while True:
        if team_identifier_column(team_count) not in df.columns.values:
            break
        team_count += 1
    return team_count


def find_player_count(df: pd.DataFrame, team_count: int) -> int:
    """Find the number of players in a team in the dataframe."""
    player_count = 0
    while True:
        found_player = False
        for i in range(team_count):
            if player_identifier_column(i, player_count) not in df.columns.values:
                continue
            found_player = True
        if not found_player:
            break
        player_count += 1
    return player_count


def find_coach_count(df: pd.DataFrame, team_count: int) -> int:
    """Find the number of coaches in a team in the dataframe."""
    coach_count = 0
    while True:
        found_coach = False
        for i in range(team_count):
            if coach_identifier_column(i, coach_count) not in df.columns.values:
                continue
            found_coach = True
        if not found_coach:
            break
        coach_count += 1
    return coach_count


def venue_identifier_column() -> str:
    """Generate a venue identifier column."""
    return DELIMITER.join([VENUE_COLUMN_PREFIX, VENUE_IDENTIFIER_COLUMN])


def kick_column(team_idx: int, player_idx: int) -> str:
    """Generate a kick column."""
    return DELIMITER.join(
        [player_column_prefix(team_idx, player_idx), PLAYER_KICKS_COLUMN]
    )


def fumbles_lost_column(team_idx: int, player_idx: int) -> str:
    """Generate a fumbles lost column."""
    return DELIMITER.join(
        [player_column_prefix(team_idx, player_idx), PLAYER_FUMBLES_LOST_COLUMN]
    )


def week_column() -> str:
    """Generate a week column."""
    return DELIMITER.join([GAME_WEEK_COLUMN])


def odds_column_prefix(team_idx: int, odds_idx: int) -> str:
    """Generates an odds column_prefix."""
    return DELIMITER.join(
        [team_column_prefix(team_idx), TEAM_ODDS_COLUMN, str(odds_idx)]
    )


def odds_odds_column(team_idx: int, odds_idx: int) -> str:
    """Generates an odds odds column."""
    return DELIMITER.join([odds_column_prefix(team_idx, odds_idx), ODDS_ODDS_COLUMN])


def find_odds_count(df: pd.DataFrame, team_count: int) -> int:
    """Find the number of odds in a team in the dataframe."""
    odds_count = 0
    while True:
        found_odds = False
        for i in range(team_count):
            if odds_odds_column(i, odds_count) not in df.columns.values:
                continue
            found_odds = True
        if not found_odds:
            break
        odds_count += 1
    return odds_count


def news_column_prefix(team_idx: int, news_idx: int) -> str:
    """Generates an news column_prefix."""
    return DELIMITER.join(
        [team_column_prefix(team_idx), TEAM_NEWS_COLUMN, str(news_idx)]
    )


def news_summary_column(team_idx: int, news_idx: int) -> str:
    """Generates an news summary column."""
    return DELIMITER.join([news_column_prefix(team_idx, news_idx), NEWS_SUMMARY_COLUMN])


def find_news_count(df: pd.DataFrame, team_count: int) -> int:
    """Find the number of news articles in a team in the dataframe."""
    news_count = 0
    while True:
        found_news = False
        for i in range(team_count):
            if news_summary_column(i, news_count) not in df.columns.values:
                continue
            found_news = True
        if not found_news:
            break
        news_count += 1
    return news_count
