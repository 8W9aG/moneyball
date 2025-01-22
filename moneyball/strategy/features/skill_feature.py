"""The skill feature extractor."""

import datetime
from warnings import simplefilter

import pandas as pd
import tqdm
from dateutil.relativedelta import relativedelta
from openskill.models import PlackettLuce, PlackettLuceRating
from pandarallel import pandarallel  # type: ignore
from sportsball.data.field_type import FieldType  # type: ignore
from sportsball.data.game_model import GAME_DT_COLUMN  # type: ignore
from sportsball.data.league_model import DELIMITER  # type: ignore

from .columns import (find_player_count, find_team_count, player_column_prefix,
                      player_identifier_column, team_column_prefix,
                      team_identifier_column, team_points_column)
from .feature import Feature

SKILL_COLUMN_PREFIX = "skill"
SKILL_MU_COLUMN = "mu"
SKILL_SIGMA_COLUMN = "sigma"
SKILL_RANKING_COLUMN = "ranking"
SKILL_PROBABILITY_COLUMN = "probability"
YEAR_SLICE_ALL = "all"


def _slice_df(
    df: pd.DataFrame, date: datetime.date, year_slice: int | None
) -> pd.DataFrame:
    df_slice = df[df[GAME_DT_COLUMN].dt.date < date]
    if year_slice is not None:
        start_date = date - relativedelta(years=year_slice)
        df_slice = df_slice[df_slice[GAME_DT_COLUMN].dt.date > start_date]
    return df_slice


def _create_teams(
    df_slice: pd.DataFrame, team_count: int, group: pd.DataFrame
) -> tuple[PlackettLuce, dict[str, PlackettLuceRating]]:
    team_model = PlackettLuce()
    teams = {}
    for i in range(team_count):
        for team_identifier in df_slice[team_identifier_column(i)].unique():
            if not isinstance(team_identifier, str):
                continue
            if team_identifier in teams:
                continue
            teams[team_identifier] = team_model.rating(name=team_identifier)
        for team_identifier in group[team_identifier_column(i)].unique():
            if not isinstance(team_identifier, str):
                continue
            if team_identifier in teams:
                continue
            teams[team_identifier] = team_model.rating(name=team_identifier)
    return team_model, teams


def _create_player_teams(
    df_slice: pd.DataFrame, team_count: int, player_count: int, group: pd.DataFrame
) -> tuple[PlackettLuce, dict[str, PlackettLuceRating]]:
    player_model = PlackettLuce()
    players = {}
    for i in range(player_count):
        for j in range(team_count):
            if player_identifier_column(j, i) in df_slice.columns.values:
                for player_identifier in df_slice[
                    player_identifier_column(j, i)
                ].unique():
                    if not isinstance(player_identifier, str):
                        continue
                    if player_identifier in players:
                        continue
                    players[player_identifier] = player_model.rating(
                        name=player_identifier
                    )
            if player_identifier_column(j, i) in group.columns.values:
                for player_identifier in group[player_identifier_column(j, i)].unique():
                    if not isinstance(player_identifier, str):
                        continue
                    if player_identifier in players:
                        continue
                    players[player_identifier] = player_model.rating(
                        name=player_identifier
                    )
    return player_model, players


def _find_matches(
    row: pd.Series,
    team_count: int,
    player_count: int,
    teams: dict[str, PlackettLuceRating],
    players: dict[str, PlackettLuceRating],
) -> tuple[list[float], list[list[PlackettLuceRating]], list[list[PlackettLuceRating]]]:
    points = []
    team_match = []
    player_match = []
    for i in range(team_count):
        points_col = team_points_column(i)
        if pd.isnull(row[points_col]):
            continue
        points.append(float(row[points_col]))
        team_idx = row[team_identifier_column(i)]
        team_match.append([teams[team_idx]])
        player_team = []
        for j in range(player_count):
            try:
                if pd.isnull(row[player_identifier_column(i, j)]):
                    continue
            except KeyError:
                continue
            player_team.append(players[row[player_identifier_column(i, j)]])
        player_match.append(player_team)
    return points, team_match, player_match


def _rate_match(
    model: PlackettLuce,
    match: list[list[PlackettLuceRating]],
    points: list[float],
    ratings: dict[str, PlackettLuceRating],
) -> dict[str, PlackettLuceRating]:
    if not match:
        return ratings
    for team in match:
        if not team:
            return ratings
    output = model.rate(match, scores=points)
    for team in output:
        for player in team:
            name = player.name
            if name is None:
                continue
            ratings[name] = player
    return ratings


def _simulate_games(
    df_slice: pd.DataFrame,
    team_count: int,
    player_count: int,
    model_team: tuple[PlackettLuce, dict[str, PlackettLuceRating]],
    model_players: tuple[PlackettLuce, dict[str, PlackettLuceRating]],
) -> tuple[
    tuple[PlackettLuce, dict[str, PlackettLuceRating]],
    tuple[PlackettLuce, dict[str, PlackettLuceRating]],
]:
    team_model, teams = model_team
    player_model, players = model_players

    def _simulate_match(row: pd.Series) -> pd.Series:
        nonlocal team_count
        nonlocal player_count
        nonlocal teams
        nonlocal players
        nonlocal team_model
        nonlocal player_model
        points, team_match, player_match = _find_matches(
            row, team_count, player_count, teams, players
        )
        teams = _rate_match(team_model, team_match, points, teams)
        players = _rate_match(player_model, player_match, points, players)
        return row

    df_slice.apply(_simulate_match, axis=1)

    return (team_model, teams), (player_model, players)


def _find_player_team(
    player_count: int,
    players: dict[str, PlackettLuceRating],
    row: pd.Series,
    team_index: int,
) -> list[PlackettLuceRating]:
    player_team = []
    for j in range(player_count):
        try:
            if pd.isnull(row[player_identifier_column(team_index, j)]):
                continue
        except KeyError:
            continue
        player_idx = row[player_identifier_column(team_index, j)]
        player = players[player_idx]
        player_team.append(player)
    return player_team


def _find_team_team(
    team_index: int, row: pd.Series, year_col: str, teams: dict[str, PlackettLuceRating]
) -> tuple[list[PlackettLuceRating], pd.Series]:
    team_col_prefix = team_column_prefix(team_index)
    team_idx_col = team_identifier_column(team_index)
    if pd.isnull(row[team_idx_col]):
        return [], row
    team_idx = row[team_idx_col]
    team_skill_col_prefix = DELIMITER.join(
        [team_col_prefix, SKILL_COLUMN_PREFIX, year_col]
    )
    team_mu_col = DELIMITER.join([team_skill_col_prefix, SKILL_MU_COLUMN])
    team_sigma_col = DELIMITER.join([team_skill_col_prefix, SKILL_SIGMA_COLUMN])
    team = teams[team_idx]
    row[team_mu_col] = team.mu
    row[team_sigma_col] = team.sigma
    return [team], row


def _find_row_matches(
    row: pd.Series,
    counts: tuple[int, int],
    year_col: str,
    teams: dict[str, PlackettLuceRating],
    players: dict[str, PlackettLuceRating],
) -> tuple[pd.Series, list[list[PlackettLuceRating]], list[list[PlackettLuceRating]]]:
    team_count, player_count = counts
    team_match = []
    player_match = []
    for i in range(team_count):
        team, row = _find_team_team(i, row, year_col, teams)
        if not team:
            continue
        team_match.append(team)
        player_team = _find_player_team(player_count, players, row, i)
        player_match.append(player_team)
    return row, team_match, player_match


def _rank_team_predictions(
    team_match: list[list[PlackettLuceRating]],
    row: pd.Series,
    team_model: PlackettLuce,
    year_col: str,
) -> pd.Series:
    rank_team_predictions = team_model.predict_rank(team_match)
    for i, (rank, prob) in enumerate(rank_team_predictions):
        team_skill_col_prefix = DELIMITER.join(
            [
                team_column_prefix(i),
                SKILL_COLUMN_PREFIX,
                year_col,
            ]
        )
        team_ranking_col = DELIMITER.join([team_skill_col_prefix, SKILL_RANKING_COLUMN])
        team_prob_col = DELIMITER.join(
            [team_skill_col_prefix, SKILL_PROBABILITY_COLUMN]
        )
        row[team_ranking_col] = rank
        row[team_prob_col] = prob
    return row


def _rank_player_predictions(
    row: pd.Series,
    player_model: PlackettLuce,
    player_match: list[list[PlackettLuceRating]],
    year_col: str,
) -> pd.Series:
    for match in player_match:
        if not match:
            return row
    rank_player_predictions = player_model.predict_rank(player_match)
    for i, (rank, prob) in enumerate(rank_player_predictions):
        player_skill_col_prefix = DELIMITER.join(
            [
                player_column_prefix(i, None),
                SKILL_COLUMN_PREFIX,
                year_col,
            ]
        )
        player_ranking_col = DELIMITER.join(
            [player_skill_col_prefix, SKILL_RANKING_COLUMN]
        )
        player_prob_col = DELIMITER.join(
            [player_skill_col_prefix, SKILL_PROBABILITY_COLUMN]
        )
        row[player_ranking_col] = rank
        row[player_prob_col] = prob
    return row


def _create_feature_cols(
    group: pd.DataFrame,
    year_slice: int | None,
    counts: tuple[int, int],
    model_team: tuple[PlackettLuce, dict[str, PlackettLuceRating]],
    model_players: tuple[PlackettLuce, dict[str, PlackettLuceRating]],
) -> pd.DataFrame:
    year_col = str(year_slice)
    team_model, teams = model_team
    player_model, players = model_players
    team_count, player_count = counts

    def _apply_group_skills_features(row: pd.Series) -> pd.Series:
        row, team_match, player_match = _find_row_matches(
            row, (team_count, player_count), year_col, teams, players
        )
        row = _rank_team_predictions(team_match, row, team_model, year_col)
        row = _rank_player_predictions(row, player_model, player_match, year_col)
        return row

    return group.apply(_apply_group_skills_features, axis=1)


def _create_all_features(
    df: pd.DataFrame, team_count: int, player_count: int
) -> pd.DataFrame:
    team_model, teams = _create_teams(df, team_count, df)
    player_model, players = _create_player_teams(df, team_count, player_count, df)

    def _apply_skills_features(row: pd.Series) -> pd.Series:
        nonlocal team_model
        nonlocal teams
        nonlocal player_model
        nonlocal players
        row, team_match, player_match = _find_row_matches(
            row, (team_count, player_count), YEAR_SLICE_ALL, teams, players
        )
        (team_model, teams), (player_model, players) = _simulate_games(
            row.to_frame().T,
            team_count,
            player_count,
            (team_model, teams),
            (player_model, players),
        )
        row = _rank_team_predictions(team_match, row, team_model, YEAR_SLICE_ALL)
        row = _rank_player_predictions(row, player_model, player_match, YEAR_SLICE_ALL)
        return row

    df = df.apply(_apply_skills_features, axis=1)
    return df


class SkillFeature(Feature):
    """The skill feature extractor class."""

    # pylint: disable=too-few-public-methods

    def __init__(self, year_slices: list[int | None]) -> None:
        super().__init__()
        self._year_slices = year_slices
        tqdm.tqdm.pandas()
        simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
        pandarallel.initialize(progress_bar=True)

    def _create_columns(self, df: pd.DataFrame, team_count: int) -> pd.DataFrame:
        golden_features = df.attrs.get(str(FieldType.GOLDEN), [])
        for year_slice in self._year_slices:
            year_col = str(year_slice) if year_slice is not None else YEAR_SLICE_ALL
            for i in range(team_count):
                for team_col_suffix in [
                    SKILL_MU_COLUMN,
                    SKILL_SIGMA_COLUMN,
                    SKILL_RANKING_COLUMN,
                    SKILL_PROBABILITY_COLUMN,
                ]:
                    team_col = DELIMITER.join(
                        [
                            team_column_prefix(i),
                            SKILL_COLUMN_PREFIX,
                            year_col,
                            team_col_suffix,
                        ]
                    )
                    df[team_col] = 0.0
                    golden_features.append(team_col)
                for empty_player_col_suffix in [
                    SKILL_RANKING_COLUMN,
                    SKILL_PROBABILITY_COLUMN,
                ]:
                    player_col = DELIMITER.join(
                        [
                            player_column_prefix(i, None),
                            SKILL_COLUMN_PREFIX,
                            year_col,
                            empty_player_col_suffix,
                        ]
                    )
                    df[player_col] = 0.0
                    golden_features.append(player_col)
        df.attrs[str(FieldType.GOLDEN)] = golden_features
        return df.copy()

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the dataframe and add the necessary features."""
        print(f"Calculating Skills Features over the year slices: {self._year_slices}")
        team_count = find_team_count(df)
        player_count = find_player_count(df, team_count)
        df = self._create_columns(df, team_count)
        if None in self._year_slices:
            df = _create_all_features(df, team_count, player_count)

        def calculate_skills(group: pd.DataFrame) -> pd.DataFrame:
            nonlocal df
            nonlocal team_count
            nonlocal player_count
            dates = group[GAME_DT_COLUMN].dt.date.values.tolist()
            if not dates:
                return group
            date = dates[0]
            for year_slice in self._year_slices:
                if year_slice is None:
                    continue
                df_slice = _slice_df(df, date, year_slice)  # type: ignore
                if df_slice.empty:
                    continue
                team_model, teams = _create_teams(df_slice, team_count, group)
                player_model, players = _create_player_teams(
                    df_slice, team_count, player_count, group
                )
                (team_model, teams), (player_model, players) = _simulate_games(
                    df_slice,
                    team_count,
                    player_count,
                    (team_model, teams),
                    (player_model, players),
                )
                group = _create_feature_cols(
                    group,
                    year_slice,
                    (team_count, player_count),
                    (team_model, teams),
                    (player_model, players),
                )
            return group

        return (
            df.groupby(  # type: ignore
                [df[GAME_DT_COLUMN].dt.date]
            )
            .parallel_apply(calculate_skills)
            .reset_index(drop=True)
        )
