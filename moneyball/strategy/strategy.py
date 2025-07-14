"""The strategy class."""

# pylint: disable=too-many-statements,line-too-long,invalid-unary-operand-type
import datetime
import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytz
import wavetrainer as wt  # type: ignore
from sportsball.data.address_model import (ADDRESS_LATITUDE_COLUMN,
                                           ADDRESS_LONGITUDE_COLUMN)
from sportsball.data.bookie_model import BOOKIE_IDENTIFIER_COLUMN
from sportsball.data.field_type import FieldType  # type: ignore
from sportsball.data.game_model import GAME_DT_COLUMN  # type: ignore
from sportsball.data.game_model import VENUE_COLUMN_PREFIX
from sportsball.data.league_model import DELIMITER  # type: ignore
from sportsball.data.news_model import (NEWS_PUBLISHED_COLUMN,
                                        NEWS_SOURCE_COLUMN, NEWS_TITLE_COLUMN)
from sportsball.data.odds_model import (DT_COLUMN, ODDS_BET_COLUMN,
                                        ODDS_BOOKIE_COLUMN,
                                        ODDS_CANONICAL_COLUMN)
from sportsball.data.player_model import \
    ASSISTS_COLUMN as PLAYER_ASSISTS_COLUMN  # type: ignore
from sportsball.data.player_model import \
    FIELD_GOALS_ATTEMPTED_COLUMN as \
    PLAYER_FIELD_GOALS_ATTEMPTED_COLUMN  # type: ignore
from sportsball.data.player_model import \
    FIELD_GOALS_COLUMN as PLAYER_FIELD_GOALS_COLUMN  # type: ignore
from sportsball.data.player_model import \
    OFFENSIVE_REBOUNDS_COLUMN as PLAYER_OFFENSIVE_REBOUNDS_COLUMN
from sportsball.data.player_model import (
    PLAYER_ASSIST_TACKLES_COLUMN, PLAYER_AVERAGE_GAIN_COLUMN,
    PLAYER_AVERAGE_INTERCEPTION_YARDS_COLUMN,
    PLAYER_AVERAGE_KICKOFF_RETURN_YARDS_COLUMN,
    PLAYER_AVERAGE_KICKOFF_YARDS_COLUMN,
    PLAYER_AVERAGE_PUNT_RETURN_YARDS_COLUMN, PLAYER_AVERAGE_SACK_YARDS_COLUMN,
    PLAYER_AVERAGE_STUFF_YARDS_COLUMN, PLAYER_BEHINDS_COLUMN,
    PLAYER_BIRTH_DATE_COLUMN, PLAYER_BLOCKED_FIELD_GOAL_TOUCHDOWNS_COLUMN,
    PLAYER_BLOCKED_PUNT_EZ_REC_TD_COLUMN,
    PLAYER_BLOCKED_PUNT_TOUCHDOWNS_COLUMN, PLAYER_BLOCKS_COLUMN,
    PLAYER_BOUNCES_COLUMN, PLAYER_BROWNLOW_VOTES_COLUMN,
    PLAYER_CLANGERS_COLUMN, PLAYER_CLEARANCES_COLUMN,
    PLAYER_COMPLETION_PERCENTAGE_COLUMN, PLAYER_COMPLETIONS_COLUMN,
    PLAYER_CONTESTED_MARKS_COLUMN, PLAYER_CONTESTED_POSSESSIONS_COLUMN,
    PLAYER_CORSI_FOR_PERCENTAGE_COLUMN, PLAYER_DECISION_COLUMN,
    PLAYER_DEFENSIVE_FUMBLE_RETURN_YARDS_COLUMN,
    PLAYER_DEFENSIVE_FUMBLE_RETURNS_COLUMN,
    PLAYER_DEFENSIVE_FUMBLES_TOUCHDOWNS_COLUMN, PLAYER_DEFENSIVE_POINTS_COLUMN,
    PLAYER_DEFENSIVE_REBOUNDS_COLUMN, PLAYER_DEFENSIVE_TOUCHDOWNS_COLUMN,
    PLAYER_DEFENSIVE_ZONE_STARTS_COLUMN, PLAYER_DISPOSALS_COLUMN,
    PLAYER_ESPN_QUARTERBACK_RATING_COLUMN,
    PLAYER_ESPN_RUNNINGBACK_RATING_COLUMN, PLAYER_ESPN_WIDERECEIVER_COLUMN,
    PLAYER_EVEN_STRENGTH_ASSISTS_COLUMN, PLAYER_EVEN_STRENGTH_GOALS_COLUMN,
    PLAYER_EXTRA_POINT_ATTEMPTS_COLUMN, PLAYER_EXTRA_POINT_BLOCKED_COLUMN,
    PLAYER_EXTRA_POINT_PERCENTAGE_COLUMN,
    PLAYER_EXTRA_POINTS_BLOCKED_PERCENTAGE_COLUMN,
    PLAYER_EXTRA_POINTS_MADE_COLUMN, PLAYER_FAIR_CATCH_PERCENTAGE_COLUMN,
    PLAYER_FAIR_CATCHES_COLUMN, PLAYER_FIELD_GOAL_ATTEMPT_YARDS_COLUMN,
    PLAYER_FIELD_GOAL_ATTEMPTS_ABOVE_50_YARDS_COLUMN,
    PLAYER_FIELD_GOAL_ATTEMPTS_MAX_19_YARDS_COLUMN,
    PLAYER_FIELD_GOAL_ATTEMPTS_MAX_29_YARDS_COLUMN,
    PLAYER_FIELD_GOAL_ATTEMPTS_MAX_39_YARDS_COLUMN,
    PLAYER_FIELD_GOAL_ATTEMPTS_MAX_49_YARDS_COLUMN,
    PLAYER_FIELD_GOAL_ATTEMPTS_MAX_59_YARDS_COLUMN,
    PLAYER_FIELD_GOAL_ATTEMPTS_MAX_99_YARDS_COLUMN,
    PLAYER_FIELD_GOALS_BLOCKED_COLUMN,
    PLAYER_FIELD_GOALS_BLOCKED_PERCENTAGE_COLUMN,
    PLAYER_FIELD_GOALS_MADE_ABOVE_50_YARDS_COLUMN,
    PLAYER_FIELD_GOALS_MADE_COLUMN,
    PLAYER_FIELD_GOALS_MADE_MAX_19_YARDS_COLUMN,
    PLAYER_FIELD_GOALS_MADE_MAX_29_YARDS_COLUMN,
    PLAYER_FIELD_GOALS_MADE_MAX_39_YARDS_COLUMN,
    PLAYER_FIELD_GOALS_MADE_MAX_49_YARDS_COLUMN,
    PLAYER_FIELD_GOALS_MADE_MAX_59_YARDS_COLUMN,
    PLAYER_FIELD_GOALS_MADE_MAX_99_YARDS_COLUMN,
    PLAYER_FIELD_GOALS_MADE_YARDS_COLUMN,
    PLAYER_FIELD_GOALS_MISSED_YARDS_COLUMN,
    PLAYER_FIELD_GOALS_PERCENTAGE_COLUMN, PLAYER_FORCED_FUMBLES_COLUMN,
    PLAYER_FREE_KICKS_AGAINST_COLUMN, PLAYER_FREE_KICKS_FOR_COLUMN,
    PLAYER_FREE_THROWS_ATTEMPTED_COLUMN, PLAYER_FREE_THROWS_COLUMN,
    PLAYER_FREE_THROWS_PERCENTAGE_COLUMN, PLAYER_FUMBLE_RECOVERIES_COLUMN,
    PLAYER_FUMBLE_RECOVERY_YARDS_COLUMN, PLAYER_FUMBLES_COLUMN,
    PLAYER_FUMBLES_LOST_COLUMN, PLAYER_FUMBLES_RECOVERED_COLUMN,
    PLAYER_FUMBLES_RECOVERED_YARDS_COLUMN, PLAYER_FUMBLES_TOUCHDOWNS_COLUMN,
    PLAYER_GAME_SCORE_COLUMN, PLAYER_GAME_WINNING_GOALS_COLUMN,
    PLAYER_GOAL_ASSISTS_COLUMN, PLAYER_GOALS_AGAINST_COLUMN,
    PLAYER_GOALS_COLUMN, PLAYER_GROSS_AVERAGE_PUNT_YARDS_COLUMN,
    PLAYER_HANDBALLS_COLUMN, PLAYER_HEADSHOT_COLUMN, PLAYER_HEIGHT_COLUMN,
    PLAYER_HIT_OUTS_COLUMN, PLAYER_HITS_COLUMN, PLAYER_HURRIES_COLUMN,
    PLAYER_INDIVIDUAL_CORSI_FOR_EVENTS_COLUMN, PLAYER_INSIDES_COLUMN,
    PLAYER_INTERCEPTION_PERCENTAGE_COLUMN,
    PLAYER_INTERCEPTION_TOUCHDOWNS_COLUMN, PLAYER_INTERCEPTION_YARDS_COLUMN,
    PLAYER_INTERCEPTIONS_COLUMN,
    PLAYER_KICK_RETURN_FAIR_CATCH_PERCENTAGE_COLUMN,
    PLAYER_KICK_RETURN_FAIR_CATCHES_COLUMN, PLAYER_KICK_RETURN_FUMBLES_COLUMN,
    PLAYER_KICK_RETURN_FUMBLES_LOST_COLUMN,
    PLAYER_KICK_RETURN_TOUCHDOWNS_COLUMN, PLAYER_KICK_RETURN_YARDS_COLUMN,
    PLAYER_KICK_RETURNS_COLUMN, PLAYER_KICKOFF_OUT_OF_BOUNDS_COLUMN,
    PLAYER_KICKOFF_RETURN_YARDS_COLUMN, PLAYER_KICKOFF_RETURNS_COLUMN,
    PLAYER_KICKOFF_RETURNS_TOUCHDOWNS_COLUMN, PLAYER_KICKOFF_YARDS_COLUMN,
    PLAYER_KICKOFFS_COLUMN, PLAYER_KICKS_BLOCKED_COLUMN, PLAYER_KICKS_COLUMN,
    PLAYER_LONG_FIELD_GOAL_ATTEMPT_COLUMN, PLAYER_LONG_FIELD_GOAL_MADE_COLUMN,
    PLAYER_LONG_INTERCEPTION_COLUMN, PLAYER_LONG_KICK_RETURN_COLUMN,
    PLAYER_LONG_KICKOFF_COLUMN, PLAYER_LONG_PASSING_COLUMN,
    PLAYER_LONG_PUNT_COLUMN, PLAYER_LONG_PUNT_RETURN_COLUMN,
    PLAYER_LONG_RECEPTION_COLUMN, PLAYER_LONG_RUSHING_COLUMN,
    PLAYER_MARKS_COLUMN, PLAYER_MARKS_INSIDE_COLUMN,
    PLAYER_MISC_FUMBLE_RETURN_YARDS_COLUMN, PLAYER_MISC_FUMBLE_RETURNS_COLUMN,
    PLAYER_MISC_POINTS_COLUMN, PLAYER_MISC_TOUCHDOWNS_COLUMN,
    PLAYER_MISC_YARDS_COLUMN, PLAYER_MISSED_FIELD_GOAL_RETURN_TD_COLUMN,
    PLAYER_NET_AVERAGE_PUNT_YARDS_COLUMN, PLAYER_NET_PASSING_ATTEMPTS_COLUMN,
    PLAYER_NET_PASSING_YARDS_COLUMN, PLAYER_NET_TOTAL_YARDS_COLUMN,
    PLAYER_NET_YARDS_PER_PASS_ATTEMPT_COLUMN,
    PLAYER_OFFENSIVE_FUMBLES_TOUCHDOWNS_COLUMN,
    PLAYER_OFFENSIVE_TWO_POINT_RETURNS_COLUMN,
    PLAYER_OFFENSIVE_ZONE_START_PERCENTAGE_COLUMN,
    PLAYER_OFFENSIVE_ZONE_STARTS_COLUMN,
    PLAYER_ON_SHOT_ICE_AGAINST_EVENTS_COLUMN,
    PLAYER_ON_SHOT_ICE_FOR_EVENTS_COLUMN, PLAYER_ONE_PERCENTERS_COLUMN,
    PLAYER_ONE_POINT_SAFETIES_MADE_COLUMN,
    PLAYER_OPPOSITION_FUMBLE_RECOVERIES_COLUMN,
    PLAYER_OPPOSITION_FUMBLE_RECOVERY_YARDS_COLUMN,
    PLAYER_OPPOSITION_SPECIAL_TEAM_FUMBLE_RETURN_YARDS_COLUMN,
    PLAYER_OPPOSITION_SPECIAL_TEAM_FUMBLE_RETURNS_COLUMN,
    PLAYER_PASSES_BATTED_DOWN_COLUMN, PLAYER_PASSES_DEFENDED_COLUMN,
    PLAYER_PASSING_ATTEMPTS_COLUMN, PLAYER_PASSING_BIG_PLAYS_COLUMN,
    PLAYER_PASSING_FIRST_DOWNS_COLUMN, PLAYER_PASSING_FUMBLES_COLUMN,
    PLAYER_PASSING_FUMBLES_LOST_COLUMN,
    PLAYER_PASSING_TOUCHDOWN_PERCENTAGE_COLUMN,
    PLAYER_PASSING_TOUCHDOWNS_9_YARDS_COLUMN,
    PLAYER_PASSING_TOUCHDOWNS_19_YARDS_COLUMN,
    PLAYER_PASSING_TOUCHDOWNS_29_YARDS_COLUMN,
    PLAYER_PASSING_TOUCHDOWNS_39_YARDS_COLUMN,
    PLAYER_PASSING_TOUCHDOWNS_49_YARDS_COLUMN,
    PLAYER_PASSING_TOUCHDOWNS_ABOVE_50_YARDS_COLUMN,
    PLAYER_PASSING_TOUCHDOWNS_COLUMN, PLAYER_PASSING_YARDS_AFTER_CATCH_COLUMN,
    PLAYER_PASSING_YARDS_AT_CATCH_COLUMN, PLAYER_PASSING_YARDS_COLUMN,
    PLAYER_PENALTIES_IN_MINUTES_COLUMN, PLAYER_PERCENTAGE_PLAYED_COLUMN,
    PLAYER_PERSONAL_FOULS_COLUMN, PLAYER_POINT_DIFFERENTIAL_COLUMN,
    PLAYER_POINTS_ALLOWED_COLUMN, PLAYER_POINTS_COLUMN,
    PLAYER_POWER_PLAY_ASSISTS_COLUMN, PLAYER_POWER_PLAY_GOALS_COLUMN,
    PLAYER_PUNT_RETURN_FAIR_CATCH_PERCENTAGE_COLUMN,
    PLAYER_PUNT_RETURN_FAIR_CATCHES_COLUMN, PLAYER_PUNT_RETURN_FUMBLES_COLUMN,
    PLAYER_PUNT_RETURN_FUMBLES_LOST_COLUMN,
    PLAYER_PUNT_RETURN_TOUCHDOWNS_COLUMN, PLAYER_PUNT_RETURN_YARDS_COLUMN,
    PLAYER_PUNT_RETURNS_COLUMN,
    PLAYER_PUNT_RETURNS_STARTED_INSIDE_THE_10_COLUMN,
    PLAYER_PUNT_RETURNS_STARTED_INSIDE_THE_20_COLUMN, PLAYER_PUNT_YARDS_COLUMN,
    PLAYER_PUNTS_BLOCKED_COLUMN, PLAYER_PUNTS_BLOCKED_PERCENTAGE_COLUMN,
    PLAYER_PUNTS_COLUMN, PLAYER_PUNTS_INSIDE_10_COLUMN,
    PLAYER_PUNTS_INSIDE_10_PERCENTAGE_COLUMN, PLAYER_PUNTS_INSIDE_20_COLUMN,
    PLAYER_PUNTS_INSIDE_20_PERCENTAGE_COLUMN, PLAYER_PUNTS_OVER_50_COLUMN,
    PLAYER_QUARTERBACK_HITS_COLUMN, PLAYER_QUARTERBACK_RATING_COLUMN,
    PLAYER_REBOUNDS_COLUMN, PLAYER_RECEIVING_BIG_PLAYS_COLUMN,
    PLAYER_RECEIVING_FIRST_DOWNS_COLUMN, PLAYER_RECEIVING_FUMBLES_COLUMN,
    PLAYER_RECEIVING_FUMBLES_LOST_COLUMN, PLAYER_RECEIVING_TARGETS_COLUMN,
    PLAYER_RECEIVING_TOUCHDOWNS_9_YARDS_COLUMN,
    PLAYER_RECEIVING_TOUCHDOWNS_19_YARDS_COLUMN,
    PLAYER_RECEIVING_TOUCHDOWNS_29_YARDS_COLUMN,
    PLAYER_RECEIVING_TOUCHDOWNS_39_YARDS_COLUMN,
    PLAYER_RECEIVING_TOUCHDOWNS_49_YARDS_COLUMN,
    PLAYER_RECEIVING_TOUCHDOWNS_ABOVE_50_YARDS_COLUMN,
    PLAYER_RECEIVING_TOUCHDOWNS_COLUMN,
    PLAYER_RECEIVING_YARDS_AFTER_CATCH_COLUMN,
    PLAYER_RECEIVING_YARDS_AT_CATCH_COLUMN, PLAYER_RECEIVING_YARDS_COLUMN,
    PLAYER_RECEPTIONS_COLUMN, PLAYER_RELATIVE_CORSI_FOR_PERCENTAGE_COLUMN,
    PLAYER_RETURN_TOUCHDOWNS_COLUMN, PLAYER_RUSHING_ATTEMPTS_COLUMN,
    PLAYER_RUSHING_BIG_PLAYS_COLUMN, PLAYER_RUSHING_FIRST_DOWNS_COLUMN,
    PLAYER_RUSHING_FUMBLES_COLUMN, PLAYER_RUSHING_FUMBLES_LOST_COLUMN,
    PLAYER_RUSHING_TOUCHDOWNS_9_YARDS_COLUMN,
    PLAYER_RUSHING_TOUCHDOWNS_19_YARDS_COLUMN,
    PLAYER_RUSHING_TOUCHDOWNS_29_YARDS_COLUMN,
    PLAYER_RUSHING_TOUCHDOWNS_39_YARDS_COLUMN,
    PLAYER_RUSHING_TOUCHDOWNS_49_YARDS_COLUMN,
    PLAYER_RUSHING_TOUCHDOWNS_ABOVE_50_YARDS_COLUMN,
    PLAYER_RUSHING_TOUCHDOWNS_COLUMN, PLAYER_RUSHING_YARDS_COLUMN,
    PLAYER_SACKS_ASSISTED_COLUMN, PLAYER_SACKS_COLUMN,
    PLAYER_SACKS_UNASSISTED_COLUMN, PLAYER_SACKS_YARDS_COLUMN,
    PLAYER_SACKS_YARDS_LOST_COLUMN, PLAYER_SAFETIES_COLUMN,
    PLAYER_SAVE_PERCENTAGE_COLUMN, PLAYER_SAVES_COLUMN,
    PLAYER_SECONDS_PLAYED_COLUMN, PLAYER_SHIFTS_COLUMN,
    PLAYER_SHOOTING_PERCENTAGE_COLUMN, PLAYER_SHORT_HANDED_ASSISTS_COLUMN,
    PLAYER_SHORT_HANDED_GOALS_COLUMN, PLAYER_SHOTS_AGAINST_COLUMN,
    PLAYER_SHOTS_ON_GOAL_COLUMN, PLAYER_SHUTOUTS_COLUMN,
    PLAYER_SOLO_TACKLES_COLUMN, PLAYER_SPECIAL_TEAM_FUMBLE_RETURN_YARDS_COLUMN,
    PLAYER_SPECIAL_TEAM_FUMBLE_RETURNS_COLUMN, PLAYER_STEALS_COLUMN,
    PLAYER_STUFF_YARDS_COLUMN, PLAYER_STUFF_YARDS_LOST, PLAYER_STUFFS_COLUMN,
    PLAYER_TACKLES_COLUMN, PLAYER_TACKLES_FOR_LOSS_COLUMN,
    PLAYER_TACKLES_YARDS_LOST_COLUMN,
    PLAYER_THREE_POINT_FIELD_GOALS_ATTEMPTED_COLUMN,
    PLAYER_THREE_POINT_FIELD_GOALS_COLUMN,
    PLAYER_THREE_POINT_FIELD_GOALS_PERCENTAGE_COLUMN,
    PLAYER_TIME_ON_ICE_COLUMN, PLAYER_TOTAL_KICKING_POINTS_COLUMN,
    PLAYER_TOTAL_OFFENSIVE_PLAYS_COLUMN, PLAYER_TOTAL_POINTS_COLUMN,
    PLAYER_TOTAL_REBOUNDS_COLUMN, PLAYER_TOTAL_TOUCHDOWNS_COLUMN,
    PLAYER_TOTAL_TWO_POINT_CONVERSIONS_COLUMN, PLAYER_TOTAL_YARDS_COLUMN,
    PLAYER_TOTAL_YARDS_FROM_SCRIMMAGE_COLUMN,
    PLAYER_TOUCHBACK_PERCENTAGE_COLUMN, PLAYER_TOUCHBACKS_COLUMN,
    PLAYER_TRUE_SHOOTING_PERCENTAGE_COLUMN,
    PLAYER_TWO_POINT_PASS_ATTEMPT_COLUMN, PLAYER_TWO_POINT_PASS_COLUMN,
    PLAYER_TWO_POINT_RECEPTION_ATTEMPTS_COLUMN,
    PLAYER_TWO_POINT_RECEPTIONS_COLUMN, PLAYER_TWO_POINT_RUSH_ATTEMPTS_COLUMN,
    PLAYER_TWO_POINT_RUSH_COLUMN, PLAYER_UNCONTESTED_POSSESSIONS_COLUMN,
    PLAYER_YARDS_ALLOWED_COLUMN, PLAYER_YARDS_PER_COMPLETION_COLUMN,
    PLAYER_YARDS_PER_KICK_RETURN_COLUMN, PLAYER_YARDS_PER_PASS_ATTEMPT_COLUMN,
    PLAYER_YARDS_PER_RECEPTION_COLUMN, PLAYER_YARDS_PER_RETURN_COLUMN,
    PLAYER_YARDS_PER_RUSH_ATTEMPT_COLUMN)
from sportsball.data.player_model import \
    TURNOVERS_COLUMN as PLAYER_TURNOVERS_COLUMN  # type: ignore
from sportsball.data.team_model import ASSISTS_COLUMN  # type: ignore
from sportsball.data.team_model import (
    FIELD_GOALS_ATTEMPTED_COLUMN, FIELD_GOALS_COLUMN, KICKS_COLUMN,
    OFFENSIVE_REBOUNDS_COLUMN, TEAM_BEHINDS_COLUMN, TEAM_BLOCKS_COLUMN,
    TEAM_BOUNCES_COLUMN, TEAM_BROWNLOW_VOTES_COLUMN, TEAM_CLANGERS_COLUMN,
    TEAM_CLEARANCES_COLUMN, TEAM_CONTESTED_MARKS_COLUMN,
    TEAM_CONTESTED_POSSESSIONS_COLUMN, TEAM_DEFENSIVE_REBOUNDS_COLUMN,
    TEAM_DISPOSALS_COLUMN, TEAM_FIELD_GOALS_PERCENTAGE_COLUMN,
    TEAM_FORCED_FUMBLES_COLUMN, TEAM_FREE_KICKS_AGAINST_COLUMN,
    TEAM_FREE_KICKS_FOR_COLUMN, TEAM_FREE_THROWS_ATTEMPTED_COLUMN,
    TEAM_FREE_THROWS_COLUMN, TEAM_FREE_THROWS_PERCENTAGE_COLUMN,
    TEAM_FUMBLES_RECOVERED_COLUMN, TEAM_FUMBLES_TOUCHDOWNS_COLUMN,
    TEAM_GOAL_ASSISTS_COLUMN, TEAM_GOALS_COLUMN, TEAM_HANDBALLS_COLUMN,
    TEAM_HIT_OUTS_COLUMN, TEAM_INSIDES_COLUMN,
    TEAM_LENGTH_BEHIND_WINNER_COLUMN, TEAM_MARKS_COLUMN,
    TEAM_MARKS_INSIDE_COLUMN, TEAM_ONE_PERCENTERS_COLUMN,
    TEAM_PERSONAL_FOULS_COLUMN, TEAM_REBOUNDS_COLUMN, TEAM_STEALS_COLUMN,
    TEAM_TACKLES_COLUMN, TEAM_THREE_POINT_FIELD_GOALS_ATTEMPTED_COLUMN,
    TEAM_THREE_POINT_FIELD_GOALS_COLUMN,
    TEAM_THREE_POINT_FIELD_GOALS_PERCENTAGE_COLUMN, TEAM_TOTAL_REBOUNDS_COLUMN,
    TEAM_UNCONTESTED_POSSESSIONS_COLUMN, TURNOVERS_COLUMN)
from sportsball.data.venue_model import VENUE_ADDRESS_COLUMN
from sportsfeatures.bet import Bet
from sportsfeatures.embedding_column import is_embedding_column
from sportsfeatures.entity_type import EntityType  # type: ignore
from sportsfeatures.identifier import Identifier  # type: ignore
from sportsfeatures.news import News
from sportsfeatures.process import process  # type: ignore

from .features.columns import (coach_column_prefix, coach_identifier_column,
                               find_coach_count, find_news_count,
                               find_odds_count, find_player_count,
                               find_team_count, news_column_prefix,
                               news_summary_column, odds_column_prefix,
                               odds_odds_column, player_column_prefix,
                               player_identifier_column, team_column_prefix,
                               team_identifier_column, team_name_column,
                               team_points_column, venue_identifier_column)
from .kelly_fractions import (augment_kelly_fractions, calculate_returns,
                              calculate_value)

AWAY_WIN_COLUMN = "away_win"

_DF_FILENAME = "df.parquet.gzip"
_CONFIG_FILENAME = "config.json"
_PLACE_KEY = "place"
_VALIDATION_SIZE = datetime.timedelta(days=365)


class Strategy:
    """The strategy class."""

    # pylint: disable=too-many-locals,too-many-instance-attributes

    _returns: pd.Series | None
    _place: int

    def __init__(self, name: str, place: int | None = None) -> None:
        self._df = None
        self._name = name
        os.makedirs(name, exist_ok=True)

        # Load dataframe previously used.
        df_file = os.path.join(name, _DF_FILENAME)
        if os.path.exists(df_file):
            self._df = pd.read_parquet(df_file)

        self._wt = wt.create(
            self._name,
            dt_column=GAME_DT_COLUMN,
            walkforward_timedelta=datetime.timedelta(days=7),
            validation_size=_VALIDATION_SIZE,
            max_train_timeout=datetime.timedelta(hours=12),
            cutoff_dt=datetime.datetime.now(tz=pytz.UTC),
            test_size=datetime.timedelta(days=365 * 2),
            allowed_models={"catboost"},
            max_false_positive_reduction_steps=1,
            correlation_chunk_size=5000,
        )

        # Load config
        config_filename = os.path.join(name, _CONFIG_FILENAME)
        if os.path.exists(config_filename) and place is None:
            with open(config_filename, "r", encoding="utf8") as handle:
                config = json.load(handle)
                place = config.get(_PLACE_KEY)
        elif place is not None:
            with open(config_filename, "w", encoding="utf8") as handle:
                json.dump({_PLACE_KEY: place}, handle)
        self._place = place if place is not None else 1

        self._returns = None

    @property
    def df(self) -> pd.DataFrame | None:
        """Fetch the dataframe currently being operated on."""
        df = self._df
        if df is None:
            return None
        return df.sort_values(by=DT_COLUMN, ascending=True)

    @df.setter
    def df(self, df: pd.DataFrame) -> None:
        """Set the dataframe."""
        self._df = df.sort_values(by=DT_COLUMN, ascending=True)
        df.to_parquet(os.path.join(self._name, _DF_FILENAME), compression="gzip")
        self._df = pd.read_parquet(os.path.join(self._name, _DF_FILENAME))

    @property
    def name(self) -> str:
        """Fetch the name of the strategy."""
        return self._name

    def kelly_ratio(self, df: pd.DataFrame) -> float:
        """Find the best kelly ratio for this strategy."""
        main_df = self.df
        if main_df is None:
            raise ValueError("main_df is null")
        points_cols = main_df.attrs[str(FieldType.POINTS)]
        df[points_cols] = main_df[points_cols].to_numpy()
        cutoff_dt = pd.to_datetime(datetime.datetime.now() - _VALIDATION_SIZE).date()
        df = df[df[GAME_DT_COLUMN].dt.date > cutoff_dt]
        df = augment_kelly_fractions(df, len(points_cols))
        df.to_parquet(os.path.join(self._name, "returns_df.parquet.gzip"))
        max_return = 0.0
        max_kelly = 0.0
        for i in range(100):
            test_kelly_ratio = (100 - i) / 100.0
            returns = calculate_returns(
                test_kelly_ratio,
                df.copy(),
                self._name,
            )
            value = calculate_value(returns)
            if value > max_return or max_kelly == 0.0:
                max_return = value
                max_kelly = test_kelly_ratio
        print(f"Max Kelly: {max_kelly}")
        self._returns = calculate_returns(max_kelly, df.copy(), self._name)

        return max_kelly

    def fit(self):
        """Fits the strategy to the dataset by walking forward."""
        df = self.df
        if df is None:
            raise ValueError("df is null")
        training_cols = df.attrs[str(FieldType.POINTS)]
        x_df = self._process()
        y = df[training_cols]
        teams = find_team_count(df)

        def make_y() -> pd.Series | pd.DataFrame:
            nonlocal y
            if teams == 2:
                y_max = np.argmax(y.to_numpy(), axis=1)
                y[AWAY_WIN_COLUMN] = y_max
                y[AWAY_WIN_COLUMN] = y[AWAY_WIN_COLUMN].astype(bool)
                return y[AWAY_WIN_COLUMN]
            ind = np.argpartition(y.to_numpy(), -self._place)[-self._place :]
            for i in range(teams):
                y[DELIMITER.join(["team", str(i), "win"])] = i in ind
            return y.drop(columns=training_cols)  # type: ignore

        y = make_y()
        x_df = x_df.drop(columns=training_cols)
        x_df = x_df.drop(columns=df.attrs[str(FieldType.LOOKAHEAD)], errors="ignore")
        self._wt.embedding_cols = self._calculate_embedding_columns(x_df)
        self._wt.fit(x_df, y=y)

    def predict(self) -> pd.DataFrame:
        """Predict the results from walk-forward."""
        df = self.df
        if df is None:
            raise ValueError("df is null.")

        x_df = self._process()
        training_cols = df.attrs[str(FieldType.POINTS)]
        x_df = x_df.drop(columns=training_cols, errors="ignore")
        x_df = x_df.drop(columns=df.attrs[str(FieldType.LOOKAHEAD)], errors="ignore")
        self._wt.embedding_cols = self._calculate_embedding_columns(x_df)

        # Ensure correct odds
        today = (datetime.datetime.today() - datetime.timedelta(days=1)).date()
        future_rows = x_df[x_df[GAME_DT_COLUMN].dt.date >= today]
        for idx, row in future_rows.iterrows():
            for team_id in range(find_team_count(x_df)):
                odds_col = f"teams/{team_id}_odds"
                if pd.isna(row.get(odds_col)):
                    while True:
                        name_col = team_name_column(team_id)
                        try:
                            new_odds = float(
                                input(
                                    f"Enter new odds for {odds_col} at row {idx} for team {row.get(name_col)} @ {row.get(GAME_DT_COLUMN)}: "
                                )
                            )
                            x_df.at[idx, odds_col] = new_odds
                            break
                        except ValueError:
                            print("Invalid input. Please enter a numeric value.")

        x_df = self._wt.transform(x_df)
        for points_col in df.attrs[str(FieldType.POINTS)]:
            x_df[points_col] = df[points_col]
        return x_df

    def returns(self) -> pd.Series:
        """Render the returns of the strategy."""
        df = self.predict()
        self.kelly_ratio(df)
        returns = self._returns
        if returns is None:
            raise ValueError("returns is null")
        return returns

    def next(
        self,
    ) -> tuple[
        pd.DataFrame, dict[str, tuple[dict[str, float], list[dict[str, float]]]], float
    ]:
        """Find the next predictions for betting."""
        dt_column = DELIMITER.join([GAME_DT_COLUMN])
        df = self.predict()
        kelly_ratio = self.kelly_ratio(df)
        start_dt = datetime.datetime.now(datetime.timezone.utc)
        end_dt = start_dt + datetime.timedelta(days=3.0)
        df = df[df[dt_column] > start_dt]
        df = df[df[dt_column] <= end_dt]
        return df, self._wt.feature_importances(df=df), kelly_ratio

    def _process(self) -> pd.DataFrame:
        df = self.df
        if df is None:
            raise ValueError("df is null")

        df_hash = hashlib.sha256(df.to_csv().encode()).hexdigest()
        df_cache_path = os.path.join(self._name, f"processed_{df_hash}.parquet")
        if os.path.exists(df_cache_path):
            return pd.read_parquet(df_cache_path)

        team_count = find_team_count(df)

        identifiers = [
            Identifier(
                EntityType.VENUE,
                venue_identifier_column(),
                [],
                VENUE_COLUMN_PREFIX,
                latitude_column=DELIMITER.join(
                    [VENUE_COLUMN_PREFIX, VENUE_ADDRESS_COLUMN, ADDRESS_LATITUDE_COLUMN]
                ),
                longitude_column=DELIMITER.join(
                    [
                        VENUE_COLUMN_PREFIX,
                        VENUE_ADDRESS_COLUMN,
                        ADDRESS_LONGITUDE_COLUMN,
                    ]
                ),
            )
        ]
        odds_count = find_odds_count(df, team_count)
        news_count = find_news_count(df, team_count)
        datetime_columns: set[str] = set()
        for i in range(team_count):
            identifiers.append(
                Identifier(
                    EntityType.TEAM,
                    team_identifier_column(i),
                    [
                        DELIMITER.join([team_column_prefix(i), x])
                        for x in [
                            FIELD_GOALS_COLUMN,
                            FIELD_GOALS_ATTEMPTED_COLUMN,
                            OFFENSIVE_REBOUNDS_COLUMN,
                            ASSISTS_COLUMN,
                            TURNOVERS_COLUMN,
                            KICKS_COLUMN,
                            TEAM_MARKS_COLUMN,
                            TEAM_HANDBALLS_COLUMN,
                            TEAM_DISPOSALS_COLUMN,
                            TEAM_GOALS_COLUMN,
                            TEAM_BEHINDS_COLUMN,
                            TEAM_HIT_OUTS_COLUMN,
                            TEAM_TACKLES_COLUMN,
                            TEAM_REBOUNDS_COLUMN,
                            TEAM_INSIDES_COLUMN,
                            TEAM_CLEARANCES_COLUMN,
                            TEAM_CLANGERS_COLUMN,
                            TEAM_FREE_KICKS_FOR_COLUMN,
                            TEAM_FREE_KICKS_AGAINST_COLUMN,
                            TEAM_BROWNLOW_VOTES_COLUMN,
                            TEAM_CONTESTED_POSSESSIONS_COLUMN,
                            TEAM_UNCONTESTED_POSSESSIONS_COLUMN,
                            TEAM_CONTESTED_MARKS_COLUMN,
                            TEAM_MARKS_INSIDE_COLUMN,
                            TEAM_ONE_PERCENTERS_COLUMN,
                            TEAM_BOUNCES_COLUMN,
                            TEAM_GOAL_ASSISTS_COLUMN,
                            TEAM_LENGTH_BEHIND_WINNER_COLUMN,
                            TEAM_FIELD_GOALS_PERCENTAGE_COLUMN,
                            TEAM_THREE_POINT_FIELD_GOALS_COLUMN,
                            TEAM_THREE_POINT_FIELD_GOALS_ATTEMPTED_COLUMN,
                            TEAM_THREE_POINT_FIELD_GOALS_PERCENTAGE_COLUMN,
                            TEAM_FREE_THROWS_COLUMN,
                            TEAM_FREE_THROWS_ATTEMPTED_COLUMN,
                            TEAM_FREE_THROWS_PERCENTAGE_COLUMN,
                            TEAM_DEFENSIVE_REBOUNDS_COLUMN,
                            TEAM_TOTAL_REBOUNDS_COLUMN,
                            TEAM_STEALS_COLUMN,
                            TEAM_BLOCKS_COLUMN,
                            TEAM_PERSONAL_FOULS_COLUMN,
                            TEAM_FORCED_FUMBLES_COLUMN,
                            TEAM_FUMBLES_RECOVERED_COLUMN,
                            TEAM_FUMBLES_TOUCHDOWNS_COLUMN,
                        ]
                    ],
                    team_column_prefix(i),
                    points_column=team_points_column(i),
                    field_goals_column=DELIMITER.join(
                        [team_column_prefix(i), FIELD_GOALS_COLUMN]
                    ),
                    assists_column=DELIMITER.join(
                        [team_column_prefix(i), ASSISTS_COLUMN]
                    ),
                    field_goals_attempted_column=DELIMITER.join(
                        [team_column_prefix(i), FIELD_GOALS_ATTEMPTED_COLUMN]
                    ),
                    offensive_rebounds_column=DELIMITER.join(
                        [team_column_prefix(i), OFFENSIVE_REBOUNDS_COLUMN]
                    ),
                    turnovers_column=DELIMITER.join(
                        [team_column_prefix(i), TURNOVERS_COLUMN]
                    ),
                    bets=[
                        Bet(
                            odds_column=odds_odds_column(i, x),
                            bookie_id_column=DELIMITER.join(
                                [
                                    odds_column_prefix(i, x),
                                    ODDS_BOOKIE_COLUMN,
                                    BOOKIE_IDENTIFIER_COLUMN,
                                ]
                            ),
                            dt_column=DELIMITER.join(
                                [odds_column_prefix(i, x), DT_COLUMN]
                            ),
                            canonical_column=DELIMITER.join(
                                [odds_column_prefix(i, x), ODDS_CANONICAL_COLUMN]
                            ),
                            bookie_name_column=DELIMITER.join(
                                [
                                    odds_column_prefix(i, x),
                                    ODDS_BOOKIE_COLUMN,
                                    "name",
                                ]
                            ),
                            bet_type_column=DELIMITER.join(
                                [
                                    odds_column_prefix(i, x),
                                    ODDS_BET_COLUMN,
                                ]
                            ),
                        )
                        for x in range(odds_count)
                    ],
                    news=[
                        News(
                            title_column=DELIMITER.join(
                                [news_column_prefix(i, x), NEWS_TITLE_COLUMN]
                            ),
                            published_column=DELIMITER.join(
                                [news_column_prefix(i, x), NEWS_PUBLISHED_COLUMN]
                            ),
                            summary_column=news_summary_column(i, x),
                            source_column=DELIMITER.join(
                                [news_column_prefix(i, x), NEWS_SOURCE_COLUMN]
                            ),
                        )
                        for x in range(news_count)
                    ],
                )
            )
            player_count = find_player_count(df, i)
            identifiers.extend(
                [
                    Identifier(
                        EntityType.PLAYER,
                        player_identifier_column(i, x),
                        [
                            DELIMITER.join([player_column_prefix(i, x), col])
                            for col in [
                                PLAYER_KICKS_COLUMN,
                                PLAYER_FUMBLES_COLUMN,
                                PLAYER_FUMBLES_LOST_COLUMN,
                                PLAYER_FIELD_GOALS_COLUMN,
                                PLAYER_FIELD_GOALS_ATTEMPTED_COLUMN,
                                PLAYER_OFFENSIVE_REBOUNDS_COLUMN,
                                PLAYER_ASSISTS_COLUMN,
                                PLAYER_TURNOVERS_COLUMN,
                                PLAYER_MARKS_COLUMN,
                                PLAYER_HANDBALLS_COLUMN,
                                PLAYER_DISPOSALS_COLUMN,
                                PLAYER_GOALS_COLUMN,
                                PLAYER_BEHINDS_COLUMN,
                                PLAYER_HIT_OUTS_COLUMN,
                                PLAYER_TACKLES_COLUMN,
                                PLAYER_REBOUNDS_COLUMN,
                                PLAYER_INSIDES_COLUMN,
                                PLAYER_CLEARANCES_COLUMN,
                                PLAYER_CLANGERS_COLUMN,
                                PLAYER_FREE_KICKS_FOR_COLUMN,
                                PLAYER_FREE_KICKS_AGAINST_COLUMN,
                                PLAYER_BROWNLOW_VOTES_COLUMN,
                                PLAYER_CONTESTED_POSSESSIONS_COLUMN,
                                PLAYER_UNCONTESTED_POSSESSIONS_COLUMN,
                                PLAYER_CONTESTED_MARKS_COLUMN,
                                PLAYER_MARKS_INSIDE_COLUMN,
                                PLAYER_ONE_PERCENTERS_COLUMN,
                                PLAYER_BOUNCES_COLUMN,
                                PLAYER_GOAL_ASSISTS_COLUMN,
                                PLAYER_PERCENTAGE_PLAYED_COLUMN,
                                PLAYER_SECONDS_PLAYED_COLUMN,
                                PLAYER_FIELD_GOALS_PERCENTAGE_COLUMN,
                                PLAYER_THREE_POINT_FIELD_GOALS_COLUMN,
                                PLAYER_THREE_POINT_FIELD_GOALS_ATTEMPTED_COLUMN,
                                PLAYER_THREE_POINT_FIELD_GOALS_PERCENTAGE_COLUMN,
                                PLAYER_FREE_THROWS_COLUMN,
                                PLAYER_FREE_THROWS_ATTEMPTED_COLUMN,
                                PLAYER_FREE_THROWS_PERCENTAGE_COLUMN,
                                PLAYER_DEFENSIVE_REBOUNDS_COLUMN,
                                PLAYER_TOTAL_REBOUNDS_COLUMN,
                                PLAYER_STEALS_COLUMN,
                                PLAYER_BLOCKS_COLUMN,
                                PLAYER_PERSONAL_FOULS_COLUMN,
                                PLAYER_POINTS_COLUMN,
                                PLAYER_GAME_SCORE_COLUMN,
                                PLAYER_POINT_DIFFERENTIAL_COLUMN,
                                PLAYER_HEIGHT_COLUMN,
                                PLAYER_FORCED_FUMBLES_COLUMN,
                                PLAYER_FUMBLES_RECOVERED_COLUMN,
                                PLAYER_FUMBLES_RECOVERED_YARDS_COLUMN,
                                PLAYER_FUMBLES_TOUCHDOWNS_COLUMN,
                                PLAYER_OFFENSIVE_TWO_POINT_RETURNS_COLUMN,
                                PLAYER_OFFENSIVE_FUMBLES_TOUCHDOWNS_COLUMN,
                                PLAYER_DEFENSIVE_FUMBLES_TOUCHDOWNS_COLUMN,
                                PLAYER_AVERAGE_GAIN_COLUMN,
                                PLAYER_COMPLETION_PERCENTAGE_COLUMN,
                                PLAYER_COMPLETIONS_COLUMN,
                                PLAYER_ESPN_QUARTERBACK_RATING_COLUMN,
                                PLAYER_INTERCEPTION_PERCENTAGE_COLUMN,
                                PLAYER_INTERCEPTIONS_COLUMN,
                                PLAYER_LONG_PASSING_COLUMN,
                                PLAYER_MISC_YARDS_COLUMN,
                                PLAYER_NET_PASSING_YARDS_COLUMN,
                                PLAYER_NET_TOTAL_YARDS_COLUMN,
                                PLAYER_PASSING_ATTEMPTS_COLUMN,
                                PLAYER_PASSING_BIG_PLAYS_COLUMN,
                                PLAYER_PASSING_FIRST_DOWNS_COLUMN,
                                PLAYER_PASSING_FUMBLES_COLUMN,
                                PLAYER_PASSING_FUMBLES_LOST_COLUMN,
                                PLAYER_PASSING_TOUCHDOWN_PERCENTAGE_COLUMN,
                                PLAYER_PASSING_TOUCHDOWNS_COLUMN,
                                PLAYER_PASSING_YARDS_COLUMN,
                                PLAYER_PASSING_YARDS_AFTER_CATCH_COLUMN,
                                PLAYER_PASSING_YARDS_AT_CATCH_COLUMN,
                                PLAYER_QUARTERBACK_RATING_COLUMN,
                                PLAYER_SACKS_COLUMN,
                                PLAYER_SACKS_YARDS_LOST_COLUMN,
                                PLAYER_NET_PASSING_ATTEMPTS_COLUMN,
                                PLAYER_TOTAL_OFFENSIVE_PLAYS_COLUMN,
                                PLAYER_TOTAL_POINTS_COLUMN,
                                PLAYER_TOTAL_TOUCHDOWNS_COLUMN,
                                PLAYER_TOTAL_YARDS_COLUMN,
                                PLAYER_TOTAL_YARDS_FROM_SCRIMMAGE_COLUMN,
                                PLAYER_TWO_POINT_PASS_COLUMN,
                                PLAYER_TWO_POINT_PASS_ATTEMPT_COLUMN,
                                PLAYER_YARDS_PER_COMPLETION_COLUMN,
                                PLAYER_YARDS_PER_PASS_ATTEMPT_COLUMN,
                                PLAYER_NET_YARDS_PER_PASS_ATTEMPT_COLUMN,
                                PLAYER_ESPN_RUNNINGBACK_RATING_COLUMN,
                                PLAYER_LONG_RUSHING_COLUMN,
                                PLAYER_RUSHING_ATTEMPTS_COLUMN,
                                PLAYER_RUSHING_BIG_PLAYS_COLUMN,
                                PLAYER_RUSHING_FIRST_DOWNS_COLUMN,
                                PLAYER_RUSHING_FUMBLES_COLUMN,
                                PLAYER_RUSHING_FUMBLES_LOST_COLUMN,
                                PLAYER_RUSHING_TOUCHDOWNS_COLUMN,
                                PLAYER_RUSHING_YARDS_COLUMN,
                                PLAYER_STUFFS_COLUMN,
                                PLAYER_STUFF_YARDS_LOST,
                                PLAYER_TWO_POINT_RUSH_COLUMN,
                                PLAYER_TWO_POINT_RUSH_ATTEMPTS_COLUMN,
                                PLAYER_YARDS_PER_RUSH_ATTEMPT_COLUMN,
                                PLAYER_ESPN_WIDERECEIVER_COLUMN,
                                PLAYER_LONG_RECEPTION_COLUMN,
                                PLAYER_RECEIVING_BIG_PLAYS_COLUMN,
                                PLAYER_RECEIVING_FIRST_DOWNS_COLUMN,
                                PLAYER_RECEIVING_FUMBLES_COLUMN,
                                PLAYER_RECEIVING_FUMBLES_LOST_COLUMN,
                                PLAYER_RECEIVING_TARGETS_COLUMN,
                                PLAYER_RECEIVING_TOUCHDOWNS_COLUMN,
                                PLAYER_RECEIVING_YARDS_COLUMN,
                                PLAYER_RECEIVING_YARDS_AFTER_CATCH_COLUMN,
                                PLAYER_RECEIVING_YARDS_AT_CATCH_COLUMN,
                                PLAYER_RECEPTIONS_COLUMN,
                                PLAYER_TWO_POINT_RECEPTIONS_COLUMN,
                                PLAYER_TWO_POINT_RECEPTION_ATTEMPTS_COLUMN,
                                PLAYER_YARDS_PER_RECEPTION_COLUMN,
                                PLAYER_ASSIST_TACKLES_COLUMN,
                                PLAYER_AVERAGE_INTERCEPTION_YARDS_COLUMN,
                                PLAYER_AVERAGE_SACK_YARDS_COLUMN,
                                PLAYER_AVERAGE_STUFF_YARDS_COLUMN,
                                PLAYER_BLOCKED_FIELD_GOAL_TOUCHDOWNS_COLUMN,
                                PLAYER_BLOCKED_PUNT_TOUCHDOWNS_COLUMN,
                                PLAYER_DEFENSIVE_TOUCHDOWNS_COLUMN,
                                PLAYER_HURRIES_COLUMN,
                                PLAYER_KICKS_BLOCKED_COLUMN,
                                PLAYER_LONG_INTERCEPTION_COLUMN,
                                PLAYER_MISC_TOUCHDOWNS_COLUMN,
                                PLAYER_PASSES_BATTED_DOWN_COLUMN,
                                PLAYER_PASSES_DEFENDED_COLUMN,
                                PLAYER_QUARTERBACK_HITS_COLUMN,
                                PLAYER_SACKS_ASSISTED_COLUMN,
                                PLAYER_SACKS_UNASSISTED_COLUMN,
                                PLAYER_SACKS_YARDS_COLUMN,
                                PLAYER_SAFETIES_COLUMN,
                                PLAYER_SOLO_TACKLES_COLUMN,
                                PLAYER_STUFF_YARDS_COLUMN,
                                PLAYER_TACKLES_FOR_LOSS_COLUMN,
                                PLAYER_TACKLES_YARDS_LOST_COLUMN,
                                PLAYER_YARDS_ALLOWED_COLUMN,
                                PLAYER_POINTS_ALLOWED_COLUMN,
                                PLAYER_ONE_POINT_SAFETIES_MADE_COLUMN,
                                PLAYER_MISSED_FIELD_GOAL_RETURN_TD_COLUMN,
                                PLAYER_BLOCKED_PUNT_EZ_REC_TD_COLUMN,
                                PLAYER_INTERCEPTION_TOUCHDOWNS_COLUMN,
                                PLAYER_INTERCEPTION_YARDS_COLUMN,
                                PLAYER_AVERAGE_KICKOFF_RETURN_YARDS_COLUMN,
                                PLAYER_AVERAGE_KICKOFF_YARDS_COLUMN,
                                PLAYER_EXTRA_POINT_ATTEMPTS_COLUMN,
                                PLAYER_EXTRA_POINT_PERCENTAGE_COLUMN,
                                PLAYER_EXTRA_POINT_BLOCKED_COLUMN,
                                PLAYER_EXTRA_POINTS_BLOCKED_PERCENTAGE_COLUMN,
                                PLAYER_EXTRA_POINTS_MADE_COLUMN,
                                PLAYER_FAIR_CATCHES_COLUMN,
                                PLAYER_FAIR_CATCH_PERCENTAGE_COLUMN,
                                PLAYER_FIELD_GOAL_ATTEMPTS_MAX_19_YARDS_COLUMN,
                                PLAYER_FIELD_GOAL_ATTEMPTS_MAX_29_YARDS_COLUMN,
                                PLAYER_FIELD_GOAL_ATTEMPTS_MAX_39_YARDS_COLUMN,
                                PLAYER_FIELD_GOAL_ATTEMPTS_MAX_49_YARDS_COLUMN,
                                PLAYER_FIELD_GOAL_ATTEMPTS_MAX_59_YARDS_COLUMN,
                                PLAYER_FIELD_GOAL_ATTEMPTS_MAX_99_YARDS_COLUMN,
                                PLAYER_FIELD_GOAL_ATTEMPTS_ABOVE_50_YARDS_COLUMN,
                                PLAYER_FIELD_GOAL_ATTEMPT_YARDS_COLUMN,
                                PLAYER_FIELD_GOALS_BLOCKED_COLUMN,
                                PLAYER_FIELD_GOALS_BLOCKED_PERCENTAGE_COLUMN,
                                PLAYER_FIELD_GOALS_MADE_COLUMN,
                                PLAYER_FIELD_GOALS_MADE_MAX_19_YARDS_COLUMN,
                                PLAYER_FIELD_GOALS_MADE_MAX_29_YARDS_COLUMN,
                                PLAYER_FIELD_GOALS_MADE_MAX_39_YARDS_COLUMN,
                                PLAYER_FIELD_GOALS_MADE_MAX_49_YARDS_COLUMN,
                                PLAYER_FIELD_GOALS_MADE_MAX_59_YARDS_COLUMN,
                                PLAYER_FIELD_GOALS_MADE_MAX_99_YARDS_COLUMN,
                                PLAYER_FIELD_GOALS_MADE_ABOVE_50_YARDS_COLUMN,
                                PLAYER_FIELD_GOALS_MADE_YARDS_COLUMN,
                                PLAYER_FIELD_GOALS_MISSED_YARDS_COLUMN,
                                PLAYER_KICKOFF_OUT_OF_BOUNDS_COLUMN,
                                PLAYER_KICKOFF_RETURNS_COLUMN,
                                PLAYER_KICKOFF_RETURNS_TOUCHDOWNS_COLUMN,
                                PLAYER_KICKOFF_RETURN_YARDS_COLUMN,
                                PLAYER_KICKOFFS_COLUMN,
                                PLAYER_KICKOFF_YARDS_COLUMN,
                                PLAYER_LONG_FIELD_GOAL_ATTEMPT_COLUMN,
                                PLAYER_LONG_FIELD_GOAL_MADE_COLUMN,
                                PLAYER_LONG_KICKOFF_COLUMN,
                                PLAYER_TOTAL_KICKING_POINTS_COLUMN,
                                PLAYER_TOUCHBACK_PERCENTAGE_COLUMN,
                                PLAYER_TOUCHBACKS_COLUMN,
                                PLAYER_DEFENSIVE_FUMBLE_RETURNS_COLUMN,
                                PLAYER_DEFENSIVE_FUMBLE_RETURN_YARDS_COLUMN,
                                PLAYER_FUMBLE_RECOVERIES_COLUMN,
                                PLAYER_FUMBLE_RECOVERY_YARDS_COLUMN,
                                PLAYER_KICK_RETURN_FAIR_CATCHES_COLUMN,
                                PLAYER_KICK_RETURN_FAIR_CATCH_PERCENTAGE_COLUMN,
                                PLAYER_KICK_RETURN_FUMBLES_COLUMN,
                                PLAYER_KICK_RETURN_FUMBLES_LOST_COLUMN,
                                PLAYER_KICK_RETURNS_COLUMN,
                                PLAYER_KICK_RETURN_TOUCHDOWNS_COLUMN,
                                PLAYER_KICK_RETURN_YARDS_COLUMN,
                                PLAYER_LONG_KICK_RETURN_COLUMN,
                                PLAYER_LONG_PUNT_RETURN_COLUMN,
                                PLAYER_MISC_FUMBLE_RETURNS_COLUMN,
                                PLAYER_MISC_FUMBLE_RETURN_YARDS_COLUMN,
                                PLAYER_OPPOSITION_FUMBLE_RECOVERIES_COLUMN,
                                PLAYER_OPPOSITION_FUMBLE_RECOVERY_YARDS_COLUMN,
                                PLAYER_OPPOSITION_SPECIAL_TEAM_FUMBLE_RETURNS_COLUMN,
                                PLAYER_OPPOSITION_SPECIAL_TEAM_FUMBLE_RETURN_YARDS_COLUMN,
                                PLAYER_PUNT_RETURN_FAIR_CATCHES_COLUMN,
                                PLAYER_PUNT_RETURN_FAIR_CATCH_PERCENTAGE_COLUMN,
                                PLAYER_PUNT_RETURN_FUMBLES_COLUMN,
                                PLAYER_PUNT_RETURN_FUMBLES_LOST_COLUMN,
                                PLAYER_PUNT_RETURNS_COLUMN,
                                PLAYER_PUNT_RETURNS_STARTED_INSIDE_THE_10_COLUMN,
                                PLAYER_PUNT_RETURNS_STARTED_INSIDE_THE_20_COLUMN,
                                PLAYER_PUNT_RETURN_TOUCHDOWNS_COLUMN,
                                PLAYER_PUNT_RETURN_YARDS_COLUMN,
                                PLAYER_SPECIAL_TEAM_FUMBLE_RETURNS_COLUMN,
                                PLAYER_SPECIAL_TEAM_FUMBLE_RETURN_YARDS_COLUMN,
                                PLAYER_YARDS_PER_KICK_RETURN_COLUMN,
                                PLAYER_YARDS_PER_RETURN_COLUMN,
                                PLAYER_AVERAGE_PUNT_RETURN_YARDS_COLUMN,
                                PLAYER_GROSS_AVERAGE_PUNT_YARDS_COLUMN,
                                PLAYER_LONG_PUNT_COLUMN,
                                PLAYER_NET_AVERAGE_PUNT_YARDS_COLUMN,
                                PLAYER_PUNTS_COLUMN,
                                PLAYER_PUNTS_BLOCKED_COLUMN,
                                PLAYER_PUNTS_BLOCKED_PERCENTAGE_COLUMN,
                                PLAYER_PUNTS_INSIDE_10_COLUMN,
                                PLAYER_PUNTS_INSIDE_10_PERCENTAGE_COLUMN,
                                PLAYER_PUNTS_INSIDE_20_COLUMN,
                                PLAYER_PUNTS_INSIDE_20_PERCENTAGE_COLUMN,
                                PLAYER_PUNTS_OVER_50_COLUMN,
                                PLAYER_PUNT_YARDS_COLUMN,
                                PLAYER_DEFENSIVE_POINTS_COLUMN,
                                PLAYER_MISC_POINTS_COLUMN,
                                PLAYER_RETURN_TOUCHDOWNS_COLUMN,
                                PLAYER_TOTAL_TWO_POINT_CONVERSIONS_COLUMN,
                                PLAYER_PASSING_TOUCHDOWNS_9_YARDS_COLUMN,
                                PLAYER_PASSING_TOUCHDOWNS_19_YARDS_COLUMN,
                                PLAYER_PASSING_TOUCHDOWNS_29_YARDS_COLUMN,
                                PLAYER_PASSING_TOUCHDOWNS_39_YARDS_COLUMN,
                                PLAYER_PASSING_TOUCHDOWNS_49_YARDS_COLUMN,
                                PLAYER_PASSING_TOUCHDOWNS_ABOVE_50_YARDS_COLUMN,
                                PLAYER_RECEIVING_TOUCHDOWNS_9_YARDS_COLUMN,
                                PLAYER_RECEIVING_TOUCHDOWNS_19_YARDS_COLUMN,
                                PLAYER_RECEIVING_TOUCHDOWNS_29_YARDS_COLUMN,
                                PLAYER_RECEIVING_TOUCHDOWNS_39_YARDS_COLUMN,
                                PLAYER_RECEIVING_TOUCHDOWNS_49_YARDS_COLUMN,
                                PLAYER_RECEIVING_TOUCHDOWNS_ABOVE_50_YARDS_COLUMN,
                                PLAYER_RUSHING_TOUCHDOWNS_9_YARDS_COLUMN,
                                PLAYER_RUSHING_TOUCHDOWNS_19_YARDS_COLUMN,
                                PLAYER_RUSHING_TOUCHDOWNS_29_YARDS_COLUMN,
                                PLAYER_RUSHING_TOUCHDOWNS_39_YARDS_COLUMN,
                                PLAYER_RUSHING_TOUCHDOWNS_49_YARDS_COLUMN,
                                PLAYER_RUSHING_TOUCHDOWNS_ABOVE_50_YARDS_COLUMN,
                                PLAYER_PENALTIES_IN_MINUTES_COLUMN,
                                PLAYER_EVEN_STRENGTH_GOALS_COLUMN,
                                PLAYER_POWER_PLAY_GOALS_COLUMN,
                                PLAYER_SHORT_HANDED_GOALS_COLUMN,
                                PLAYER_GAME_WINNING_GOALS_COLUMN,
                                PLAYER_EVEN_STRENGTH_ASSISTS_COLUMN,
                                PLAYER_POWER_PLAY_ASSISTS_COLUMN,
                                PLAYER_SHORT_HANDED_ASSISTS_COLUMN,
                                PLAYER_SHOTS_ON_GOAL_COLUMN,
                                PLAYER_SHOOTING_PERCENTAGE_COLUMN,
                                PLAYER_SHIFTS_COLUMN,
                                PLAYER_TIME_ON_ICE_COLUMN,
                                PLAYER_DECISION_COLUMN,
                                PLAYER_GOALS_AGAINST_COLUMN,
                                PLAYER_SHOTS_AGAINST_COLUMN,
                                PLAYER_SAVES_COLUMN,
                                PLAYER_SAVE_PERCENTAGE_COLUMN,
                                PLAYER_SHUTOUTS_COLUMN,
                                PLAYER_INDIVIDUAL_CORSI_FOR_EVENTS_COLUMN,
                                PLAYER_ON_SHOT_ICE_FOR_EVENTS_COLUMN,
                                PLAYER_ON_SHOT_ICE_AGAINST_EVENTS_COLUMN,
                                PLAYER_CORSI_FOR_PERCENTAGE_COLUMN,
                                PLAYER_RELATIVE_CORSI_FOR_PERCENTAGE_COLUMN,
                                PLAYER_OFFENSIVE_ZONE_STARTS_COLUMN,
                                PLAYER_DEFENSIVE_ZONE_STARTS_COLUMN,
                                PLAYER_OFFENSIVE_ZONE_START_PERCENTAGE_COLUMN,
                                PLAYER_HITS_COLUMN,
                                PLAYER_TRUE_SHOOTING_PERCENTAGE_COLUMN,
                            ]
                        ],
                        player_column_prefix(i, x),
                        points_column=team_points_column(i),
                        field_goals_column=DELIMITER.join(
                            [player_column_prefix(i, x), PLAYER_FIELD_GOALS_COLUMN]
                        ),
                        assists_column=DELIMITER.join(
                            [player_column_prefix(i, x), PLAYER_ASSISTS_COLUMN]
                        ),
                        field_goals_attempted_column=DELIMITER.join(
                            [
                                player_column_prefix(i, x),
                                PLAYER_FIELD_GOALS_ATTEMPTED_COLUMN,
                            ]
                        ),
                        offensive_rebounds_column=DELIMITER.join(
                            [
                                player_column_prefix(i, x),
                                PLAYER_OFFENSIVE_REBOUNDS_COLUMN,
                            ]
                        ),
                        turnovers_column=DELIMITER.join(
                            [player_column_prefix(i, x), PLAYER_TURNOVERS_COLUMN]
                        ),
                        team_identifier_column=team_identifier_column(i),
                        birth_date_column=DELIMITER.join(
                            [
                                player_column_prefix(i, x),
                                PLAYER_BIRTH_DATE_COLUMN,
                            ]
                        ),
                        image_columns=[PLAYER_HEADSHOT_COLUMN],
                    )
                    for x in range(player_count)
                ]
            )
            for player_id in range(player_count):
                datetime_columns.add(
                    DELIMITER.join(
                        [player_column_prefix(i, player_id), PLAYER_BIRTH_DATE_COLUMN]
                    )
                )
            coach_count = find_coach_count(df, i)
            identifiers.extend(
                [
                    Identifier(
                        entity_type=EntityType.COACH,
                        column=coach_identifier_column(i, x),
                        feature_columns=[],
                        column_prefix=coach_column_prefix(i, x),
                        points_column=team_points_column(i),
                        team_identifier_column=team_identifier_column(i),
                    )
                    for x in range(coach_count)
                ]
            )
        df_processed = process(
            df,
            GAME_DT_COLUMN,
            identifiers,
            [None]
            + [datetime.timedelta(days=365 * i) for i in [1, 2, 4, 8]]
            + [datetime.timedelta(days=i * 7) for i in [2, 4]],
            df.attrs[str(FieldType.CATEGORICAL)],
            use_bets_features=False,
            use_news_features=True,
            datetime_columns=datetime_columns,
            use_players_feature=True,
        )
        df_processed.to_parquet(df_cache_path)
        return df_processed

    def _calculate_embedding_columns(self, df: pd.DataFrame) -> list[list[str]]:
        team_count = find_team_count(df)

        embedding_cols = []
        for i in range(team_count):
            col_prefix = team_column_prefix(i)
            embedding_cols.append(
                [
                    x
                    for x in df.columns.values.tolist()
                    if x.startswith(col_prefix) and is_embedding_column(x)
                ]
            )

        return embedding_cols
