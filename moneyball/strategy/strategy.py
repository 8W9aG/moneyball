"""The strategy class."""

# pylint: disable=too-many-statements
import datetime
import hashlib
import os
import pickle

import numpy as np
import optuna
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
from sportsball.data.odds_model import (DT_COLUMN, ODDS_BOOKIE_COLUMN,
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
from sportsball.data.player_model import (PLAYER_DISPOSALS_COLUMN,
                                          PLAYER_FUMBLES_COLUMN,
                                          PLAYER_FUMBLES_LOST_COLUMN,
                                          PLAYER_GOALS_COLUMN,
                                          PLAYER_HANDBALLS_COLUMN,
                                          PLAYER_KICKS_COLUMN,
                                          PLAYER_MARKS_COLUMN)
from sportsball.data.player_model import \
    TURNOVERS_COLUMN as PLAYER_TURNOVERS_COLUMN  # type: ignore
from sportsball.data.team_model import ASSISTS_COLUMN  # type: ignore
from sportsball.data.team_model import (FIELD_GOALS_ATTEMPTED_COLUMN,
                                        FIELD_GOALS_COLUMN, KICKS_COLUMN,
                                        OFFENSIVE_REBOUNDS_COLUMN,
                                        TEAM_BEHINDS_COLUMN,
                                        TEAM_BOUNCES_COLUMN,
                                        TEAM_BROWNLOW_VOTES_COLUMN,
                                        TEAM_CLANGERS_COLUMN,
                                        TEAM_CLEARANCES_COLUMN,
                                        TEAM_CONTESTED_MARKS_COLUMN,
                                        TEAM_CONTESTED_POSSESSIONS_COLUMN,
                                        TEAM_DISPOSALS_COLUMN,
                                        TEAM_FREE_KICKS_AGAINST_COLUMN,
                                        TEAM_FREE_KICKS_FOR_COLUMN,
                                        TEAM_GOAL_ASSISTS_COLUMN,
                                        TEAM_GOALS_COLUMN,
                                        TEAM_HANDBALLS_COLUMN,
                                        TEAM_HIT_OUTS_COLUMN,
                                        TEAM_INSIDES_COLUMN, TEAM_MARKS_COLUMN,
                                        TEAM_MARKS_INSIDE_COLUMN,
                                        TEAM_ONE_PERCENTERS_COLUMN,
                                        TEAM_REBOUNDS_COLUMN,
                                        TEAM_TACKLES_COLUMN,
                                        TEAM_UNCONTESTED_POSSESSIONS_COLUMN,
                                        TURNOVERS_COLUMN)
from sportsball.data.venue_model import VENUE_ADDRESS_COLUMN
from sportsfeatures.bet import Bet
from sportsfeatures.entity_type import EntityType  # type: ignore
from sportsfeatures.identifier import Identifier  # type: ignore
from sportsfeatures.process import process  # type: ignore

from .features.columns import (find_odds_count, find_player_count,
                               find_team_count, odds_column_prefix,
                               odds_odds_column, player_column_prefix,
                               player_identifier_column, team_column_prefix,
                               team_identifier_column, team_points_column,
                               venue_identifier_column)
from .kelly_fractions import (augment_kelly_fractions, calculate_returns,
                              calculate_value)

HOME_WIN_COLUMN = "home_win"

_KELLY_SAMPLER_FILENAME = "kelly_sampler.pkl"
_DF_FILENAME = "df.parquet.gzip"


class Strategy:
    """The strategy class."""

    # pylint: disable=too-many-locals,too-many-instance-attributes

    _returns: pd.Series | None

    def __init__(self, name: str) -> None:
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
            validation_size=datetime.timedelta(days=365),
            max_train_timeout=datetime.timedelta(hours=12),
            cutoff_dt=datetime.datetime.now(tz=pytz.UTC),
            test_size=datetime.timedelta(days=365 * 2),
        )

        # Load kelly study
        kelly_storage_name = f"sqlite:///{name}/kelly_study.db"
        kelly_sampler_file = os.path.join(name, _KELLY_SAMPLER_FILENAME)
        kelly_restored_sampler = None
        if os.path.exists(kelly_sampler_file):
            with open(kelly_sampler_file, "rb") as handle:
                kelly_restored_sampler = pickle.load(handle)
        self._kelly_study = optuna.create_study(
            study_name=f"kelly_{name}",
            storage=kelly_storage_name,
            load_if_exists=True,
            sampler=kelly_restored_sampler,
            direction=optuna.study.StudyDirection.MAXIMIZE,
        )

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

    @property
    def name(self) -> str:
        """Fetch the name of the strategy."""
        return self._name

    @property
    def kelly_ratio(self) -> float:
        """Find the best kelly ratio for this strategy."""
        return self._kelly_study.best_trial.suggest_float("kelly_ratio", 0.0, 2.0)

    def fit(self):
        """Fits the strategy to the dataset by walking forward."""
        df = self.df
        if df is None:
            raise ValueError("df is null")
        training_cols = df.attrs[str(FieldType.POINTS)]
        x_df = self._process()
        y = df[training_cols]
        y[HOME_WIN_COLUMN] = np.argmax(y.to_numpy(), axis=1)
        x_df = x_df.drop(columns=training_cols)
        x_df = x_df.drop(columns=df.attrs[str(FieldType.LOOKAHEAD)], errors="ignore")
        self._wt.fit(x_df, y=y[HOME_WIN_COLUMN].astype(bool))

    def predict(self) -> pd.DataFrame:
        """Predict the results from walk-forward."""
        df = self.df
        if df is None:
            raise ValueError("df is null.")

        x_df = self._process()
        training_cols = df.attrs[str(FieldType.POINTS)]
        x_df = x_df.drop(columns=training_cols, errors="ignore")
        x_df = x_df.drop(columns=df.attrs[str(FieldType.LOOKAHEAD)], errors="ignore")
        x_df = self._wt.transform(x_df)
        for points_col in df.attrs[str(FieldType.POINTS)]:
            x_df[points_col] = df[points_col]
        return x_df

    def returns(self) -> pd.Series:
        """Render the returns of the strategy."""
        main_df = self.df
        if main_df is None:
            raise ValueError("main_df is null.")
        main_df = main_df.copy()

        returns = self._returns
        if returns is None:
            df = self.predict()
            points_cols = main_df.attrs[str(FieldType.POINTS)]
            df[points_cols] = main_df[points_cols].to_numpy()
            df = augment_kelly_fractions(df, len(points_cols), HOME_WIN_COLUMN)
            df.to_parquet(os.path.join(self._name, "returns_df.parquet.gzip"))

            def objective(trial: optuna.Trial) -> float:
                ret = calculate_returns(
                    trial.suggest_float("kelly_ratio", 0.0, 1.0), df.copy(), self._name
                )
                return calculate_value(ret)

            self._kelly_study.optimize(
                objective,
                n_trials=max(100 - len(self._kelly_study.trials), 1),
                show_progress_bar=True,
            )

            returns = calculate_returns(
                self._kelly_study.best_trial.suggest_float("kelly_ratio", 0.0, 1.0),
                df.copy(),
                self._name,
            )
            self._returns = returns
        return returns

    def next(self) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
        """Find the next predictions for betting."""
        dt_column = DELIMITER.join([GAME_DT_COLUMN])
        df = self.predict()
        start_dt = datetime.datetime.now(datetime.timezone.utc)
        end_dt = start_dt + datetime.timedelta(days=3.0)
        df = df[df[dt_column] > start_dt]
        df = df[df[dt_column] <= end_dt]
        return df, self._wt.feature_importances()

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
                        )
                        for x in range(odds_count)
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
                    )
                    for x in range(player_count)
                ]
            )
        df_processed = process(
            df,
            GAME_DT_COLUMN,
            identifiers,
            [None] + [datetime.timedelta(days=365 * i) for i in [1, 2, 4, 8]],
            df.attrs[str(FieldType.CATEGORICAL)],
            use_bets_features=False,
        )
        df_processed.to_parquet(df_cache_path)
        return df_processed
