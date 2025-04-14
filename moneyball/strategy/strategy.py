"""The strategy class."""

# pylint: disable=too-many-statements
import datetime
import multiprocessing
import os
import pickle

import empyrical  # type: ignore
import numpy as np
import optuna
import pandas as pd
import pytz
import wavetrainer as wt  # type: ignore
from joblib import parallel_backend  # type: ignore
from sportsball.data.field_type import FieldType  # type: ignore
from sportsball.data.game_model import GAME_DT_COLUMN  # type: ignore
from sportsball.data.game_model import VENUE_COLUMN_PREFIX
from sportsball.data.league_model import DELIMITER  # type: ignore
from sportsball.data.player_model import \
    ASSISTS_COLUMN as PLAYER_ASSISTS_COLUMN  # type: ignore
from sportsball.data.player_model import \
    FIELD_GOALS_ATTEMPTED_COLUMN as \
    PLAYER_FIELD_GOALS_ATTEMPTED_COLUMN  # type: ignore
from sportsball.data.player_model import \
    FIELD_GOALS_COLUMN as PLAYER_FIELD_GOALS_COLUMN  # type: ignore
from sportsball.data.player_model import \
    OFFENSIVE_REBOUNDS_COLUMN as PLAYER_OFFENSIVE_REBOUNDS_COLUMN
from sportsball.data.player_model import (  # type: ignore
    PLAYER_FUMBLES_COLUMN, PLAYER_FUMBLES_LOST_COLUMN, PLAYER_KICKS_COLUMN)
from sportsball.data.player_model import \
    TURNOVERS_COLUMN as PLAYER_TURNOVERS_COLUMN  # type: ignore
from sportsball.data.team_model import ASSISTS_COLUMN  # type: ignore
from sportsball.data.team_model import (FIELD_GOALS_ATTEMPTED_COLUMN,
                                        FIELD_GOALS_COLUMN,
                                        OFFENSIVE_REBOUNDS_COLUMN,
                                        TURNOVERS_COLUMN)
from sportsfeatures.entity_type import EntityType  # type: ignore
from sportsfeatures.identifier import Identifier  # type: ignore
from sportsfeatures.process import process  # type: ignore

from .features.columns import (find_player_count, find_team_count,
                               player_column_prefix, player_identifier_column,
                               team_column_prefix, team_identifier_column,
                               team_points_column, venue_identifier_column)

HOME_WIN_COLUMN = "home_win"

_KELLY_SAMPLER_FILENAME = "kelly_sampler.pkl"
_DF_FILENAME = "df.parquet.gzip"


class Strategy:
    """The strategy class."""

    # pylint: disable=too-many-locals,too-many-instance-attributes

    _returns: pd.Series | None

    def __init__(self, name: str, use_sports_feature: bool = False) -> None:
        self._df = None
        self._name = name
        self._use_sports_features = use_sports_feature
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
        return self._df

    @df.setter
    def df(self, df: pd.DataFrame) -> None:
        """Set the dataframe."""
        self._df = df
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
        return self._wt.transform(x_df)

    def returns(self) -> pd.Series:
        """Render the returns of the strategy."""
        main_df = self.df
        if main_df is None:
            raise ValueError("main_df is null.")

        returns = self._returns
        if returns is None:
            df = self.predict()
            dt_column = DELIMITER.join([GAME_DT_COLUMN])
            points_cols = main_df.attrs[str(FieldType.POINTS)]
            prob_col = "_".join(
                [HOME_WIN_COLUMN, wt.model.model.PROBABILITY_COLUMN_PREFIX]  # type: ignore
            )

            def calculate_returns(kelly_ratio: float) -> pd.Series:
                index = []
                data = []
                for date, group in df.groupby([df[dt_column].dt.date]):
                    date = date[0]
                    index.append(date)

                    # Find the kelly criterion for each bet
                    fs = []
                    for _, row in group.iterrows():
                        row_df = row.to_frame().T
                        row_df = row_df[
                            [x for x in row_df.columns.values if x.startswith(prob_col)]
                        ]
                        if row_df.isnull().values.any():
                            continue
                        arr = row_df.to_numpy().flatten()
                        team_idx = np.argmax(arr)
                        prob = arr[team_idx]
                        odds = row[f"teams/{team_idx}_odds"]
                        bet_prob = 1.0 / odds
                        f = max(prob - ((1.0 - prob) / bet_prob), 0.0) * kelly_ratio
                        fs.append(f)

                    # Make sure we aren't overallocating our capital
                    fs_sum = sum(fs)
                    if fs_sum > 1.0:
                        fs = [x / fs_sum for x in fs]

                    # Simulate the bets
                    bet_idx = 0
                    pl = 0.0
                    for _, row in group.iterrows():
                        row_df = row.to_frame().T
                        points_df = row_df[points_cols]
                        odds_df = row_df[main_df.attrs[str(FieldType.ODDS)]]
                        row_df = row_df[
                            [x for x in row_df.columns.values if x.startswith(prob_col)]
                        ]
                        if row_df.isnull().values.any():
                            continue
                        arr = row_df.to_numpy().flatten()
                        team_idx = np.argmax(arr)
                        win_team_idx = np.argmax(points_df.to_numpy().flatten())
                        odds = list(
                            odds_df[main_df.attrs[str(FieldType.ODDS)][team_idx]].values
                        )[0]
                        if team_idx == win_team_idx:
                            pl += odds * fs[bet_idx]
                        else:
                            pl -= fs[bet_idx]
                        bet_idx += 1

                    data.append(pl)

                return pd.Series(index=index, data=data, name=self._name)

            def objective(trial: optuna.Trial) -> float:
                ret = calculate_returns(trial.suggest_float("kelly_ratio", 0.0, 2.0))
                if abs(empyrical.max_drawdown(ret)) >= 1.0:
                    return 0.0
                return empyrical.calmar_ratio(ret)  # type: ignore

            with parallel_backend("multiprocessing"):
                self._kelly_study.optimize(
                    objective,
                    n_trials=100,
                    show_progress_bar=True,
                    n_jobs=multiprocessing.cpu_count(),
                )

            returns = calculate_returns(self.kelly_ratio)
            self._returns = returns
        return returns

    def next(self) -> pd.DataFrame:
        """Find the next predictions for betting."""
        dt_column = DELIMITER.join([GAME_DT_COLUMN])
        df = self.predict()
        start_dt = datetime.datetime.now()
        end_dt = start_dt + datetime.timedelta(days=1.0)
        df = df[df[dt_column] > start_dt]
        df = df[df[dt_column] <= end_dt]
        return df

    def _process(self) -> pd.DataFrame:
        df = self.df
        if df is None:
            raise ValueError("df is null")
        team_count = find_team_count(df)

        identifiers = [
            Identifier(
                EntityType.VENUE,
                venue_identifier_column(),
                [],
                VENUE_COLUMN_PREFIX,
            )
        ]
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
                            "kicks",
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
                )
            )
            player_count = find_player_count(df, i)
            identifiers.extend(
                [
                    Identifier(
                        EntityType.PLAYER,
                        player_identifier_column(i, x),
                        [
                            DELIMITER.join([player_column_prefix(i, x), y])
                            for y in [
                                PLAYER_KICKS_COLUMN,
                                PLAYER_FUMBLES_COLUMN,
                                PLAYER_FUMBLES_LOST_COLUMN,
                                PLAYER_FIELD_GOALS_COLUMN,
                                PLAYER_FIELD_GOALS_ATTEMPTED_COLUMN,
                                PLAYER_OFFENSIVE_REBOUNDS_COLUMN,
                                PLAYER_ASSISTS_COLUMN,
                                PLAYER_TURNOVERS_COLUMN,
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
        return process(
            df,
            GAME_DT_COLUMN,
            identifiers,
            [None] + [datetime.timedelta(days=365 * i) for i in [1, 2, 4, 8]],
        )
