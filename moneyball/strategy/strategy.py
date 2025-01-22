"""The strategy class."""

import datetime
import os
import pickle
import statistics
import uuid

import empyrical  # type: ignore
import numpy as np
import optuna
import pandas as pd
import pytz
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from sklearn.metrics import precision_score  # type: ignore
from sklearn.metrics import accuracy_score, recall_score
from sportsball.data.field_type import FieldType  # type: ignore
from sportsball.data.game_model import (GAME_DT_COLUMN,  # type: ignore
                                        GAME_WEEK_COLUMN)
from sportsball.data.league_model import DELIMITER  # type: ignore

from .features import CombinedFeature
from .reducers import CombinedReducer
from .trainers import (FEATURES_USR_ATTR, HASH_USR_ATTR, CatboostTrainer,
                       VennAbersTrainer)
from .trainers.output_column import OUTPUT_COLUMN, OUTPUT_PROB_COLUMN_PREFIX

_SAMPLER_FILENAME = "sampler.pkl"


def _next_week_dt(
    current_dt: datetime.datetime | None, df: pd.DataFrame
) -> datetime.datetime | None:
    if df.empty:
        return None
    week_column = DELIMITER.join([GAME_WEEK_COLUMN])
    dt_column = DELIMITER.join([GAME_DT_COLUMN])
    if current_dt is not None:
        df = df[df[dt_column] >= current_dt]
    current_week = df.iloc[0][week_column]
    for _, row in df.iterrows():
        week = row[week_column]
        if current_week != week:
            return pd.to_datetime(row[dt_column]).to_pydatetime()
    try:
        return pd.to_datetime(df[dt_column]).to_pydatetime()[-1]  # type: ignore
    except AttributeError:
        return None


def _print_metrics(y_test: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    return accuracy


class Strategy:
    """The strategy class."""

    # pylint: disable=too-many-locals

    _returns: pd.Series | None

    def __init__(self, df: pd.DataFrame, name: str) -> None:
        self._df = df
        self._name = name
        self._features = CombinedFeature()
        self._reducers = CombinedReducer(
            [
                DELIMITER.join([GAME_WEEK_COLUMN]),
                DELIMITER.join([GAME_DT_COLUMN]),
            ]
        )
        os.makedirs(name, exist_ok=True)
        storage_name = f"sqlite:///{name}/study.db"
        sampler_file = os.path.join(name, _SAMPLER_FILENAME)
        restored_sampler = None
        if os.path.exists(sampler_file):
            with open(sampler_file, "rb") as handle:
                restored_sampler = pickle.load(handle)
        self._study = optuna.create_study(
            study_name=name,
            storage=storage_name,
            load_if_exists=True,
            sampler=restored_sampler,
            direction=optuna.study.StudyDirection.MAXIMIZE,
        )
        self._returns = None

    def fit(self, start_dt: datetime.datetime | None = None):
        """Fits the strategy to the dataset by walking forward."""
        # pylint: disable=too-many-statements

        if start_dt is None:
            start_dt = max(
                self._df[[GAME_DT_COLUMN, x]]
                .dropna()[GAME_DT_COLUMN]
                .iloc[0]
                .to_pydatetime()
                for x in self._df.attrs[str(FieldType.ODDS)]
            )
            start_dt = max(
                start_dt,  # type: ignore
                pytz.utc.localize(datetime.datetime.now() - relativedelta(years=5)),
            )

        df = self._df.copy()
        df = df[
            df[GAME_DT_COLUMN]
            < pytz.utc.localize(datetime.datetime.now() - datetime.timedelta(days=1.0))
        ]
        training_cols = set(df.attrs[str(FieldType.POINTS)])
        x = self._features.process(df)
        x = self._reducers.process(x)
        y = df[list(training_cols)]
        y[OUTPUT_COLUMN] = np.argmax(y.to_numpy(), axis=1)
        if len(training_cols) == 2:
            y[OUTPUT_COLUMN] = y[OUTPUT_COLUMN].astype(bool)
        y = y[[OUTPUT_COLUMN]]

        # Walkforward by week
        predictions = []
        current_n_trials = max(32 - len(self._study.trials), 1)
        while True:
            start_dt = _next_week_dt(start_dt, x)
            if start_dt is None:
                break
            x_walk = x[x[GAME_DT_COLUMN] < start_dt]
            y_walk = y.iloc[: len(x_walk)]
            if len(x_walk) == len(x) or len(y_walk) == len(y):
                break
            folder = os.path.join(self._name, str(start_dt.date()))
            if os.path.exists(folder):
                continue
            os.makedirs(folder, exist_ok=True)
            print(f"Trainer {folder}")
            # next_dt = _next_week_dt(start_dt, x)
            x_test = x[x[GAME_DT_COLUMN] >= start_dt]
            y_test = y.iloc[len(x_walk) : len(x_walk) + len(x_test)]

            def objective(trial: optuna.Trial) -> float:
                trial.set_user_attr(HASH_USR_ATTR, str(uuid.uuid4()))
                trainer = VennAbersTrainer(
                    folder,
                    CatboostTrainer(
                        folder,
                        df.attrs[str(FieldType.CATEGORICAL)],
                        df.attrs[str(FieldType.TEXT)],
                        trial=trial,
                    ),
                )
                x_split, y_split = trainer.split_train_test(x_walk, y_walk)
                features, iterations = trainer.select_features(x_split, y_split)
                trial.set_user_attr(FEATURES_USR_ATTR, features)
                trial.set_user_attr("ITERATIONS", iterations)

                print("In Sample Metrics:")
                y_pred = trainer.predict(x_split[0])
                if y_pred is None:
                    raise ValueError("y_pred is null")
                _print_metrics(y_split[0], y_pred)

                print("Out of Sample Metrics:")
                y_pred = trainer.predict(x_test)
                if y_pred is None:
                    raise ValueError("y_pred is null")
                _print_metrics(y_test, y_pred)

                print("Test Metrics:")
                y_pred = trainer.predict(x_split[1])
                if y_pred is None:
                    raise ValueError("y_pred is null")
                return _print_metrics(y_split[1], y_pred)

            self._study.optimize(objective, n_trials=current_n_trials)
            current_n_trials = 1
            with open(os.path.join(self._name, _SAMPLER_FILENAME), "wb") as handle:
                pickle.dump(self._study.sampler, handle)
            best_trial = self._study.best_trial
            trainer = VennAbersTrainer(
                folder,
                CatboostTrainer(
                    folder,
                    df.attrs[str(FieldType.CATEGORICAL)],
                    df.attrs[str(FieldType.TEXT)],
                    trial=best_trial,
                ),
            )
            trainer.fit(
                (
                    x_walk[
                        [
                            x
                            for x in best_trial.user_attrs[FEATURES_USR_ATTR]
                            if x in x_walk.columns.values
                        ]
                    ],
                    None,
                ),
                (y_walk, None),
            )
            trainer.save()

            y_pred = trainer.predict(x_walk)
            if y_pred is None:
                raise ValueError("y_pred is null")
            print("Final In Sample Metrics:")
            _print_metrics(y_walk, y_pred)

            y_pred = trainer.predict(x_test)
            if y_pred is None:
                raise ValueError("y_pred is null")

            print("Final Out of Sample Metrics:")
            predictions.append(_print_metrics(y_test, y_pred))
        return statistics.mean(predictions)

    def predict(self, start_dt: datetime.datetime | None = None) -> pd.DataFrame:
        """Predict the results from walk-forward."""
        dt_column = DELIMITER.join([GAME_DT_COLUMN])

        if start_dt is None:
            start_dt = max(
                self._df[[dt_column, x]].dropna()[dt_column].iloc[0].to_pydatetime()
                for x in self._df.attrs[str(FieldType.ODDS)]
            )
            start_dt = max(
                start_dt,  # type: ignore
                pytz.utc.localize(datetime.datetime.now() - relativedelta(years=5)),
            )

        df = self._df.copy()
        df = df[
            df[dt_column]
            < pytz.utc.localize(datetime.datetime.now() - datetime.timedelta(days=1.0))
        ]
        x = self._features.process(df)

        for folder_name in sorted(os.listdir(self._name)):
            folder = os.path.join(self._name, folder_name)
            if not os.path.isdir(folder):
                continue
            start_dt = parse(folder_name)
            x_test = x[x[dt_column] >= start_dt]
            trainer = VennAbersTrainer(
                folder,
                CatboostTrainer(
                    folder,
                    self._df.attrs[str(FieldType.CATEGORICAL)],
                    self._df.attrs[str(FieldType.TEXT)],
                ),
            )
            trainer.load()
            y_prob = trainer.predict_proba(x_test)
            if y_prob is None:
                raise ValueError("y_prob is null")

            for column in y_prob.columns.values:
                if column not in x:
                    x[column] = None
                x_small = x[x[dt_column] >= start_dt]
                i_start = len(x) - len(x_small)
                i_end = i_start + len(x_small)
                x.iloc[i_start:i_end, x.columns.get_loc(column)] = list(  # type: ignore
                    y_prob[column].values
                )

        df = df.reset_index().drop(columns=["index"])
        for points_col in df.attrs[str(FieldType.POINTS)]:
            x[points_col] = df[points_col].values

        return x

    def returns(self) -> pd.Series:
        """Render the returns of the strategy."""
        returns = self._returns
        if returns is None:
            df = self.predict()
            dt_column = DELIMITER.join([GAME_DT_COLUMN])
            points_cols = self._df.attrs[str(FieldType.POINTS)]

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
                        odds_df = row_df[self._df.attrs[str(FieldType.ODDS)]]
                        row_df = row_df[
                            [
                                x
                                for x in row_df.columns.values
                                if x.startswith(OUTPUT_PROB_COLUMN_PREFIX)
                            ]
                        ]
                        if row_df.isnull().values.any():
                            continue
                        arr = row_df.to_numpy().flatten()
                        team_idx = np.argmax(arr)
                        prob = arr[team_idx]
                        odds = list(
                            odds_df[
                                self._df.attrs[str(FieldType.ODDS)][team_idx]
                            ].values
                        )[0]
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
                        odds_df = row_df[self._df.attrs[str(FieldType.ODDS)]]
                        row_df = row_df[
                            [
                                x
                                for x in row_df.columns.values
                                if x.startswith(OUTPUT_PROB_COLUMN_PREFIX)
                            ]
                        ]
                        if row_df.isnull().values.any():
                            continue
                        arr = row_df.to_numpy().flatten()
                        team_idx = np.argmax(arr)
                        win_team_idx = np.argmax(points_df.to_numpy().flatten())
                        odds = list(
                            odds_df[
                                self._df.attrs[str(FieldType.ODDS)][team_idx]
                            ].values
                        )[0]
                        if team_idx == win_team_idx:
                            pl += odds * fs[bet_idx]
                        else:
                            pl -= fs[bet_idx]
                        bet_idx += 1

                    data.append(pl)

                return pd.Series(index=index, data=data, name=self._name)

            study = optuna.create_study(
                study_name="kelly",
                direction=optuna.study.StudyDirection.MAXIMIZE,
            )

            def objective(trial: optuna.Trial) -> float:
                ret = calculate_returns(trial.suggest_float("kelly_ratio", 0.0, 2.0))
                if abs(empyrical.max_drawdown(ret)) >= 1.0:
                    return 0.0
                return empyrical.calmar_ratio(ret)  # type: ignore

            study.optimize(objective, n_trials=100, show_progress_bar=True)

            returns = calculate_returns(
                study.best_trial.suggest_float("kelly_ratio", 0.0, 2.0)
            )
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
