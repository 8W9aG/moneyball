"""The prototype class for a trainer."""

import hashlib
import os
from typing import Any

import optuna
import pandas as pd
from sklearn.model_selection import train_test_split  # type: ignore

HASH_USR_ATTR = "HASH"
FEATURES_USR_ATTR = "FEATURES"


def _hash_df(df: pd.DataFrame) -> int:
    return int(
        hashlib.sha256(
            pd.util.hash_pandas_object(df, index=True).values,  # type: ignore
        ).hexdigest(),
        16,
    )


class Trainer:
    """The prototype trainer class."""

    def __init__(
        self,
        folder: str,
        trial: optuna.trial.Trial | optuna.trial.FrozenTrial | None = None,
    ) -> None:
        self._folder = folder
        self._test_size = 0.2
        self._trial = trial
        if trial is not None:
            self._test_size = trial.suggest_float("test_size", 0.0, 0.5)

    @property
    def clf(self) -> Any:
        """The underlying classifier"""
        raise NotImplementedError("clf is not implemented in parent class.")

    @property
    def salt(self) -> str:
        """The salt to use when hashing the predictions."""
        raise NotImplementedError("salt is not implemented in parent class.")

    def fit(
        self,
        x: tuple[pd.DataFrame, pd.DataFrame | None],
        y: tuple[pd.DataFrame, pd.DataFrame | None],
    ):
        """Fit the data."""
        raise NotImplementedError("fit is not implemented in parent class.")

    def save(self):
        """Save the trainer."""
        raise NotImplementedError("save is not implemented in parent class.")

    def load(self):
        """Load the trainer."""
        raise NotImplementedError("load is not implemented in parent class.")

    def predict(self, x: pd.DataFrame) -> pd.DataFrame | None:
        """Predict the Y values."""
        filename = os.path.join(self._folder, f"{self.salt}_{_hash_df(x)}.parquet")
        if os.path.exists(filename):
            return pd.read_parquet(filename)
        return None

    def predict_proba(self, x: pd.DataFrame) -> pd.DataFrame | None:
        """Predict the Y probabilities."""
        filename = os.path.join(
            self._folder, f"{self.salt}_{_hash_df(x)}_proba.parquet"
        )
        if os.path.exists(filename):
            return pd.read_parquet(filename)
        return None

    def save_prediction(self, x: pd.DataFrame, y: pd.DataFrame):
        """Save the prediction for later."""
        filename = os.path.join(self._folder, f"{self.salt}_{_hash_df(x)}.parquet")
        y.to_parquet(filename)

    def save_prediction_proba(self, x: pd.DataFrame, y: pd.DataFrame):
        """Save the prediction probabilities for later."""
        filename = os.path.join(
            self._folder, f"{self.salt}_{_hash_df(x)}_proba.parquet"
        )
        y.to_parquet(filename)

    def select_features(
        self,
        x: tuple[pd.DataFrame, pd.DataFrame | None],
        y: tuple[pd.DataFrame, pd.DataFrame | None],
    ) -> tuple[list[str], int]:
        """Select the features from the training data."""
        raise NotImplementedError("select_features is not implemented in parent class.")

    def split_train_test(
        self, x: pd.DataFrame, y: pd.DataFrame
    ) -> tuple[tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]:
        """Splits the x/y data into train and test sets."""
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=self._test_size, shuffle=False
        )
        return (x_train, x_test), (y_train, y_test)  # type: ignore
