"""The vennabers class for a trainer."""

import os
from typing import Any

import joblib  # type: ignore
import pandas as pd
from venn_abers import VennAbers  # type: ignore

from .output_column import output_prob_column
from .trainer import Trainer

_MODEL_FILENAME = "va.sav"


class VennAbersTrainer(Trainer):
    """The catboost trainer class."""

    def __init__(self, folder: str, wrapped_trainer: Trainer) -> None:
        super().__init__(folder)
        self._wrapped_trainer = wrapped_trainer
        self._model = VennAbers()

    @property
    def clf(self) -> Any:
        """The underlying classifier"""
        return self._model

    @property
    def salt(self) -> str:
        """The salt to use when hashing the predictions."""
        return "vennabers-" + self._wrapped_trainer.salt

    def fit(
        self,
        x: tuple[pd.DataFrame, pd.DataFrame | None],
        y: tuple[pd.DataFrame, pd.DataFrame | None],
    ):
        """Fit the data."""
        self._wrapped_trainer.fit(x, y)
        y_pred = self._wrapped_trainer.predict_proba(x[0])
        if y_pred is None:
            raise ValueError("y_pred is null")
        self._model.fit(y_pred.to_numpy(), y[0].to_numpy())

    def save(self):
        """Save the trainer."""
        self._wrapped_trainer.save()
        joblib.dump(self._model, os.path.join(self._folder, _MODEL_FILENAME))

    def load(self):
        """Load the trainer."""
        self._wrapped_trainer.load()
        self._model = joblib.load(os.path.join(self._folder, _MODEL_FILENAME))

    def predict(self, x: pd.DataFrame) -> pd.DataFrame | None:
        """Predict the Y values."""
        return self._wrapped_trainer.predict(x)

    def predict_proba(self, x: pd.DataFrame) -> pd.DataFrame | None:
        """Predict the Y probabilities."""
        y = super().predict(x)
        if y is not None:
            return y

        y_prob = self._wrapped_trainer.predict_proba(x)
        if y_prob is None:
            raise ValueError("y_prob is null")

        p_prime, _ = self._model.predict_proba(y_prob.to_numpy())
        y = pd.DataFrame(
            index=x.index,
            data={
                output_prob_column(i): p_prime[:, i]
                for i in range(len(y_prob.columns.values))
            },
        )

        self.save_prediction_proba(x, y)
        return y

    def select_features(
        self,
        x: tuple[pd.DataFrame, pd.DataFrame | None],
        y: tuple[pd.DataFrame, pd.DataFrame | None],
    ) -> tuple[list[str], int]:
        """Select the features from the training data."""
        return self._wrapped_trainer.select_features(x, y)
