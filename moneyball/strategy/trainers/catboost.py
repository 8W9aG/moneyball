"""The catboost class for a trainer."""

import json
import os
from typing import Any

import optuna
import pandas as pd
import torch
from catboost import CatBoostClassifier  # type: ignore
from catboost import EFeaturesSelectionAlgorithm, EShapCalcType, Pool

# from ...data.columns import GOLDEN_FEATURES_COLUMNS_ATTR
from ..weights import WEIGHTS, CombinedWeight
from .output_column import OUTPUT_COLUMN, output_prob_column
from .trainer import FEATURES_USR_ATTR, HASH_USR_ATTR, Trainer

# from optuna.integration import CatBoostPruningCallback  # type: ignore


_MODEL_FILENAME = "model.cbm"
_USR_ATTR_FILENAME = "usr_attr.json"
# _BORDERS_TSV_FILENAME = "borders.tsv"
_BOOTSTRAP_TYPE_BAYESIAN = "Bayesian"
_BOOTSTRAP_TYPE_BERNOULLI = "Bernoulli"
_OBJECTIVE_LOGLOSS = "Logloss"


def _sanitise_features(x: pd.DataFrame, text_features: list[str]) -> pd.DataFrame:
    # Remove all datetime columns
    x = x[
        [
            column
            for column in x.columns
            if not pd.api.types.is_datetime64_any_dtype(x[column])
        ]
    ]
    x = x[[column for column in x.columns if column not in text_features]]
    return x


class CatboostTrainer(Trainer):
    """The catboost trainer class."""

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        folder: str,
        categorical_features: list[str],
        text_features: list[str],
        trial: optuna.trial.Trial | optuna.trial.FrozenTrial | None = None,
    ) -> None:
        # pylint: disable=too-many-locals
        super().__init__(folder, trial=trial)
        self._categorical_features = categorical_features
        self._text_features = text_features
        self._golden_feature_border_count = (
            None
            if trial is None
            else trial.suggest_categorical(
                "golden_feature_border_count", [16, 32, 64, 1028]
            )
        )
        if trial is None:
            self._features_ratio = 0.0
            self._steps = 0
            self._model = CatBoostClassifier(
                task_type=None if not torch.cuda.is_available() else "GPU",
                devices=None if not torch.cuda.is_available() else "0",
            )

        else:
            print("Catboost Trial:")
            print(f"Golden Features Border Count: {self._golden_feature_border_count}")
            self._features_ratio = trial.suggest_float("features_ratio", 0.1, 0.9)
            print(f"Features Ratio: {self._features_ratio}")
            self._steps = trial.suggest_int("steps", 1, 10)
            print(f"Steps: {self._steps}")
            self._weight = CombinedWeight(
                trial.suggest_categorical("weight", list(WEIGHTS.keys()))
            )
            print(f"Weight: {self._weight.weight_name}")
            self._usr_attrs = trial.user_attrs
            bootstrap_type = trial.suggest_categorical(
                "bootstrap_type",
                [_BOOTSTRAP_TYPE_BAYESIAN, _BOOTSTRAP_TYPE_BERNOULLI, "MVS"],
            )
            print(f"Bootstrap Type: {bootstrap_type}")
            bagging_temperature = None
            subsample = None
            if bootstrap_type == _BOOTSTRAP_TYPE_BAYESIAN:
                bagging_temperature = trial.suggest_float(
                    "bagging_temperature", 0.0, 10.0
                )
            elif bootstrap_type == _BOOTSTRAP_TYPE_BERNOULLI:
                subsample = trial.suggest_float("subsample", 0.1, 1.0)
            print(f"Bagging Temperature: {bagging_temperature}")
            print(f"Subsample: {subsample}")
            objective = trial.suggest_categorical(
                "objective", [_OBJECTIVE_LOGLOSS, "CrossEntropy"]
            )
            print(f"Objective: {objective}")
            random_strength = None
            if objective == _OBJECTIVE_LOGLOSS:
                random_strength = trial.suggest_uniform("random_strength", 0.5, 5.0)
            print(f"Random Strength: {random_strength}")
            iterations = trial.user_attrs.get(
                "ITERATIONS", trial.suggest_int("iterations", 100, 10000)
            )
            print(f"Iterations: {iterations}")
            learning_rate = trial.suggest_float("learning_rate", 0.001, 0.3)
            print(f"Learning Rate: {learning_rate}")
            depth = trial.suggest_int("depth", 1, 12)
            print(f"Depth: {depth}")
            l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 3.0, 50.0)
            print(f"L2 Leaf Reg: {l2_leaf_reg}")
            task_type = None if not torch.cuda.is_available() else "GPU"
            print(f"Task Type: {task_type}")
            devices = None if not torch.cuda.is_available() else "0"
            print(f"Devices: {devices}")
            boosting_type = trial.suggest_categorical(
                "boosting_type", ["Ordered", "Plain"]
            )
            print(f"Boosting Type: {boosting_type}")
            self._model = CatBoostClassifier(
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                l2_leaf_reg=l2_leaf_reg,
                early_stopping_rounds=100,
                # task_type=task_type,
                # devices=devices,
                objective=objective,
                # colsample_bylevel=trial.suggest_float("colsample_bylevel", 0.01, 0.1),
                boosting_type=boosting_type,
                bootstrap_type=bootstrap_type,
                bagging_temperature=bagging_temperature,
                subsample=subsample,
                random_strength=random_strength,
            )

    @property
    def clf(self) -> Any:
        """The underlying classifier"""
        return self._model

    @property
    def salt(self) -> str:
        """The salt to use when hashing the predictions."""
        return "catboost-" + self._usr_attrs[HASH_USR_ATTR]

    def fit(
        self,
        x: tuple[pd.DataFrame, pd.DataFrame | None],
        y: tuple[pd.DataFrame, pd.DataFrame | None],
    ):
        """Fit the data."""
        x_train, x_test = x
        y_train, y_test = y
        train_pool = self._create_pool(x_train, y_train, True)  # type: ignore
        eval_pool = None
        if x_test is not None and y_test is not None:
            eval_pool = self._create_pool(x_test, y_test, True)  # type: ignore
        # callbacks = []
        # if self._trial is not None:
        #    callbacks.append(CatBoostPruningCallback(self._trial, "Accuracy"))  # type: ignore
        self._model.fit(
            train_pool,
            eval_set=eval_pool,
            early_stopping_rounds=100,
            # callbacks=callbacks,
        )
        # if callbacks:
        #    callbacks[0].check_pruned()
        feature_importances = self._model.get_feature_importance(prettified=True)
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(feature_importances)

    def save(self):
        """Save the trainer."""
        self._model.save_model(os.path.join(self._folder, _MODEL_FILENAME))
        with open(
            os.path.join(self._folder, _USR_ATTR_FILENAME), "w", encoding="utf8"
        ) as handle:
            json.dump(self._usr_attrs, handle)

    def load(self):
        """Load the trainer."""
        self._model.load_model(os.path.join(self._folder, _MODEL_FILENAME))
        with open(
            os.path.join(self._folder, _USR_ATTR_FILENAME), encoding="utf8"
        ) as handle:
            self._usr_attrs = json.load(handle)

    def predict(self, x: pd.DataFrame) -> pd.DataFrame | None:
        """Predict the Y values."""
        y = super().predict(x)
        if y is not None:
            return y
        train_pool = self._create_pool(x, None, True)
        y = pd.DataFrame(
            index=x.index, data={OUTPUT_COLUMN: self._model.predict(train_pool)}
        )
        self.save_prediction(x, y)
        return y

    def predict_proba(self, x: pd.DataFrame) -> pd.DataFrame | None:
        """Predict the Y probabilities."""
        y = super().predict_proba(x)
        if y is not None:
            return y
        pool = self._create_pool(x, None, True)
        proba = self._model.predict_proba(pool)
        y = pd.DataFrame(
            index=x.index,
            data={output_prob_column(i): proba[:, i] for i in range(proba.shape[1])},
        )
        self.save_prediction_proba(x, y)
        return y

    def select_features(
        self,
        x: tuple[pd.DataFrame, pd.DataFrame | None],
        y: tuple[pd.DataFrame, pd.DataFrame | None],
    ) -> tuple[list[str], int]:
        """Select the features from the training data."""
        x_train, x_test = x
        y_train, y_test = y
        x_train = _sanitise_features(x_train, self._text_features)
        y_train = _sanitise_features(y_train, self._text_features)
        train_pool = self._create_pool(x_train, y_train, False)  # type: ignore
        eval_pool = (
            None
            if x_test is None or y_test is None
            else self._create_pool(x_test, y_test, False)
        )  # type: ignore
        summary = self._model.select_features(
            train_pool,
            num_features_to_select=int(
                self._features_ratio * len(x_train.columns.values)
            ),
            steps=self._steps,
            train_final_model=True,
            features_for_select=x_train.columns.values,
            eval_set=eval_pool,
            algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
            shap_calc_type=EShapCalcType.Regular,
        )
        return summary["selected_features_names"], self._model.get_best_iteration()

    def _create_pool(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame | None,
        enforce_model_features: bool = False,
    ) -> Pool:
        # pylint: disable=pointless-string-statement
        text_features = list(set(x.columns.values) & set(self._text_features))
        x[text_features] = x[text_features].fillna("").astype(str)
        cat_features = list(set(x.columns.values) & set(self._categorical_features))
        x[cat_features] = x[cat_features].fillna(0).astype(int)
        if enforce_model_features:
            valid_cols = [col for col in self._feature_names if col in x.columns.values]
            x = x[valid_cols]
            text_features = list(set(text_features) & set(valid_cols))
            cat_features = list(set(cat_features) & set(valid_cols))
        x = _sanitise_features(x, self._text_features)
        weight = None
        if y is not None:
            weight = self._weight.process(y)
        pool = Pool(
            x,
            label=y,
            cat_features=cat_features,
            # text_features=text_features,
            weight=weight,
        )
        """
        if self._golden_feature_border_count is not None:
            # pylint: disable=line-too-long
            pool.quantize(
                per_float_feature_quantization=[
                    f"{x.columns.values.tolist().index(feature)}:border_count={self._golden_feature_border_count}"
                    for feature in x.attrs.get(GOLDEN_FEATURES_COLUMNS_ATTR, [])
                    if feature in x.columns.values
                ],
                task_type=None if not torch.cuda.is_available() else "GPU",
            )
            pool.save_quantization_borders(
                os.path.join(self._folder, _BORDERS_TSV_FILENAME)
            )
        else:
            pool.quantize(
                input_borders=os.path.join(self._folder, _BORDERS_TSV_FILENAME),
                task_type=None if not torch.cuda.is_available() else "GPU",
            )
        """
        return pool

    @property
    def _feature_names(self) -> list[str]:
        feature_names = self._model.feature_names_
        if feature_names is None:
            feature_names = self._usr_attrs[FEATURES_USR_ATTR]
        return feature_names
