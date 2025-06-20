"""A function for processing kelly fractions."""

# pylint: disable=too-many-locals
import datetime

import empyrical  # type: ignore
import numpy as np
import pandas as pd
import wavetrainer as wt  # type: ignore
from sportsball.data.game_model import GAME_DT_COLUMN

from .features.columns import find_team_count, team_points_column

KELLY_FRACTION_RATIO_COL_PREFIX = "kelly_fraction_ratio_"
KELLY_FRACTION_COL_PREFIX = "kelly_fraction_"
RETURN_MULTIPLIER_COL_PREFIX = "return_multiplier_"
BET_WON_COL_PREFIX = "bet_won_"
BET_ODDS_COL_PREFIX = "bet_odds_"
ADJUSTED_FRACTION_COL_PREFIX = "adjusted_fraction_"


def probability_columns(df: pd.DataFrame) -> list[str]:
    """Probability columns generated."""
    teams = find_team_count(df)
    points_cols = [team_points_column(x) for x in range(teams)]
    prob_cols = sorted(
        [
            x
            for x in df.columns.values.tolist()
            if x[:-1].endswith(wt.model.model.PROBABILITY_COLUMN_PREFIX)  # type: ignore
        ]
    )
    if len(prob_cols) > len(points_cols):
        prob_cols = [x for x in prob_cols if x.endswith("_1")]
    return prob_cols


def augment_kelly_fractions(df: pd.DataFrame, teams: int) -> pd.DataFrame:
    """Augment the dataframe with kelly fractions."""
    points_cols = [team_points_column(x) for x in range(teams)]
    prob_cols = probability_columns(df)

    odds_cols = [f"teams/{x}_odds" for x in range(teams)]
    df = df[df[GAME_DT_COLUMN].dt.year >= datetime.datetime.now().year - 1]

    probs = df[prob_cols].to_numpy()
    odds = df[odds_cols].to_numpy()
    points = df[points_cols].to_numpy()
    wins_idx = points.argmax(axis=1)
    for i in range(len(points_cols)):
        p = probs[np.arange(len(df)), i]
        o = odds[np.arange(len(df)), i]
        b = o - 1.0
        q = 1.0 - p
        kelly_fraction = (b * p - q) / b
        kelly_fraction = np.clip(kelly_fraction, 0, 1)
        df[KELLY_FRACTION_COL_PREFIX + str(i)] = kelly_fraction
        df[BET_WON_COL_PREFIX + str(i)] = i == wins_idx
        df[BET_ODDS_COL_PREFIX + str(i)] = o

    def scale_fractions(group):
        total = 0.0
        for i in range(len(points_cols)):
            total += group[KELLY_FRACTION_COL_PREFIX + str(i)].sum()
        if total > 1.0:
            scaling_factor = 1 / total
            for i in range(len(points_cols)):
                group[ADJUSTED_FRACTION_COL_PREFIX + str(i)] = (
                    group[KELLY_FRACTION_COL_PREFIX + str(i)] * scaling_factor
                )
        else:
            for i in range(len(points_cols)):
                group[ADJUSTED_FRACTION_COL_PREFIX + str(i)] = group[
                    KELLY_FRACTION_COL_PREFIX + str(i)
                ]
        return group

    # Check if the dt column is somehow in an index
    if GAME_DT_COLUMN in df.index.names:
        dt_series = df[GAME_DT_COLUMN].copy()
        df = df.drop(columns=GAME_DT_COLUMN)
        df = df.reset_index(level=GAME_DT_COLUMN)
        df[GAME_DT_COLUMN] = dt_series.tolist()

    df = df.groupby(df[GAME_DT_COLUMN].dt.date).apply(scale_fractions)  # type: ignore
    df[GAME_DT_COLUMN] = df[GAME_DT_COLUMN].dt.date
    df = df.set_index(GAME_DT_COLUMN)
    return df


def calculate_returns(kelly_ratio: float, df: pd.DataFrame, name: str) -> pd.Series:
    """Calculate the returns with a kelly ratio."""
    i = 0
    while True:
        adjusted_fraction_col = ADJUSTED_FRACTION_COL_PREFIX + str(i)
        kelly_fraction_ratio_col = KELLY_FRACTION_RATIO_COL_PREFIX + str(i)
        if adjusted_fraction_col not in df.columns.values.tolist():
            break
        df[kelly_fraction_ratio_col] = df[adjusted_fraction_col] * kelly_ratio
        df[RETURN_MULTIPLIER_COL_PREFIX + str(i)] = (
            np.where(
                df[BET_WON_COL_PREFIX + str(i)],
                1
                + df[kelly_fraction_ratio_col] * (df[BET_ODDS_COL_PREFIX + str(i)] - 1),
                1 - df[kelly_fraction_ratio_col],
            )
            - 1.0
        )
        i += 1

    # Convert net return to multiplier
    df["return_with_base"] = (
        df[[RETURN_MULTIPLIER_COL_PREFIX + str(x) for x in range(i)]].sum(axis=1) + 1.0
    )

    # Aggregate per day by multiplying
    daily_return = df.groupby(df.index)["return_with_base"].prod() - 1.0

    return daily_return.rename(name)


def calculate_value(ret: pd.Series) -> float:
    """Calculates the value of the returns."""
    if abs(empyrical.max_drawdown(ret)) >= 1.0:
        return 0.0
    return empyrical.calmar_ratio(ret)  # type: ignore
