"""A function for processing kelly fractions."""

# pylint: disable=too-many-locals
import datetime
import math
import warnings

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

# Internal column to mark matches that ended in a draw (max points tie)
_MATCH_DRAW_COL = "__match_is_draw__"


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


def augment_kelly_fractions(df: pd.DataFrame, teams: int, eta: float) -> pd.DataFrame:
    """Augment the dataframe with kelly fractions."""
    points_cols = sorted([team_points_column(x) for x in range(teams)])
    prob_cols = sorted(probability_columns(df))

    odds_cols = sorted([f"teams/{x}_odds" for x in range(teams)])
    df = df[df[GAME_DT_COLUMN].dt.year >= datetime.datetime.now().year - 1]

    probs = df[prob_cols].to_numpy()
    odds = df[odds_cols].to_numpy()
    points = df[points_cols].to_numpy()

    # --- Draw mask: row is a draw if the top points value appears ≥ 2 times ---
    row_max = points.max(axis=1, keepdims=True)
    draw_mask = (np.isclose(points, row_max)).sum(axis=1) > 1
    df[_MATCH_DRAW_COL] = draw_mask

    wins_idx = points.argmax(axis=1)
    probs_idx = probs.argmax(axis=1)
    print(f"Accuracy: {float((wins_idx == probs_idx).sum()) / float(len(df))}")
    for i in range(len(points_cols)):
        orig_p = probs[np.arange(len(df)), i]
        p = (orig_p**eta) / ((orig_p**eta) + ((1 - orig_p) ** eta))
        o = np.clip(odds[np.arange(len(df)), i], 1.0, None)
        b = o - 1.0
        q = 1.0 - p
        kelly_fraction = (b * p - q) / b
        kelly_fraction = np.clip(kelly_fraction, 0, 1)
        if len(points_cols) == 2:
            kelly_fraction[orig_p < 0.5] = 0.0
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
    """Calculate the returns with a kelly ratio.
    Draws are treated as push (stake returned)."""
    warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    # Ensure draw mask exists (robust if df didn't come from augment_kelly_fractions)
    if _MATCH_DRAW_COL not in df.columns:
        teams = find_team_count(df)
        points_cols = sorted([team_points_column(x) for x in range(teams)])
        if all(col in df.columns for col in points_cols):
            pts = df[points_cols].to_numpy()
            row_max = pts.max(axis=1, keepdims=True)
            df[_MATCH_DRAW_COL] = (np.isclose(pts, row_max)).sum(axis=1) > 1
        else:
            # Fallback: assume no draws if we can't compute points
            df[_MATCH_DRAW_COL] = False

    i = 0
    while True:
        adjusted_fraction_col = ADJUSTED_FRACTION_COL_PREFIX + str(i)
        kelly_fraction_ratio_col = KELLY_FRACTION_RATIO_COL_PREFIX + str(i)
        if adjusted_fraction_col not in df.columns.values.tolist():
            break

        df[kelly_fraction_ratio_col] = df[adjusted_fraction_col] * kelly_ratio

        win_col = BET_WON_COL_PREFIX + str(i)
        odds_col = BET_ODDS_COL_PREFIX + str(i)
        ret_mult_col = RETURN_MULTIPLIER_COL_PREFIX + str(i)

        # Win → 1 + f*(odds-1)
        # Draw (push) → 1
        # Loss → 1 - f
        df[ret_mult_col] = (
            np.select(
                [
                    df[win_col].to_numpy(),
                    df[_MATCH_DRAW_COL].to_numpy(),
                ],
                [
                    1
                    + df[kelly_fraction_ratio_col].to_numpy()
                    * (df[odds_col].to_numpy() - 1),
                    1,
                ],
                default=(1 - df[kelly_fraction_ratio_col].to_numpy()),
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
    print(f"Sharpe: {empyrical.sharpe_ratio(ret, annualization=365)}")
    print(f"Calmar: {empyrical.calmar_ratio(ret, annualization=365)}")
    print(f"Max Drawdown: {empyrical.max_drawdown(ret)}")
    print(f"Sortino: {empyrical.sortino_ratio(ret, annualization=365)}")
    print(f"Return: {empyrical.annual_return(ret, annualization=365)}")
    if abs(empyrical.max_drawdown(ret)) >= 1.0:
        return 0.0
    calmar = float(empyrical.calmar_ratio(ret, annualization=365))  # type: ignore
    if math.isnan(calmar):
        calmar = float(empyrical.annual_return(ret, annualization=365))  # type: ignore
    return calmar
