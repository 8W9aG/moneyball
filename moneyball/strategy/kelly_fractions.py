"""A function for processing kelly fractions."""

# pylint: disable=too-many-locals
import datetime

import numpy as np
import pandas as pd
import wavetrainer as wt  # type: ignore
from sportsball.data.game_model import GAME_DT_COLUMN
from sportsfeatures.columns import DELIMITER

from .features.columns import team_points_column


def augment_kelly_fractions(
    df: pd.DataFrame, teams: int, predictand_column: str
) -> pd.DataFrame:
    """Augment the dataframe with kelly fractions."""
    points_cols = [team_points_column(x) for x in range(teams)]
    prob_col = DELIMITER.join(
        [predictand_column, wt.model.model.PROBABILITY_COLUMN_PREFIX]  # type: ignore
    )
    odds_cols = [f"teams/{x}_odds" for x in range(teams)]
    prob_cols = [x for x in df.columns.values if x.startswith(prob_col)]
    df = df[df[GAME_DT_COLUMN].dt.year >= datetime.datetime.now().year - 1]

    probs = df[prob_cols].to_numpy()
    odds = df[odds_cols].to_numpy()
    points = df[points_cols].to_numpy()
    best_idx = probs.argmax(axis=1)
    wins_idx = points.argmax(axis=1)
    p = probs[np.arange(len(df)), best_idx]
    o = odds[np.arange(len(df)), best_idx]
    b = o - 1.0
    q = 1.0 - p
    kelly_fraction = (b * p - q) / b
    kelly_fraction = np.clip(kelly_fraction, 0, 1)
    df["kelly_fraction"] = kelly_fraction
    df["bet_won"] = best_idx == wins_idx
    df["bet_odds"] = o
    df = df.dropna(subset=["kelly_fraction", "bet_won", "bet_odds"])

    def scale_fractions(group):
        total = group["kelly_fraction"].sum()
        if total > 1:
            scaling_factor = 1 / total
            group["adjusted_fraction"] = group["kelly_fraction"] * scaling_factor
        else:
            group["adjusted_fraction"] = group["kelly_fraction"]
        return group

    # Check if the dt column is somehow in an index
    if GAME_DT_COLUMN in df.index.names:
        df = df.drop(columns=GAME_DT_COLUMN)
        df = df.reset_index(level=GAME_DT_COLUMN)

    return df.groupby(
        df.columns[df.columns.values.tolist().index(GAME_DT_COLUMN)]
    ).apply(scale_fractions)
