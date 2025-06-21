"""The portfolio class."""

# pylint: disable=line-too-long,too-many-locals
import datetime
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfolio as pf  # type: ignore
import riskfolio as rp  # type: ignore
from fullmonte import plot, simulate  # type: ignore
from sportsball.data.game_model import GAME_DT_COLUMN  # type: ignore
from sportsball.data.game_model import LEAGUE_COLUMN
from sportsball.data.league_model import DELIMITER
from sportsball.data.player_model import (PLAYER_IDENTIFIER_COLUMN,
                                          PLAYER_NAME_COLUMN)

from ..strategy.features.columns import (find_player_count, find_team_count,
                                         player_column_prefix,
                                         team_identifier_column,
                                         team_name_column)
from ..strategy.kelly_fractions import probability_columns
from ..strategy.strategy import Strategy
from .next_bets import NextBets

_PORTFOLIO_FILENAME = "portfolio.json"
_STRATEGIES_KEY = "strategies"


class Portfolio:
    """The portfolio class."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._strategies = []
        self._weights = {}
        os.makedirs(name, exist_ok=True)
        strategy_file = os.path.join(name, _PORTFOLIO_FILENAME)
        if os.path.exists(strategy_file):
            with open(strategy_file, encoding="utf8") as handle:
                data = json.load(handle)
                self._strategies = [Strategy(x) for x in data[_STRATEGIES_KEY].keys()]
                self._weights = data[_STRATEGIES_KEY]

    @property
    def strategies(self) -> list[Strategy]:
        """Find the strategies associated with the portfolio."""
        return self._strategies

    @strategies.setter
    def strategies(self, strategies: list[Strategy]) -> None:
        """Set the strategies associated with the portfolio"""
        self._strategies = strategies
        self._weights = {x.name: 0.0 for x in strategies}
        strategy_file = os.path.join(self._name, _PORTFOLIO_FILENAME)
        with open(strategy_file, "w", encoding="utf8") as handle:
            json.dump(
                {
                    _STRATEGIES_KEY: self._weights,
                },
                handle,
            )

    def fit(self) -> pd.DataFrame:
        """Fits the portfolio to the strategies."""
        # pylint: disable=unsubscriptable-object
        returns = pd.DataFrame([x.returns() for x in self._strategies]).T.fillna(0.0)
        returns.index = pd.to_datetime(returns.index)  # pyright: ignore
        returns.to_parquet(os.path.join(self._name, "returns.parquet"))

        # Walkforward sharpe optimization
        ret = returns.copy()
        if len(returns.columns.values) > 1:
            ret[self._name] = np.nan
            for index in returns.index:
                dt = index
                x = returns[returns.index < dt]
                if x.empty or len(np.unique(x)) < 10:
                    ret.loc[index, self._name] = (
                        returns.loc[index] * (1.0 / len(returns.columns.values))
                    ).sum()
                else:
                    port = rp.Portfolio(returns=returns)
                    weights = port.optimization(
                        model="Classic", rm="MV", obj="MaxRet", hist=True
                    )
                    total_ret = 0.0
                    for col in returns:
                        ret.loc[index, col] *= weights[col]  # type: ignore
                        total_ret += ret.loc[index, col]  # type: ignore
                        self._weights[str(col)] = weights[col]
                    ret.loc[index, self._name] = total_ret
        else:
            self._weights[returns.columns.values[0]] = 1.0

        ret = ret.asfreq("D").fillna(0.0)
        ret.index = ret.index.tz_localize("UTC")  # type: ignore
        return ret

    def render(
        self,
        returns: pd.DataFrame,
        start_money: float = 100000.0,
        from_date: datetime.datetime | None = None,
    ):
        """Renders the statistics of the portfolio."""

        def render_series(series: pd.Series) -> None:
            pf.create_full_tear_sheet(series)
            plt.savefig(os.path.join(self._name, f"{col}_tear_sheet.png"), dpi=300)
            ret = np.concatenate(
                (np.array([start_money]), series.to_numpy().flatten() + 1.0)
            ).cumprod()
            plot(simulate(pd.Series(ret)))
            plt.savefig(os.path.join(self._name, f"{col}_monte_carlo.png"), dpi=300)
            log_series = pd.Series(data=np.log(ret)[1:], index=series.index)
            log_series.plot()
            plt.savefig(os.path.join(self._name, f"{col}_log_returns.png"), dpi=300)

        if from_date is not None:
            returns = returns.loc[returns.index.date >= from_date]  # type: ignore
        for col in returns.columns.values:
            series = returns[col].dropna()
            first_index = series.where(series != 0.0).first_valid_index()
            if first_index is not None:
                series = series[series.index >= first_index]
            render_series(series)

    def next_bets(self) -> NextBets:
        """Find the strategies next bet information."""
        bets: NextBets = {"bets": [], "feature_importances": {}}
        for strategy in self._strategies:
            next_df, feature_importances, kelly_ratio = strategy.next()
            prob_cols = probability_columns(next_df)
            high_feature_importance_dt = sorted(
                list(feature_importances.keys()),
                key=datetime.datetime.fromisoformat,
            )[-1]
            used_feature_names = feature_importances[high_feature_importance_dt]
            bets["feature_importances"][strategy.name] = {
                high_feature_importance_dt: used_feature_names
            }
            next_df.to_parquet(
                os.path.join(self._name, f"next_df_{strategy.name}.parquet")
            )
            team_count = find_team_count(next_df)
            player_count = find_player_count(next_df, team_count)
            for i in range(team_count):
                for ii in range(player_count):
                    identifier_column = DELIMITER.join(
                        [player_column_prefix(i, ii), PLAYER_IDENTIFIER_COLUMN]
                    )
                    name_column = DELIMITER.join(
                        [player_column_prefix(i, ii), PLAYER_IDENTIFIER_COLUMN]
                    )
                    if identifier_column not in next_df.columns.values.tolist():
                        next_df[identifier_column] = None
                    if name_column not in next_df.columns.values.tolist():
                        next_df[name_column] = None
            for row in next_df.itertuples(name=None):
                row_dict = {
                    x: row[count + 1]
                    for count, x in enumerate(next_df.columns.values.tolist())
                }
                for k, v in row_dict.items():
                    if v is not None:
                        continue
                    logging.info("Row %s Feature %s has null value", str(row[0]), k)

                best_idx = 0
                best_prob = 0.0
                for i in range(team_count):
                    prob = row_dict[prob_cols[i]]
                    if prob > best_prob:
                        best_idx = i
                        best_prob = prob
                o = row_dict[f"teams/{best_idx}_odds"]
                b = o - 1.0
                q = 1.0 - best_prob
                kelly_fraction = (b * best_prob - q) / b
                kelly_fraction = np.clip(kelly_fraction, 0, 1)

                bets["bets"].append(
                    {
                        "strategy": strategy.name,
                        "league": row_dict[LEAGUE_COLUMN],
                        "kelly": kelly_ratio,
                        "weight": self._weights[strategy.name],
                        "amount": kelly_fraction,
                        "teams": [
                            {
                                "name": row_dict[team_name_column(x)],
                                # We should fix this as well
                                "probability": row_dict[prob_cols[x]],
                                "players": [
                                    {
                                        "name": row_dict.get(
                                            DELIMITER.join(
                                                [
                                                    player_column_prefix(x, y),
                                                    PLAYER_NAME_COLUMN,
                                                ]
                                            )
                                        ),
                                        "identifier": row_dict.get(
                                            DELIMITER.join(
                                                [
                                                    player_column_prefix(x, y),
                                                    PLAYER_IDENTIFIER_COLUMN,
                                                ]
                                            )
                                        ),
                                    }
                                    for y in range(player_count)
                                ],
                                "identifier": row_dict[team_identifier_column(x)],
                            }
                            for x in range(team_count)
                        ],
                        "dt": row_dict[GAME_DT_COLUMN].isoformat(),
                        "row": {
                            k: v for k, v in row_dict.items() if k in used_feature_names
                        },
                    }
                )
        return bets
