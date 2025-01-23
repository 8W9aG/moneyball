"""The portfolio class."""

import datetime
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfolio as pf  # type: ignore
from fullmonte import plot, simulate  # type: ignore
from skfolio import RiskMeasure  # type: ignore
from skfolio.optimization import MeanRisk, ObjectiveFunction  # type: ignore
from sportsball.data.game_model import GAME_DT_COLUMN  # type: ignore
from sportsball.data.game_model import LEAGUE_COLUMN

from ..strategy import Strategy
from ..strategy.features.columns import find_team_count, team_name_column
from ..strategy.trainers.output_column import output_prob_column
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
        returns.index = pd.to_datetime(returns.index)

        # Walkforward sharpe optimization
        ret = returns.copy()
        ret[self._name] = np.NaN
        for index in returns.index:
            dt = index
            x = returns[returns.index < dt]
            if x.empty or len(np.unique(x)) < 10:
                ret.loc[index, self._name] = (
                    returns.loc[index] * (1.0 / len(returns.columns.values))
                ).sum()
            else:
                model = MeanRisk(
                    risk_measure=RiskMeasure.VARIANCE,
                    objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
                    portfolio_params={"name": "Max Sharpe"},
                )
                try:
                    model.fit(x.to_numpy())
                    ret.loc[index, self._name] = (
                        returns.loc[index] * model.weights_
                    ).sum()
                    self._weights = {
                        self._strategies[i].name: model.weights_[i]
                        for i in range(len(model.weights_))
                    }
                except ValueError as e:
                    logging.warning(
                        "Encountered %s when fitting meanrisk model.", str(e)
                    )
                    ret.loc[index, self._name] = (
                        returns.loc[index] * (1.0 / len(returns.columns.values))
                    ).sum()
                    self._weights = {
                        self._strategies[i].name: 0.0
                        for i in range(len(self._strategies))
                    }

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
            series = returns[col]
            series = series[
                series.index >= series.where(series != 0.0).first_valid_index()
            ]
            render_series(series)

    def next_bets(self) -> NextBets:
        """Find the strategies next bet information."""
        bets: NextBets = {"bets": []}
        for strategy in self._strategies:
            next_df = strategy.next()
            team_count = find_team_count(next_df)
            for _, row in next_df.iterrows():
                bets["bets"].append(
                    {
                        "strategy": strategy.name,
                        "league": row[LEAGUE_COLUMN],
                        "kelly": strategy.kelly_ratio,
                        "weight": self._weights[strategy.name],
                        "teams": [
                            {
                                "name": row[team_name_column(x)],
                                "probability": row[output_prob_column(x)],
                            }
                            for x in range(team_count)
                        ],
                        "dt": row[GAME_DT_COLUMN],
                    }
                )
        return bets
