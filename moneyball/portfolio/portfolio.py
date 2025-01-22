"""The portfolio class."""

import datetime
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfolio as pf  # type: ignore
from fullmonte import plot, simulate  # type: ignore
from skfolio import RiskMeasure  # type: ignore
from skfolio.optimization import MeanRisk, ObjectiveFunction  # type: ignore

from ..strategy import Strategy


class Portfolio:
    """The portfolio class."""

    def __init__(self, strategies: list[Strategy], name: str) -> None:
        self._name = name
        self._strategies = strategies
        os.makedirs(name, exist_ok=True)

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
                except ValueError as e:
                    logging.warning(
                        "Encountered %s when fitting meanrisk model.", str(e)
                    )
                    ret.loc[index, self._name] = (
                        returns.loc[index] * (1.0 / len(returns.columns.values))
                    ).sum()

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
            log_series = pd.Series(data=np.log(ret), index=series.index[1:])
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
