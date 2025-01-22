"""The main moneyball class for computing strategies."""

from warnings import simplefilter

import pandas as pd

from .portfolio import Portfolio
from .strategy import Strategy


class Moneyball:
    """The main moneyball class."""

    def __init__(self) -> None:
        simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    def create_strategy(self, df: pd.DataFrame, name: str) -> Strategy:
        """Creates a strategy."""
        return Strategy(df, name)

    def create_portfolio(self, strategies: list[Strategy], name: str) -> Portfolio:
        """Creates a portfolio."""
        return Portfolio(strategies, name)
