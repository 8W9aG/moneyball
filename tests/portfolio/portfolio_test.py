"""Tests for the portfolio class."""
import datetime
import shutil
import unittest

import pandas as pd
from moneyball.portfolio.portfolio import Portfolio


class TestPortfolio(unittest.TestCase):

    def setUp(self):
        self._folder = "test_pytest_portfolio"
        self._portfolio = Portfolio(self._folder)

    def tearDown(self):
        shutil.rmtree(self._folder)

    def test_render(self):
        df = pd.DataFrame(data={
            "afl": [0.0, -0.5, 0.5, 0.2, -0.2],
            "nfl": [0.0, 0.5, -0.5, -0.2, -0.2],
        }, index=[
            datetime.datetime(year=2020, month=1, day=1),
            datetime.datetime(year=2020, month=1, day=2),
            datetime.datetime(year=2020, month=1, day=3),
            datetime.datetime(year=2020, month=1, day=4),
            datetime.datetime(year=2020, month=1, day=5),
        ])
        df.index = df.index.tz_localize("UTC")
        df.index = pd.to_datetime(df.index)
        self._portfolio.render(df)
