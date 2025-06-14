"""Tests for the kelly fraction class."""
import datetime
import os
import unittest

import pandas as pd
from moneyball.strategy.kelly_fractions import augment_kelly_fractions, calculate_returns, calculate_value
from moneyball.strategy.strategy import AWAY_WIN_COLUMN
from moneyball.strategy.features.columns import team_points_column, team_identifier_column
import wavetrainer as wt
from sportsfeatures.columns import DELIMITER
from sportsball.data.game_model import GAME_DT_COLUMN
from pandas.testing import assert_frame_equal


class TestKellyFraction(unittest.TestCase):

    def setUp(self):
        self.dir = os.path.dirname(__file__)

    @property
    def mock_df(self) -> pd.DataFrame:
        rows = 10
        return pd.DataFrame(
            data={
                DELIMITER.join([AWAY_WIN_COLUMN, wt.model.model.PROBABILITY_COLUMN_PREFIX + str(0)]): [x / 10.0 for x in range(rows)],
                DELIMITER.join([AWAY_WIN_COLUMN, wt.model.model.PROBABILITY_COLUMN_PREFIX + str(1)]): [(rows - x) / 10.0 for x in range(rows)],
                "teams/0_odds": [1.0 + (x / 10.0) for x in range(rows)],
                "teams/1_odds": [2.0 + (x / 10.0) for x in range(rows)],
                GAME_DT_COLUMN: [datetime.datetime(datetime.datetime.now().year, 10, x + 1) for x in range(rows)],
                team_points_column(0): list(range(rows)),
                team_points_column(1): [x + 1.0 for x in range(rows)],
                team_identifier_column(0): ["a" for _ in range(rows)],
                team_identifier_column(1): ["b" for _ in range(rows)],
            }
        )

    def test_kelly_fraction(self):
        df = augment_kelly_fractions(self.mock_df, 2)
        #df.to_parquet("kelly_fraction.parquet")
        expected_df = pd.read_parquet(os.path.join(self.dir, "kelly_fraction.parquet"))
        assert_frame_equal(df, expected_df)

    def test_calculate_returns(self):
        df = pd.read_parquet(os.path.join(self.dir, "kelly_fraction.parquet"))
        returns = calculate_returns(0.5, df, "test")
        #returns.to_frame().to_parquet("expected_returns.parquet")
        expected_df = pd.read_parquet(os.path.join(self.dir, "expected_returns.parquet"))
        assert_frame_equal(returns.to_frame(), expected_df)
