"""Tests for the kelly fraction class."""
import datetime
import os
import unittest

import pandas as pd
from moneyball.strategy.kelly_fractions import augment_kelly_fractions, calculate_returns, calculate_value
from moneyball.strategy.strategy import HOME_WIN_COLUMN
from moneyball.strategy.features.columns import team_points_column
import wavetrainer as wt
from sportsfeatures.columns import DELIMITER
from sportsball.data.game_model import GAME_DT_COLUMN
from pandas.testing import assert_frame_equal


class TestKellyFraction(unittest.TestCase):

    def setUp(self):
        self.dir = os.path.dirname(__file__)

    def test_kelly_fraction(self):
        rows = 10
        df = pd.DataFrame(
            data={
                DELIMITER.join([HOME_WIN_COLUMN, wt.model.model.PROBABILITY_COLUMN_PREFIX, str(0)]): [x / 10.0 for x in range(rows)],
                DELIMITER.join([HOME_WIN_COLUMN, wt.model.model.PROBABILITY_COLUMN_PREFIX, str(1)]): [(rows - x) / 10.0 for x in range(rows)],
                "teams/0_odds": [1.0 + (x / 10.0) for x in range(rows)],
                "teams/1_odds": [2.0 + (x / 10.0) for x in range(rows)],
                GAME_DT_COLUMN: [datetime.datetime(datetime.datetime.now().year, 10, x + 1) for x in range(rows)],
                team_points_column(0): list(range(rows)),
                team_points_column(1): [x + 1.0 for x in range(rows)],
            }
        )
        df = augment_kelly_fractions(df, 2, HOME_WIN_COLUMN)
        expected_df = pd.read_parquet(os.path.join(self.dir, "kelly_fraction.parquet"))
        assert_frame_equal(df, expected_df)

    def test_kelly_fraction_dt_index(self):
        df = pd.read_parquet(os.path.join(self.dir, "returns.parquet"))
        df = augment_kelly_fractions(df, 2, HOME_WIN_COLUMN)
        expected_df = pd.read_parquet(os.path.join(self.dir, "kelly_fraction_returns.parquet"))
        assert_frame_equal(df, expected_df)

    def test_calculate_returns(self):
        df = pd.read_parquet(os.path.join(self.dir, "kelly_fraction_returns.parquet"))
        returns = calculate_returns(0.5, df, "test")
        expected_df = pd.read_parquet(os.path.join(self.dir, "expected_returns.parquet"))
        assert_frame_equal(returns.to_frame(), expected_df)

    def test_calculate_value(self):
        df = pd.read_parquet(os.path.join(self.dir, "returns_2.parquet"))
        ret = calculate_returns(0.5, df, "test")
        value = calculate_value(ret)
        self.assertEqual(value, 4004.999378325921)
