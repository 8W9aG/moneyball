"""Tests for the lag feature class."""
import unittest

import pandas as pd
from moneyball.strategy.features.columns import team_identifier_column, venue_identifier_column, attendance_column, kick_column, player_identifier_column
from moneyball.strategy.features.lag_feature import LagFeature, LAG_COLUMN_PREFIX
from sportsball.data.league_model import DELIMITER


class TestLagFeature(unittest.TestCase):

    def setUp(self):
        self._lag_feature = LagFeature()

    def test_process(self):
        df = pd.DataFrame(data={
            team_identifier_column(0): ["1", "1"],
            team_identifier_column(1): ["2", "2"],
            venue_identifier_column(): ["4", "4"],
            attendance_column(): [12000, 13000]
        })
        df = self._lag_feature.process(df)
        lag_attendances = df[DELIMITER.join([LAG_COLUMN_PREFIX, attendance_column()])].values.tolist()[1:]
        self.assertListEqual(lag_attendances, [12000.0])

    def test_kicks(self):
        df = pd.DataFrame(data={
            team_identifier_column(0): ["1", "1"],
            team_identifier_column(1): ["2", "2"],
            venue_identifier_column(): ["4", "4"],
            attendance_column(): [12000, 13000],
            player_identifier_column(0, 0): ["p1", "p1"],
            player_identifier_column(0, 1): ["p2", "p2"],
            player_identifier_column(1, 0): ["p3", "p3"],
            player_identifier_column(1, 1): ["p4", "p4"],
            kick_column(0, 0): [10, 20],
            kick_column(0, 1): [20, 30],
            kick_column(1, 0): [30, 40],
            kick_column(1, 1): [40, 50],
        })
        df = self._lag_feature.process(df)
        lag_kicks = df[DELIMITER.join([LAG_COLUMN_PREFIX, kick_column(0, 0)])].values.tolist()[1:]
        self.assertListEqual(lag_kicks, [10.0])
        lag_kicks = df[DELIMITER.join([LAG_COLUMN_PREFIX, kick_column(0, 1)])].values.tolist()[1:]
        self.assertListEqual(lag_kicks, [20.0])
