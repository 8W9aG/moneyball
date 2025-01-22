"""Tests for the total feature class."""
import unittest

import pandas as pd
from moneyball.strategy.features.columns import \
    team_identifier_column, venue_identifier_column, team_column_prefix, kick_column
from moneyball.strategy.features.total_feature import TotalFeature, TOTAL_COLUMN_PREFIX, TOTAL_GAMES_COLUMN
from sportsball.data.game_model import VENUE_COLUMN_PREFIX
from sportsball.data.league_model import DELIMITER


class TestTotalFeature(unittest.TestCase):

    def setUp(self):
        self._total_feature = TotalFeature()

    def test_team_venue_total_games(self):
        df = pd.DataFrame(data={
            team_identifier_column(0): ["t1", "t1", "t1"],
            team_identifier_column(1): ["t2", "t2", "t2"],
            venue_identifier_column(): ["v1", "v1", "v1"],
            kick_column(0, 0): [0, 1, 2],
            kick_column(0, 1): [1, 2, 3],
            kick_column(1, 0): [3, 4, 5],
            kick_column(1, 1): [5, 6, 7],
        })
        df = self._total_feature.process(df)
        total_games = df[DELIMITER.join([TOTAL_COLUMN_PREFIX, team_column_prefix(0), VENUE_COLUMN_PREFIX, TOTAL_GAMES_COLUMN])].values.tolist()
        self.assertListEqual(total_games, [0, 1, 2])
