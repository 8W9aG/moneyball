"""Tests for the skill feature class."""
import datetime
import unittest

import pandas as pd
from moneyball.strategy.features.columns import team_identifier_column, team_points_column, player_identifier_column, player_column_prefix
from moneyball.strategy.features.skill_feature import \
    _find_matches, SkillFeature, SKILL_COLUMN_PREFIX, SKILL_PROBABILITY_COLUMN, YEAR_SLICE_ALL
from sportsball.data.game_model import GAME_DT_COLUMN
from sportsball.data.league_model import DELIMITER
from openskill.models import PlackettLuce


class TestSkillFeatureClass(unittest.TestCase):

    def setUp(self):
        self._skill_feature = SkillFeature(year_slices=[None, 1, 2])

    def test_process(self):
        df = pd.DataFrame(data={
            GAME_DT_COLUMN: [
                datetime.datetime(2009, 12, 31, 0, 0, 0),
                datetime.datetime(2010, 1, 1, 0, 0, 0),
                datetime.datetime(2010, 1, 2, 0, 0, 0),
            ],
            team_identifier_column(0): ["t1", "t1", "t1"],
            team_identifier_column(1): ["t2", "t2", "t2"],
            team_points_column(0): [0, 1, 2],
            team_points_column(1): [3, 4, 5],
            player_identifier_column(0, 0): ["p1", "p1", "p1"],
            player_identifier_column(1, 0): ["p2", "p2", "p2"],
        })
        df = self._skill_feature.process(df)
        # df.to_csv("test_tmp.csv")
        self.assertIsNotNone(df)
        print(df.columns.values)
        print(df)
        probability = df[
            DELIMITER.join([player_column_prefix(0, None), SKILL_COLUMN_PREFIX, "1", SKILL_PROBABILITY_COLUMN])
        ].values.tolist()
        self.assertListEqual(probability, [0.0, 0.34070897206214457, 0.2287564657243515])
        probability = df[
            DELIMITER.join([player_column_prefix(0, None), SKILL_COLUMN_PREFIX, "2", SKILL_PROBABILITY_COLUMN])
        ].values.tolist()
        self.assertListEqual(probability, [0.0, 0.34070897206214457, 0.2287564657243515])
        probability = df[
            DELIMITER.join([player_column_prefix(0, None), SKILL_COLUMN_PREFIX, YEAR_SLICE_ALL, SKILL_PROBABILITY_COLUMN])
        ].values.tolist()
        self.assertListEqual(probability, [0.5, 0.34070897206214457, 0.2287564657243515])


    def test_find_matches(self):
        df = pd.DataFrame(data={
            DELIMITER: [datetime.datetime(2009, 12, 31, 0, 0, 0), datetime.datetime(2010, 1, 1, 0, 0, 0)],
            team_identifier_column(0): ["1", "1"],
            team_identifier_column(1): ["2", "2"],
            team_points_column(0): [0, 1],
            team_points_column(1): [2, 3],
            player_identifier_column(0, 0): ["1", "1"],
            player_identifier_column(1, 0): ["2", "2"],
        })
        team_model = PlackettLuce()
        player_model = PlackettLuce()
        teams = {
            "1": team_model.rating(name="1"),
        }
        players = {
            "1": player_model.rating(name="1"),
        }
        for _, row in df.iterrows():
            _find_matches(row, 1, 5, teams, players)
