"""Tests for the offensive efficiency feature class."""
import unittest

import pandas as pd
from moneyball.strategy.features.columns import team_identifier_column, attendance_column, player_identifier_column, player_column_prefix, team_column_prefix
from moneyball.strategy.features.offensive_efficiency_feature import OffensiveEfficiencyFeature, OFFENSIVE_EFFICIENCY_COLUMN
from sportsball.data.league_model import DELIMITER
from sportsball.data.field_type import FieldType
from sportsball.data.player_model import FIELD_GOALS_COLUMN, FIELD_GOALS_ATTEMPTED_COLUMN, ASSISTS_COLUMN, TURNOVERS_COLUMN, OFFENSIVE_REBOUNDS_COLUMN


class TestOffensiveEfficiencyFeature(unittest.TestCase):

    def setUp(self):
        self._offensive_efficiency_feature = OffensiveEfficiencyFeature()

    def test_process(self):
        df = pd.DataFrame(data={
            team_identifier_column(0): ["t1", "t1"],
            DELIMITER.join([team_column_prefix(0), FIELD_GOALS_COLUMN]): [10.0, 20.0],
            DELIMITER.join([team_column_prefix(0), FIELD_GOALS_ATTEMPTED_COLUMN]): [10.0, 20.0],
            DELIMITER.join([team_column_prefix(0), ASSISTS_COLUMN]): [10.0, 20.0],
            DELIMITER.join([team_column_prefix(0), TURNOVERS_COLUMN]): [10.0, 20.0],
            DELIMITER.join([team_column_prefix(0), OFFENSIVE_REBOUNDS_COLUMN]): [10.0, 20.0],
            player_identifier_column(0, 0): ["p1", "p1"],
            DELIMITER.join([player_column_prefix(0, 0), FIELD_GOALS_COLUMN]): [1.0, 2.0],
            DELIMITER.join([player_column_prefix(0, 0), FIELD_GOALS_ATTEMPTED_COLUMN]): [1.0, 2.0],
            DELIMITER.join([player_column_prefix(0, 0), ASSISTS_COLUMN]): [1.0, 2.0],
            DELIMITER.join([player_column_prefix(0, 0), TURNOVERS_COLUMN]): [1.0, 2.0],
            DELIMITER.join([player_column_prefix(0, 0), OFFENSIVE_REBOUNDS_COLUMN]): [1.0, 2.0],
            player_identifier_column(0, 1): ["p2", "p2"],
            DELIMITER.join([player_column_prefix(0, 1), FIELD_GOALS_COLUMN]): [3.0, 4.0],
            DELIMITER.join([player_column_prefix(0, 1), FIELD_GOALS_ATTEMPTED_COLUMN]): [3.0, 4.0],
            DELIMITER.join([player_column_prefix(0, 1), ASSISTS_COLUMN]): [3.0, 4.0],
            DELIMITER.join([player_column_prefix(0, 1), TURNOVERS_COLUMN]): [3.0, 4.0],
            DELIMITER.join([player_column_prefix(0, 1), OFFENSIVE_REBOUNDS_COLUMN]): [3.0, 4.0],
            team_identifier_column(1): ["t2", "t2"],
            DELIMITER.join([team_column_prefix(1), FIELD_GOALS_COLUMN]): [30.0, 40.0],
            DELIMITER.join([team_column_prefix(1), FIELD_GOALS_ATTEMPTED_COLUMN]): [30.0, 40.0],
            DELIMITER.join([team_column_prefix(1), ASSISTS_COLUMN]): [30.0, 40.0],
            DELIMITER.join([team_column_prefix(1), TURNOVERS_COLUMN]): [30.0, 40.0],
            DELIMITER.join([team_column_prefix(1), OFFENSIVE_REBOUNDS_COLUMN]): [30.0, 40.0],
            player_identifier_column(1, 0): ["p3", "p3"],
            DELIMITER.join([player_column_prefix(1, 0), FIELD_GOALS_COLUMN]): [5.0, 6.0],
            DELIMITER.join([player_column_prefix(1, 0), FIELD_GOALS_ATTEMPTED_COLUMN]): [5.0, 6.0],
            DELIMITER.join([player_column_prefix(1, 0), ASSISTS_COLUMN]): [5.0, 6.0],
            DELIMITER.join([player_column_prefix(1, 0), TURNOVERS_COLUMN]): [5.0, 6.0],
            DELIMITER.join([player_column_prefix(1, 0), OFFENSIVE_REBOUNDS_COLUMN]): [5.0, 6.0],
            player_identifier_column(1, 1): ["p4", "p4"],
            DELIMITER.join([player_column_prefix(1, 1), FIELD_GOALS_COLUMN]): [7.0, 8.0],
            DELIMITER.join([player_column_prefix(1, 1), FIELD_GOALS_ATTEMPTED_COLUMN]): [7.0, 8.0],
            DELIMITER.join([player_column_prefix(1, 1), ASSISTS_COLUMN]): [7.0, 8.0],
            DELIMITER.join([player_column_prefix(1, 1), TURNOVERS_COLUMN]): [7.0, 8.0],
            DELIMITER.join([player_column_prefix(1, 1), OFFENSIVE_REBOUNDS_COLUMN]): [7.0, 8.0],
        })
        df.attrs[str(FieldType.LOOKAHEAD)] = []
        df = self._offensive_efficiency_feature.process(df)
        oe_t1 = df[DELIMITER.join([team_column_prefix(0), OFFENSIVE_EFFICIENCY_COLUMN])].tolist()
        oe_t2 = df[DELIMITER.join([team_column_prefix(0), OFFENSIVE_EFFICIENCY_COLUMN])].tolist()
        oe_p1 = df[DELIMITER.join([player_column_prefix(0, 0), OFFENSIVE_EFFICIENCY_COLUMN])].tolist()
        oe_p2 = df[DELIMITER.join([player_column_prefix(0, 1), OFFENSIVE_EFFICIENCY_COLUMN])].tolist()
        oe_p3 = df[DELIMITER.join([player_column_prefix(1, 0), OFFENSIVE_EFFICIENCY_COLUMN])].tolist()
        oe_p4 = df[DELIMITER.join([player_column_prefix(1, 1), OFFENSIVE_EFFICIENCY_COLUMN])].tolist()
        self.assertListEqual(oe_t1, [1.0, 1.0])
        self.assertListEqual(oe_t2, [1.0, 1.0])
        self.assertListEqual(oe_p1, [1.0, 1.0])
        self.assertListEqual(oe_p2, [1.0, 1.0])
        self.assertListEqual(oe_p3, [1.0, 1.0])
        self.assertListEqual(oe_p4, [1.0, 1.0])
