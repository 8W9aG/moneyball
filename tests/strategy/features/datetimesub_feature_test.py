"""Tests for the datetime sub feature class."""
import datetime
import unittest

import pandas as pd
from moneyball.strategy.features.datetimesub_feature import DatetimeSubFeature
from sportsball.data.game_model import GAME_DT_COLUMN, END_DT_COLUMN
from sportsball.data.field_type import FieldType  # type: ignore


class TestDatetimeSubFeature(unittest.TestCase):

    def setUp(self):
        self._datetimesub_feature = DatetimeSubFeature(END_DT_COLUMN, GAME_DT_COLUMN, "m")

    def test_process(self):
        df = pd.DataFrame(data={
            GAME_DT_COLUMN: [datetime.datetime(2010, 1, 1, 12, 40, 0)],
            END_DT_COLUMN: [datetime.datetime(2010, 1, 1, 12, 45, 0)],
        })
        df.attrs[str(FieldType.LOOKAHEAD)] = []
        df = self._datetimesub_feature.process(df)
        minutes = df[self._datetimesub_feature.column_name].tolist()
        self.assertListEqual(minutes, [5])
