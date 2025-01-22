"""Tests for the class weight class."""
import unittest

import moneyball as mnb
import pandas as pd
import numpy as np


class TestClassWeight(unittest.TestCase):

    def setUp(self):
        self._class_weight = mnb.strategy.weights.class_weight.ClassWeight()

    def test_process(self):
        y = pd.DataFrame(data={"y": [True, False, True, False, True, True]})
        weight = self._class_weight.process(y)
        np.testing.assert_array_equal(weight, np.array([0.75, 1.5 , 0.75, 1.5 , 0.75, 0.75]))
