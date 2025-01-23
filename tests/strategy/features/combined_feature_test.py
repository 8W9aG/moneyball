"""Tests for the combined feature class."""
import unittest

from moneyball.strategy.features.combined_feature import CombinedFeature


class TestCombinedFeature(unittest.TestCase):

    def setUp(self):
        self._combined_feature = CombinedFeature()

    def test_notnull(self):
        self.assertIsNotNone(self._combined_feature)
