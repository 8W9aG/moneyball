"""Tests for the catboost trainer."""
import unittest

from moneyball.strategy.trainers.catboost import CatboostTrainer


class TestCatboostTrainer(unittest.TestCase):

    def setUp(self):
        self._trainer = CatboostTrainer("", [], [])

    def test_not_null(self):
        self.assertIsNotNone(self._trainer)
