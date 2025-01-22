"""Tests for the trainer."""
import unittest

from moneyball.strategy.trainers.trainer import Trainer
import pandas as pd


class TestTrainer(unittest.TestCase):

    def setUp(self):
        self._trainer = Trainer("test_tmp")

    def test_split_train_test(self):
        x = pd.DataFrame(data={"x": [1.0, 0.0, 0.2, 0.3, 0.5, 0.6]})
        y = pd.DataFrame(data={"y": [True, False, True, False, True, True]})
        (x_train, x_test), (y_train, y_test) = self._trainer.split_train_test(x, y)
        self.assertEqual(len(x_train), 4)
        self.assertEqual(len(x_test), 2)
        self.assertEqual(len(y_train), 4)
        self.assertEqual(len(y_test), 2)
