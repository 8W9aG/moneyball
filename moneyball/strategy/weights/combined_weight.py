"""A weight based on the combined weight strategies."""

import numpy as np
import pandas as pd

from .class_weight import ClassWeight
from .exponential_weight import ExponentialWeight
from .linear_weight import LinearWeight
from .noop_weight import NoopWeight
from .sigmoid_weight import SigmoidWeight
from .weight import Weight

WEIGHTS = {
    ExponentialWeight.name(): ExponentialWeight,
    LinearWeight.name(): LinearWeight,
    NoopWeight.name(): NoopWeight,
    SigmoidWeight.name(): SigmoidWeight,
}


class CombinedWeight(Weight):
    """Combined weight class."""

    def __init__(self, weight_name: str) -> None:
        self.weight_name = weight_name
        self._class_weight = ClassWeight()
        self._weight = WEIGHTS[weight_name]()

    @classmethod
    def name(cls) -> str:
        """The name of the weight class."""
        return "combined"

    def process(self, y: pd.DataFrame) -> np.ndarray:
        w1_arr = self._class_weight.process(y)
        w2_arr = self._weight.process(y)
        return w1_arr * w2_arr
