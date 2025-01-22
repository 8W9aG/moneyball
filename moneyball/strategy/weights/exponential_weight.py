"""A weight based on exponential bias towards the current time."""

import numpy as np
import pandas as pd

from .weight import Weight


class ExponentialWeight(Weight):
    """Exponential weight class."""

    @classmethod
    def name(cls) -> str:
        """The name of the weight class."""
        return "exponential"

    def process(self, y: pd.DataFrame) -> np.ndarray:
        return np.linspace(0, 1.0, len(y)) ** 2.0
