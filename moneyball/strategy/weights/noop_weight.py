"""A weight based on no bias towards the current time."""

import numpy as np
import pandas as pd

from .weight import Weight


class NoopWeight(Weight):
    """Noop weight class."""

    @classmethod
    def name(cls) -> str:
        """The name of the weight class."""
        return "noop"

    def process(self, y: pd.DataFrame) -> np.ndarray:
        return np.array([1.0 for _ in range(len(y))])
