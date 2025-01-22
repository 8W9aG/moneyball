"""A weight based on sigmoid bias towards the current time."""

import numpy as np
import pandas as pd
from scipy.special import expit  # type: ignore

from .weight import Weight


class SigmoidWeight(Weight):
    """Sigmoid weight class."""

    @classmethod
    def name(cls) -> str:
        """The name of the weight class."""
        return "sigmoid"

    def process(self, y: pd.DataFrame) -> np.ndarray:
        return expit(np.linspace(0, 1.0, len(y)))
