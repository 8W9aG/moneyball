"""A weight based on the prediction class numbers."""

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight  # type: ignore

from .weight import Weight


class ClassWeight(Weight):
    """Class weight class."""

    @classmethod
    def name(cls) -> str:
        """The name of the weight class."""
        return "class"

    def process(self, y: pd.DataFrame) -> np.ndarray:
        arr = y.astype(int).to_numpy().flatten().astype(float)
        unique_vals = np.unique(arr)
        w_arr = compute_class_weight(
            class_weight="balanced", classes=unique_vals, y=arr
        )
        for count, unique_val in enumerate(unique_vals):
            arr[arr == unique_val] = w_arr[count]
        return arr
