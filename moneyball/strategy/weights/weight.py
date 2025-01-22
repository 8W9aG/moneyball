"""The prototype class for a weight."""

import numpy as np
import pandas as pd


class Weight:
    """The prototype weight class."""

    @classmethod
    def name(cls) -> str:
        """The name of the weight class."""
        raise NotImplementedError("name is not implemented in parent class.")

    def process(self, y: pd.DataFrame) -> np.ndarray:
        """Process the dataframe and remove the necessary features."""
        raise NotImplementedError("process is not implemented in parent class.")
