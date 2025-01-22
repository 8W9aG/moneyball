"""The prototype class for a reducer."""

import pandas as pd


class Reducer:
    """The prototype reducer class."""

    # pylint: disable=too-few-public-methods

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the dataframe and remove the necessary features."""
        raise NotImplementedError("process is not implemented in parent class.")
