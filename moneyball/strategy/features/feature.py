"""The prototype class for a feature."""

import pandas as pd


class Feature:
    """The prototype feature class."""

    # pylint: disable=too-few-public-methods

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the dataframe and add the necessary features."""
        raise NotImplementedError("process is not implemented in parent class.")
