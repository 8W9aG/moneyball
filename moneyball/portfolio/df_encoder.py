"""Dataframe JSON encoder."""

import json
from typing import Any

import pandas as pd


class DFSONEncoder(json.JSONEncoder):
    """Dataframe JSON encoder."""

    def default(self, o: Any) -> Any:
        """Find the default"""
        if isinstance(o, pd.Timestamp):
            return o.isoformat()
        return super().default(o)
