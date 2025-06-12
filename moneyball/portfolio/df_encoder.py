"""Dataframe JSON encoder."""

import datetime
import json
from typing import Any

import pandas as pd


class DFSONEncoder(json.JSONEncoder):
    """Dataframe JSON encoder."""

    def default(self, o: Any) -> Any:
        """Find the default"""
        if isinstance(o, (pd.Timestamp, datetime.datetime, datetime.date)):
            return o.isoformat()
        return super().default(o)
