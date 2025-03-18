"""Caching utilities."""

import os

from joblib import Memory  # type: ignore

from . import __VERSION__

MEMORY = Memory(".moneyball_cache_" + __VERSION__, verbose=0)
_CACHE_TMP_FOLDER = ".moneyball_cache_tmp"


def moneyball_cachetmp_folder() -> str:
    """Return a valid tmp folder."""
    if not os.path.exists(_CACHE_TMP_FOLDER):
        os.mkdir(_CACHE_TMP_FOLDER)
    return _CACHE_TMP_FOLDER
