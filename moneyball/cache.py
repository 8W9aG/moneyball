"""Caching utilities."""

from joblib import Memory  # type: ignore

from . import __VERSION__

MEMORY = Memory(".moneyball_cache_" + __VERSION__, verbose=0)
