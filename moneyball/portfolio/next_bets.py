"""The next dictionary format."""

from typing import TypedDict

from .bet import Bet

NextBets = TypedDict(
    "NextBets",
    {
        "bets": list[Bet],
    },
)
