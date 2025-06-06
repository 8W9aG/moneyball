"""The bet dictionary format."""

from typing import Any, TypedDict

from .team import Team

Bet = TypedDict(
    "Bet",
    {
        "strategy": str,
        "league": str,
        "kelly": float,
        "weight": float,
        "amount": float,
        "teams": list[Team],
        "dt": str,
        "row": dict[str, Any],
    },
)
