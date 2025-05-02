"""The team dictionary format."""

from typing import TypedDict

from .player import Player

Team = TypedDict(
    "Team",
    {"name": str, "probability": float, "players": list[Player], "identifier": str},
)
