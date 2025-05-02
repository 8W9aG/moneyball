"""The player dictionary format."""

from typing import TypedDict

Player = TypedDict(
    "Player",
    {
        "name": str,
        "identifier": str,
    },
)
