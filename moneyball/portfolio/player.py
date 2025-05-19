"""The player dictionary format."""

from typing import TypedDict

Player = TypedDict(
    "Player",
    {
        "name": str | None,
        "identifier": str | None,
    },
)
