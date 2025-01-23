"""The team dictionary format."""

from typing import TypedDict

Team = TypedDict(
    "Team",
    {
        "name": str,
        "probability": float,
    },
)
