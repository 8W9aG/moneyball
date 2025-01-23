"""An enum stating the main functions of moneyball."""

from enum import StrEnum


class Function(StrEnum):
    """The function to perform on moneyball."""

    TRAIN = "train"
    PORTFOLIO = "portfolio"
    NEXT = "next"
