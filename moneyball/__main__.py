"""The CLI for executing the signal extraction."""

# pylint: disable=too-many-statements
import argparse
import io
import json
import logging
import sys
import time
import warnings

import pandas as pd
from sportsball.loglevel import LogLevel  # type: ignore

from . import __VERSION__
from .function import Function
from .portfolio import Portfolio
from .portfolio.df_encoder import DFSONEncoder
from .strategy import Strategy

warnings.simplefilter(action="ignore", category=FutureWarning)


def main() -> None:
    """The main CLI function."""
    logging.basicConfig()
    logger = logging.getLogger()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loglevel",
        default=LogLevel.INFO,
        choices=list(LogLevel),
        help="The loglevel to display logs at.",
        required=False,
    )
    parser.add_argument(
        "--strategy",
        nargs="*",
        help="A strategy to use.",
        required=False,
    )
    parser.add_argument(
        "--output",
        help="The file to use as the output.",
        required=False,
    )
    parser.add_argument(
        "--cached",
        help="Whether to use the cached dataframe.",
        required=False,
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--place",
        help="The placing to be considered a win.",
        required=False,
        default=1,
        type=int,
    )
    parser.add_argument(
        "--disable_multiprocessing",
        help="Whether to disable multiprocessing on sports features.",
        required=False,
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--input_file",
        help="The input file to read.",
        required=False,
    )
    parser.add_argument(
        "name",
        help="The name of the strategy/portfolio.",
    )
    parser.add_argument(
        "function",
        default=Function.TRAIN,
        choices=list(Function),
        help="The main function for moneyball to perform.",
    )
    args = parser.parse_args()

    match args.loglevel:
        case LogLevel.DEBUG:
            logger.setLevel(logging.DEBUG)
        case LogLevel.INFO:
            logger.setLevel(logging.INFO)
        case LogLevel.WARN:
            logger.setLevel(logging.WARN)
        case LogLevel.ERROR:
            logger.setLevel(logging.ERROR)
        case _:
            raise ValueError(f"Unrecognised loglevel: {args.loglevel}")

    logging.info("--- moneyball %s ---", __VERSION__)
    pd.options.io.parquet.engine = "pyarrow"

    match args.function:
        case Function.TRAIN:
            strategy = Strategy(args.name, args.place, not args.disable_multiprocessing)
            if not args.cached:
                logging.info("Begin loading dataframe")
                start_time = time.perf_counter()
                if args.input_file is not None:
                    df = pd.read_parquet(
                        args.input_file, engine="pyarrow", memory_map=True
                    )
                    strategy.df = df
                else:
                    parquet_bytes = io.BytesIO(sys.stdin.buffer.read())
                    parquet_bytes.seek(0)
                    df = pd.read_parquet(parquet_bytes)
                    strategy.df = df
                end_time = time.perf_counter()
                logging.info("Loaded dataframe in %f", end_time - start_time)
            strategy.fit()
        case Function.PORTFOLIO:
            if args.name is None:
                raise ValueError("--name cannot be empty when creating a portfolio.")
            strategies = [Strategy(x) for x in args.strategy]
            if not strategies:
                raise ValueError(
                    "--strategy needs to be defined at least once to create a portfolio."
                )
            portfolio = Portfolio(args.name)
            portfolio.strategies = strategies
            returns = portfolio.fit()
            portfolio.render(returns)
        case Function.NEXT:
            if args.name is None:
                raise ValueError(
                    "--name cannot be empty when finding the next bets in a portfolio."
                )
            portfolio = Portfolio(args.name)
            bets = portfolio.next_bets()
            sys.stdout.write(json.dumps(bets, cls=DFSONEncoder))
            if args.output is not None:
                with open(args.output, "w", encoding="utf8") as handle:
                    json.dump(bets, handle, cls=DFSONEncoder)
        case _:
            raise ValueError(f"Unrecognised function: {args.function}")


if __name__ == "__main__":
    main()
