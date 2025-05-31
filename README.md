# moneyball

<a href="https://pypi.org/project/moneyball/">
    <img alt="PyPi" src="https://img.shields.io/pypi/v/moneyball">
</a>

A library for determining what bets to make.

<p align="center">
    <img src="moneyball.png" alt="moneyball" width="200"/>
</p>

## Dependencies :globe_with_meridians:

Python 3.11.6:

- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [optuna](https://optuna.readthedocs.io/en/stable/)
- [pytz](https://pythonhosted.org/pytz/)
- [python-dateutil](https://github.com/dateutil/dateutil)
- [sportsball](https://github.com/8W9aG/sportsball)
- [tqdm](https://github.com/tqdm/tqdm)
- [pandarallel](https://nalepae.github.io/pandarallel/)
- [joblib](https://joblib.readthedocs.io/en/stable/)
- [matplotlib](https://matplotlib.org/)
- [pyfolio-reloaded](https://github.com/stefan-jansen/pyfolio-reloaded)
- [fullmonte](https://github.com/8W9aG/fullmonte)
- [wavetrainer](https://github.com/8W9aG/wavetrainer)
- [riskfolio](https://github.com/dcajasn/Riskfolio-Lib)
- [sports-features](https://github.com/8W9aG/sports-features)
- [empyrical-reloaded](https://empyrical.ml4trading.io/)
- [textfeats](https://github.com/8W9aG/text-features)

## Raison D'être :thought_balloon:

`moneyball` was split out of the library [sportsball](https://github.com/8W9aG/sportsball) in order to iterate separately on the quantitative strategies and the data powering them. It aims to be an automated way to come up with an optimal betting strategy when supplied with data in a `sportsball` format.

## Architecture :triangular_ruler:

`moneyball` is an object orientated library. The entities are organised like so:

* **Portfolio**: A collection of strategies.
    * **Strategy**: A method to determine what specific bet to make according.
        * **Features**: The features extracted from the data.
        * **Reducers**: The features removed from the data.
        * **Trainers**: The type of models used for training on the data.
        * **Weights**: Weight strategies to apply to the data.

## Installation :inbox_tray:

This is a python package hosted on pypi, so to install simply run the following command:

`pip install moneyball`

or install using this local repository:

`python setup.py install --old-and-unmanageable`

## Usage example :eyes:

There are many different ways of using moneyball, but we generally recommend the CLI. This pairs very well with the sister project [sportsball](https://github.com/8W9aG/sportsball).

### CLI

The following operations can be run on the CLI:

#### Train

To train a new strategy:

```
sportsball --league=nfl - | moneyball test_nfl_strategy train
```

#### Portfolio

To develop a portfolio of strategies:

```
moneyball --strategy=test_nfl_strategy --strategy=test_afl_strategy test_portfolio portfolio
```

#### Next

To get a quantitative report on the next bets to place:

```
moneyball --output=bets.json test_portfolio next
```

This will result in the following JSON written to stdout:

```json
{
    "bets": [{
        "strategy": "test_nfl_strategy",
        "league": "nfl",
        "kelly": 0.32,
        "weight": 0.1,
        "probability": 0.95,
        "teams": [{
            "name": "giants",
            "probability": 0.1
        }, {
            "name": "dolphins",
            "probability": 0.9
        }],
        "dt": "2025-01-23T16:03:46Z"
    }]
}
```

### Python

To create a portfolio, the following example can be used:

```python
from moneyball import moneyball as mnb

df = ... # Fetch the dataframe from sportsball

moneyball = mnb.Moneyball()
strategy = moneyball.create_strategy(df, "test_strategy")
strategy.fit()
portfolio = ball.create_portfolio([strategy], "test_portfolio")
returns = portfolio.fit()
portfolio.render(returns)
```

## License :memo:

The project is available under the [MIT License](LICENSE).
