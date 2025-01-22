#!/bin/sh

set -e

echo "Formatting..."
echo "--- Ruff ---"
ruff format moneyball
echo "--- isort ---"
isort moneyball

echo "Checking..."
echo "--- Flake8 ---"
flake8 moneyball
echo "--- pylint ---"
pylint moneyball
echo "--- mypy ---"
mypy moneyball
echo "--- Ruff ---"
ruff check moneyball
echo "--- pyright ---"
pyright moneyball
