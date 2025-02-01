#!/bin/sh

set -e

PYTEST_CURRENT_TEST=1 pytest --cov-report=term --cov=moneyball tests
coverage html -d coverage_html
