#!/bin/sh

set -e

PYTEST_CURRENT_TEST=1 pytest --cov-report=term --cov=sportsball tests
coverage html -d coverage_html
