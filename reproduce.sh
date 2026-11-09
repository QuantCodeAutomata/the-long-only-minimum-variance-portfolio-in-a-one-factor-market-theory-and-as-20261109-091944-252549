#!/usr/bin/env bash
# Runs the test suite and every experiment of this repository, in order.
set -e

python -m pytest

python -m exp.exp1
python -m exp.exp2
python -m exp.exp3
