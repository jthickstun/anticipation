#!/bin/bash
set -e

# cd to the location of this file on disk
cd "$(dirname -- "$0")"

# only formatting a few files so that we don't make a massive diff
# filled with formatting noise
PYTHONPATH=. ruff format ./tests/util
PYTHONPATH=. ruff format ./tests/conftest.py
PYTHONPATH=. ruff format ./tests/test_tokenize_routines.py
PYTHONPATH=. ruff format ./train/v2
PYTHONPATH=. ruff format ./tests/v2
PYTHONPATH=. ruff format ./anticipation/v2
