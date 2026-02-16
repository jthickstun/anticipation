#!/bin/bash
set -e

# cd to the location of this file on disk
cd "$(dirname -- "$0")"

# run all pytest suite
PYTHONPATH=. pytest tests/
