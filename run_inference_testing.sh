#!/bin/bash
set -e

# cd to the location of this file on disk
cd "$(dirname -- "$0")"

PYTHONPATH=. python scripts/v2/sample.py
