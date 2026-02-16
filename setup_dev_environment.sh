#!/usr/bin/env bash
# you MUST run this with bash my_script_name.sh,
# NOT ./my_script_name.sh, or sh my_script_name.sh
set -e

# cd to the location of this file on disk
cd "$(dirname -- "$0")"

source "$(conda info --base)/etc/profile.d/conda.sh"

# create a conda environment
conda create --prefix ./env python=3.11 --yes
conda activate ./env

# build our custom fork of symusic to prevent segfaults
git clone https://github.com/tempoxylophone/symusic.git

cd symusic/

git submodule sync --recursive
git submodule update --init --recursive
git submodule update --remote --recursive
pip install .

cd ../
rm -rf symusic

# install requirements
pip install pytest
pip install mido
pip install tqdm
pip install pandas
pip install ruff
pip install plotly

# run the tests
sh run_tests.sh

echo "SUCCESS. Activate your environment with conda activate ./env"
