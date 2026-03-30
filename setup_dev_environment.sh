#!/usr/bin/env bash
# you MUST run this with bash my_script_name.sh,
# NOT ./my_script_name.sh, or sh my_script_name.sh
set -e

init_conda() {
  # helper: source file if it exists
  _try_source() {
    local path="$1"
    [[ -f "$path" ]] || return 1
    # shellcheck disable=SC1090
    source "$path"
  }

  # Prefer conda’s reported base, if conda is on PATH
  if command -v conda >/dev/null 2>&1; then
    local base
    base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${base:-}" ]] && _try_source "$base/etc/profile.d/conda.sh"; then
      return 0
    fi
  fi

  # Fallbacks
  _try_source "$HOME/miniconda3/etc/profile.d/conda.sh" && return 0
  _try_source "$HOME/anaconda3/etc/profile.d/conda.sh" && return 0

  echo "Error: could not locate conda or conda.sh. Install conda or add it to PATH." >&2
  return 1
}

if ! init_conda; then
  exit 1
fi

# cd to the location of this file on disk
cd "$(dirname -- "$0")"

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

conda activate ./env
conda install pytorch torchvision torchaudio --yes
pip install lightning

# run the tests
sh run_tests.sh

echo "SUCCESS. Activate your environment with conda activate ./env"
