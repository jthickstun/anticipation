#!/bin/bash
set -e

# cd to the location of this file on disk, expected to be at repo root
cd "$(dirname -- "$0")"

mkdir -p data

DL_LINK="http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz"
wget -P ./data/ "$DL_LINK" -O ./data/lmd_full.tar.gz

cd ./data/

# extract files to ./data/lmd_full
tar -xvf lmd_full.tar.gz

# delete the archive
rm lmd_full.tar.gz
