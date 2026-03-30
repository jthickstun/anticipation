#!/bin/bash
set -e

# cd to the location of this file on disk, expected to be at repo root
cd "$(dirname -- "$0")"

# download the 'pruned' split of aria-midi, to ./data/
# see here: https://github.com/loubbrad/aria-midi
wget -P ./data/ https://huggingface.co/datasets/loubb/aria-midi/resolve/main/aria-midi-v1-pruned-ext.tar.gz?download=true -O ./data/aria-midi-v1-pruned-ext.tar.gz

cd ./data/

# extract files to ./data/aria-midi-v1-pruned-ext/...
# is about 12.6 GB uncompressed
tar -xvf aria-midi-v1-pruned-ext.tar.gz

# delete the archive
rm aria-midi-v1-pruned-ext.tar.gz