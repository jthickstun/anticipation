import os, pickle

import audio

from argparse import ArgumentParser
from multiprocessing import Pool, RLock
from tqdm import tqdm
from glob import glob

from anticipation.config import *


CACHE_FILE = 'glob_cache.pkl'

def save_cache(files):
    with open(CACHE_FILE, 'wb') as cache:
        pickle.dump(files, cache)

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as cache:
            return pickle.load(cache)
    return None


def main(args):
    datadir = '/juice4/scr4/nlp/music/audio/48khz/encodec_fma'
    outfiles = '/juice4/scr4/nlp/music/audio/48khz/encodec_fma-shard-{s}.txt'

    print('Processing...')
    ecdcs = load_cache() or glob(os.path.join(datadir, '**/*.ecdc'), recursive=True)
    if not os.path.exists(CACHE_FILE):
        save_cache(ecdcs)

    n = len(ecdcs) // PREPROC_WORKERS
    shards = [ecdcs[i:i+n] for i in range(PREPROC_WORKERS)] # dropping a few tracks (< PREPROC_WORKERS)
    outputs = [outfiles.format(s=s) for s in range(len(shards))]

    with Pool(processes=PREPROC_WORKERS, initargs=(RLock(),), initializer=tqdm.set_lock) as pool:
        results = pool.starmap(audio.pack_tokens, zip(shards, outputs, range(PREPROC_WORKERS)))

    files, bad_files, seq_count = (sum(x) for x in zip(*results))

    print('Tokenization complete.')
    print(f'  => Processed {files} input ecdc files')
    print(f'  => Processed {seq_count} training sequences')
    print(f'  => Discarded {bad_files} input files (failed to read)')

if __name__ == '__main__':
    parser = ArgumentParser(description='tokenizes an ecdc-encoded audio dataset')

    main(parser.parse_args())
