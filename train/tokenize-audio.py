import os, pickle

from argparse import ArgumentParser
from multiprocessing import Pool, RLock
from tqdm import tqdm
from glob import glob

from anticipation.config import *
from anticipation.mmvocab import vocab
from anticipation.audio import pack_tokens

SEQ_LEN = 8192
CACHE_FILE = 'glob_cache.pkl'


def save_cache(files):
    with open(CACHE_FILE, 'wb') as cache:
        pickle.dump(files, cache)

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as cache:
            return pickle.load(cache)
    return None


def preprocess(ecdcs, output, idx):
    return pack_tokens(ecdcs, output, idx, vocab, seqlen=SEQ_LEN)


def main(args):
    outfiles = args.datadir + '-shard-{s}.txt'

    print('Tokenizing dataset at ', args.datadir)
    print('Processing...')
    ecdcs = load_cache() or glob(os.path.join(args.datadir, '**/*.ecdc'), recursive=True)
    if not os.path.exists(CACHE_FILE):
        save_cache(ecdcs)

    n = len(ecdcs) // PREPROC_WORKERS
    shards = [ecdcs[i:i+n] for i in range(PREPROC_WORKERS)] # dropping a few tracks (< PREPROC_WORKERS)
    outputs = [outfiles.format(s=s) for s in range(len(shards))]

    with Pool(processes=PREPROC_WORKERS, initargs=(RLock(),), initializer=tqdm.set_lock) as pool:
        results = pool.starmap(preprocess, zip(shards, outputs, range(PREPROC_WORKERS)))

    files, bad_files, seq_count = (sum(x) for x in zip(*results))

    print('Tokenization complete.')
    print(f'  => Processed {files} input ecdc files')
    print(f'  => Processed {seq_count} training sequences')
    print(f'  => Discarded {bad_files} input files (failed to read)')

if __name__ == '__main__':
    parser = ArgumentParser(description='tokenizes an ecdc-encoded audio dataset')
    parser.add_argument('datadir', help='directory containing the audio dataset to tokenize')

    main(parser.parse_args())
