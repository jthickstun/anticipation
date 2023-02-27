import os
from argparse import ArgumentParser
from multiprocessing import Pool, RLock
from glob import glob

from tqdm import tqdm

from anticipation.config import *
from anticipation.tokenize import tokenize

def main(args):
    print('Tokenizing LakhMIDI')
    print(f'  train split: {[s for s in LAKH_SPLITS if s not in LAKH_VALID + LAKH_TEST]}')
    print(f'  validation split: {LAKH_VALID}')
    print(f'  test split: {LAKH_TEST}')

    print('Tokenization parameters:')
    print(f'  anticipation interval = {DELTA}s')
    print(f'  augment = {AUGMENT_FACTOR}x')
    print(f'  min events = {MIN_TRACK_EVENTS}')

    paths = [os.path.join(args.datadir, s) for s in LAKH_SPLITS]
    files = [glob(f'{p}/*.compound.txt') for p in paths]
    outputs = [os.path.join(args.datadir, s, f'tokenized-events-{s}.txt') for s in LAKH_SPLITS]

    # don't augment the valid/test splits
    augment = [1 if s in LAKH_VALID or s in LAKH_TEST else AUGMENT_FACTOR for s in LAKH_SPLITS]

    # parallel tokenization drops the last chunk of < M tokens
    # if concerned about waste: process larger groups of datafiles
    with Pool(processes=PREPROC_WORKERS, initargs=(RLock(),), initializer=tqdm.set_lock) as pool:
        results = pool.starmap(tokenize, zip(files, outputs, augment, range(len(LAKH_SPLITS))))

    seq_count, rest_count, discarded_tracks, discarded_seqs = (sum(x) for x in zip(*results))
    rest_ratio = round(100*float(rest_count)/(seq_count*M),2)

    print('Tokenization complete.')
    print(f'  => Processed {seq_count} training sequences')
    print(f'  => Inserted {rest_count} REST tokens ({rest_ratio}% of events)')
    print(f'  => Discarded {discarded_tracks} event sequences')
    print(f'  => Discarded {discarded_seqs} training sequences')

    print('Remember to shuffle the training split!')

if __name__ == '__main__':
    parser = ArgumentParser(description='tokenizes a MIDI dataset')
    parser.add_argument('datadir', help='directory containing preprocessed MIDI to tokenize')

    main(parser.parse_args())
