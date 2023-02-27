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

    # parallel tokenization drops the last chunk of < 1024 tokens
    # if concerned about waste: process larger groups of datafiles
    with Pool(processes=PREPROC_WORKERS, initargs=(RLock(),), initializer=tqdm.set_lock) as pool:
        pool.starmap(tokenize, zip(files, outputs, augment, range(len(LAKH_SPLITS))))

    print('Tokenization complete. Remember to shuffle the training split!')

if __name__ == '__main__':
    parser = ArgumentParser(description='tokenizes a MIDI dataset')
    parser.add_argument('datadir', help='directory containing preprocessed MIDI to tokenize')

    main(parser.parse_args())
