import os
from argparse import ArgumentParser
from multiprocessing import Pool, RLock
from glob import glob

from tqdm import tqdm

from anticipation.config import *
from anticipation.tokenize import tokenize, tokenize_ia

def main(args):
    encoding = 'interarrival' if args.interarrival else 'arrival'
    print('Tokenizing LakhMIDI')
    print(f'  encoding type: {encoding}')
    print(f'  train split: {[s for s in LAKH_SPLITS if s not in LAKH_VALID + LAKH_TEST]}')
    print(f'  validation split: {LAKH_VALID}')
    print(f'  test split: {LAKH_TEST}')

    print('Tokenization parameters:')
    print(f'  anticipation interval = {DELTA}s')
    print(f'  augment = {args.augment}x')
    print(f'  max track length = {MAX_TRACK_TIME_IN_SECONDS}s')
    print(f'  min track length = {MIN_TRACK_TIME_IN_SECONDS}s')
    print(f'  min track events = {MIN_TRACK_EVENTS}')

    paths = [os.path.join(args.datadir, s) for s in LAKH_SPLITS]
    files = [glob(f'{p}/*.compound.txt') for p in paths]
    outputs = [os.path.join(args.datadir, f'tokenized-events-{s}.txt') for s in LAKH_SPLITS]

    # don't augment the valid/test splits
    augment = [1 if s in LAKH_VALID or s in LAKH_TEST else args.augment for s in LAKH_SPLITS]

    # parallel tokenization drops the last chunk of < M tokens
    # if concerned about waste: process larger groups of datafiles
    func = tokenize_ia if args.interarrival else tokenize
    with Pool(processes=PREPROC_WORKERS, initargs=(RLock(),), initializer=tqdm.set_lock) as pool:
        results = pool.starmap(func, zip(files, outputs, augment, range(len(LAKH_SPLITS))))

    seq_count, rest_count, too_short, too_long, too_manyinstr, discarded_seqs, truncations \
            = (sum(x) for x in zip(*results))
    rest_ratio = round(100*float(rest_count)/(seq_count*M),2)

    trunc_type = 'interarrival' if args.interarrival else 'duration'
    trunc_ratio = round(100*float(truncations)/(seq_count*M),2)

    print('Tokenization complete.')
    print(f'  => Processed {seq_count} training sequences')
    print(f'  => Inserted {rest_count} REST tokens ({rest_ratio}% of events)')
    print(f'  => Discarded {too_short+too_long} event sequences')
    print(f'      - {too_short} too short')
    print(f'      - {too_long} too long')
    print(f'      - {too_manyinstr} too many instruments')
    print(f'  => Discarded {discarded_seqs} training sequences')
    print(f'  => Truncated {truncations} {trunc_type} times ({trunc_ratio}% of {trunc_type}s)')

    print('Remember to shuffle the training split!')

if __name__ == '__main__':
    parser = ArgumentParser(description='tokenizes a MIDI dataset')
    parser.add_argument('datadir', help='directory containing preprocessed MIDI to tokenize')
    parser.add_argument('-k', '--augment', type=int, default=1,
            help='dataset augmentation factor (multiple of 10)')
    parser.add_argument('-i', '--interarrival',
            action='store_true',
            help='request interarrival-time enocoding (default to arrival-time encoding)')

    main(parser.parse_args())
