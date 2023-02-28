from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from anticipation.config import *


def dataset_stats(filename):
    with open(filename, 'r') as f:
        compound_tokens = [int(token) for token in f.read().split()]

    assert len(compound_tokens) % 5 == 0
    time_length = 0 if len(compound_tokens) == 0 else compound_tokens[-5] + compound_tokens[-4]
    return (len(compound_tokens) // 5, time_length)


def loghist(filename, data, title, xlabel):
    logbins = np.geomspace(min(data), max(data), 30)

    plt.clf()
    plt.hist(data, bins=logbins)
    plt.title(title)
    plt.xscale('log')
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.savefig(filename, dpi=300)


def main(args):
    filenames = glob(args.dir + '/**/*.compound.txt', recursive=True)

    print(f'Calculating statistics for the dataset rooted at {args.dir}')
    with ProcessPoolExecutor(max_workers=PREPROC_WORKERS) as executor:
        results = list(tqdm(
            executor.map(dataset_stats, filenames),
            desc='Computing statistics',
            total=len(filenames)))

    print('Sequences over one hour: ', len([r for r in results if
        r[1] > TIME_RESOLUTION*MAX_TRACK_TIME_IN_SECONDS]))

    null_sequences = len([r for r in results if r[0] == 0])
    print('Sequences with zero tokens: ', null_sequences)

    # prefiltering (can't plot these on the log scale)
    results = [r for r in results if r[0] != 0]

    token_lengths, time_lengths = zip(*results)
    time_lengths = [t/float(TIME_RESOLUTION) for t in time_lengths]
    token_count = sum(token_lengths)

    loghist('output/unfiltered_length_tokens.png',
            token_lengths,
            'Unfiltered Sequence Length (in tokens)',
            'Length in Tokens (log10 scale)')

    loghist('output/unfiltered_length_seconds.png',
            time_lengths,
            'Unfiltered Sequence Length (in seconds)',
            'Time in Seconds (log10 scale)')

    filtered_results = [r for r in results if
        100 < r[0] 
        and TIME_RESOLUTION*MIN_TRACK_TIME_IN_SECONDS <= r[1]
        and r[1] <= TIME_RESOLUTION*MAX_TRACK_TIME_IN_SECONDS]
    filtered_ratio = len(filtered_results)/float(len(results) + null_sequences)

    token_lengths, time_lengths = zip(*filtered_results)
    time_lengths = [t/float(TIME_RESOLUTION) for t in time_lengths]
    filtered_token_count = sum(token_lengths)
    filtered_token_ratio = filtered_token_count/float(token_count)

    loghist('output/filtered_length_tokens.png',
            token_lengths,
            'Unfiltered Sequence Length (in tokens)',
            'Length in Tokens (log10 scale)')

    loghist('output/filtered_length_seconds.png',
            time_lengths,
            'Unfiltered Sequence Length (in seconds)',
            'Time in Seconds (log10 scale)')


    print('Successfully calculated statistics: detailed results available at output/')
    print('  => Number of sequences: ', len(results) + null_sequences)
    print('  => Number of tokens (unfiltered): ', token_count)
    print('  => Number of sequences (filtered): {} ({}%)'.format(
        len(filtered_results), 100*round(filtered_ratio, 2)))
    print('  => Number of tokens (filtered): {} ({}%)'.format(
        filtered_token_count, 100*round(filtered_token_ratio, 2)))


if __name__ == '__main__':
    parser = ArgumentParser(description='prepares a MIDI dataset')
    parser.add_argument('dir', help='directory containing .mid files for training')
    main(parser.parse_args())
