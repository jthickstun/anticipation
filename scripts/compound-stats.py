from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from anticipation.config import *
from anticipation.convert import compound_to_events
from anticipation.tokenize import maybe_tokenize

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']
plt.rcParams['font.size'] = 16

def dataset_stats(filename):
    with open(filename, 'r') as f:
        compound_tokens = [int(token) for token in f.read().split()]

    _, _, status = maybe_tokenize(compound_tokens)
    time_length = 0 if len(compound_tokens) == 0 else compound_tokens[-5] + compound_tokens[-4]
    return (3*(len(compound_tokens) // 5), time_length, status)


def loghist(filename, data, title, xlabel):
    sns.set_style('whitegrid')
    plt.clf()
    plt.figure(figsize=(10,4))
    #plt.title(title)
    plt.xscale('log')
    plt.xlabel(xlabel)
    plt.ylabel('Density')

    plt.grid(True, which='both', linestyle='-', linewidth=0.5)

    density = sns.kdeplot(data, bw_adjust=1.0)

    plt.tight_layout()
    fig = density.get_figure()
    fig.savefig(filename, dpi=300)


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

    status = [r[2] for r in results]
    print('Filtering statistics: ')
    print(' ==> too short:', len([s for s in status if s == 1]))
    print(' ==> too long:', len([s for s in status if s == 2]))
    print(' ==> too many instruments:', len([s for s in status if s == 3]))

    # prefiltering (can't plot these on the log scale)
    results = [r for r in results if r[0] != 0]

    token_lengths, time_lengths, status = zip(*results)
    time_lengths = [t/float(TIME_RESOLUTION) for t in time_lengths]
    token_count = sum(token_lengths)

    loghist('output/unfiltered_length_tokens.png',
            token_lengths,
            'Unfiltered Distribution of Sequence Lengths',
            'Length in Tokens (log10 scale)')

    loghist('output/unfiltered_length_seconds.png',
            time_lengths,
            'Unfiltered Distribution of Sequences Length',
            'Time in Seconds (log10 scale)')

    filtered_results = [r for r in results if r[2] == 0]
    filtered_ratio = len(filtered_results)/float(len(results) + null_sequences)

    token_lengths, time_lengths, status = zip(*filtered_results)
    time_lengths = [t/float(TIME_RESOLUTION) for t in time_lengths]
    filtered_token_count = sum(token_lengths)
    filtered_token_ratio = filtered_token_count/float(token_count)

    loghist('output/filtered_length_tokens.png',
            token_lengths,
            'Distribution of Sequence Lengths (in tokens)',
            'Length in Tokens (log10 scale)')

    loghist('output/filtered_length_seconds.png',
            time_lengths,
            'Distribution of Sequence Lengths (in seconds)',
            'Time in Seconds (log10 scale)')


    print('Successfully calculated statistics: detailed results available at output/')
    print('  => Number of sequences: ', len(results) + null_sequences)
    print('  => Number of tokens (unfiltered): ', token_count)
    print('  => Number of sequences (filtered): {} ({}%)'.format(
        len(filtered_results), 100*round(filtered_ratio, 2)))
    print('  => Number of tokens (filtered): {} ({}%)'.format(
        filtered_token_count, 100*round(filtered_token_ratio, 2)))
    print(f'  => Total time (filtered): {round(sum(time_lengths)/3600., 2)}h')
    print(f'    - mean time: {round(np.mean(time_lengths))}s')
    print(f'    - std time: {round(np.std(time_lengths))}s')
    print(f'    - mean tokens: {round(np.mean(token_lengths))}')
    print(f'    - std tokens: {round(np.std(token_lengths))}')


if __name__ == '__main__':
    parser = ArgumentParser(description='calculate statistics of the intermediate compound representation')
    parser.add_argument('dir', help='directory containing .mid files for training')
    main(parser.parse_args())
