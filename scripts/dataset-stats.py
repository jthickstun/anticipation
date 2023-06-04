from argparse import ArgumentParser
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from anticipation.vocab import MIDI_TIME_OFFSET, MIDI_START_OFFSET, TIME_RESOLUTION, SEPARATOR
from anticipation.ops import max_time

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']
plt.rcParams['font.size'] = 16

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


if __name__ == '__main__':
    parser = ArgumentParser(description='calculate statistics of a tokenized MIDI dataset')
    parser.add_argument('-f', '--filename',
                        help='file containing a tokenized MIDI dataset')
    parser.add_argument('-i', '--interarrival',
            action='store_true',
            help='request interarrival-time enocoding (default to arrival-time encoding)')
    args = parser.parse_args()

    print(f'Calculating statistics for {args.filename}')
    time_lengths  = []
    token_counts = []
    with open(args.filename, 'r') as f:
        for i,line in tqdm(list(enumerate(f))):
            if i % 10 != 0: continue
            tokens = [int(token) for token in line.split()]

            if args.interarrival:
                time_lengths.append(sum(t-MIDI_TIME_OFFSET for t in tokens if t < MIDI_START_OFFSET))
                token_counts.append(len(tokens))
            else:
                if SEPARATOR in tokens:
                    continue # counts are weird; just skip these
                time_lengths.append(max_time(tokens[1:], seconds=False))
                token_counts.append(len(tokens[1:]))

    tokens_per_second = [TIME_RESOLUTION*tokens/float(time) for (tokens, time) in zip(token_counts, time_lengths)]
    print('Total tokens:', sum(token_counts))
    print(f'Total time: {float(sum(time_lengths))/(3600*TIME_RESOLUTION)} hours')
    print('Mean tokens-per-second:', TIME_RESOLUTION*sum(token_counts)/float(sum(time_lengths)))
    print('Std tokens-per-second:', np.std(tokens_per_second))
    print(np.mean(tokens_per_second))

    loghist('output/tokens_per_second.png',
            tokens_per_second,
            'Distribution of Tokens per Second',
            'Tokens per Second (log10 scale)')
