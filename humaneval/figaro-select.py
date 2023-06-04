import os,csv

from argparse import ArgumentParser
from glob import glob

import numpy as np
from tqdm import tqdm

from anticipation import ops
from anticipation.visuals import visualize
from anticipation.convert import midi_to_events, events_to_midi

def select_prompt(filenames, clip_length, verbose=False):
    for figaro_filename in filenames:
        try:
            figaro = midi_to_events(figaro_filename)
        except Exception:
            continue

        max_time = ops.max_time(figaro)
        if max_time < clip_length:
            if verbose:
                print(f'  rejected: FIGARO continuation is too short ({max_time} seconds)')
            continue

        figaro = ops.clip(figaro, 0, clip_length, clip_duration=True)

        head, tail = os.path.split(figaro_filename)
        try:
            prompt = midi_to_events(os.path.join(head, 'prompt', tail))
        except Exception:
            continue

        max_time = ops.max_time(prompt)

        if max_time < 4:
            if verbose:
                print(f'  rejected: prompt is too short ({max_time} seconds)')
            continue

        if max_time > 6:
            if verbose:
                print(f'  rejected: prompt is too long ({max_time} seconds)')
            continue

        head, tail = os.path.split(figaro_filename)
        try:
            ground = midi_to_events(os.path.join(head, 'ground', tail))
        except Exception:
            continue

        max_time = ops.max_time(ground)
        if max_time < clip_length:
            if verbose:
                print(f'  rejected: ground truth continuation is too short ({max_time} seconds)')
            continue

        ground = ops.clip(ground, 0, clip_length, clip_duration=True)

        yield os.path.basename(figaro_filename), prompt, ground, figaro 


def main(args):
    np.random.seed(args.seed)

    print(f'Selecting random clips for prompting from: {args.dir}')
    filenames = sorted(glob(args.dir + '*.mid'))
    np.random.shuffle(filenames)

    print(f'Saving clips to: {args.output}')
    try:
        os.makedirs(args.output)
    except FileExistsError:
        pass

    try:
        os.makedirs(f'{args.output}/groundtruth')
    except FileExistsError:
        pass

    try:
        os.makedirs(f'{args.output}/figaro')
    except FileExistsError:
        pass

    with open(f'{args.output}/index.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['idx', 'original', 'prompt'])

        data = select_prompt(filenames, args.clip_length, args.verbose)
        for i in tqdm(range(args.count)):
            filename, prompt, ground, figaro = next(data)
            writer.writerow([i, filename, f'{i}-prompt.mid'])

            mid = events_to_midi(prompt)
            mid.save(f'{args.output}/{i}-prompt.mid')
            if args.visualize:
                visualize(prompt, f'{args.output}/{i}-prompt.png')

            mid = events_to_midi(ground)
            mid.save(f'{args.output}/groundtruth/{i}-clip.mid')
            if args.visualize:
                visualize(ground, f'{args.output}/groundtruth/{i}-clip.png')

            mid = events_to_midi(figaro)
            mid.save(f'{args.output}/figaro/{i}-clip.mid')
            if args.visualize:
                visualize(figaro, f'{args.output}/figaro/{i}-clip.png')


if __name__ == '__main__':
    parser = ArgumentParser(description='select prompts for completion human eval')
    parser.add_argument('dir', help='directory containing MIDI files to sample')
    parser.add_argument('-o', '--output', type=str, default='prompt',
            help='output directory')
    parser.add_argument('-s', '--seed', type=int, default=0,
            help='random seed for prompt selection')
    parser.add_argument('-c', '--count', type=int, default=10,
            help='number of clips to sample')
    parser.add_argument('-l', '--clip_length', type=int, default=20,
            help='length of the full clip (in seconds)')
    parser.add_argument('-v', '--visualize', action='store_true',
            help='plot visualizations')
    parser.add_argument('--verbose', action='store_true',
            help='verbose output')
    main(parser.parse_args())
