import os,csv

from argparse import ArgumentParser
from glob import glob

import numpy as np

from tqdm import tqdm

from anticipation import ops
from anticipation.visuals import visualize
from anticipation.tokenize import extract_instruments
from anticipation.convert import midi_to_events, events_to_midi
from anticipation.config import TIME_RESOLUTION, EVENT_SIZE
from anticipation.vocab import TIME_OFFSET, NOTE_OFFSET

def select_sample(filenames, prompt_length, clip_length, verbose=False):
    while True:
        # sampling with replacement
        idx = np.random.randint(len(filenames))
        if verbose:
            print('Loading index: ', idx)

        try:
            events = midi_to_events(filenames[idx])
        except Exception:
            continue

        max_time = ops.max_time(events) - clip_length

        # don't sample tracks with length shorter than clip_length
        if max_time < 0:
            if verbose:
                print(f'  rejected: track is too short (length {ops.max_time(events)} < {clip_length})')
            continue

        start_time = max_time*np.random.rand(1)[0]
        clip = ops.clip(events, start_time, start_time+clip_length, clip_duration=True)
        clip = ops.translate(clip, -int(TIME_RESOLUTION*start_time))

        instruments = ops.get_instruments(clip).keys()
        if len(instruments) > 15:
            if verbose:
                print(f'  rejected: track instrument count out of bounds: {len(instruments)}')
            continue

        prompt = ops.clip(clip, 0, prompt_length, clip_duration=False)

        # get clips with at least 10 events in the prompt
        if len(prompt) < EVENT_SIZE*10:
            if verbose:
                print(f'  rejected: track has {len(prompt)//EVENT_SIZE} < 10 events in the prompt')
            continue

        break # found one

    return os.path.basename(filenames[idx]), clip, prompt


def main(args):
    np.random.seed(args.seed)

    print(f'Selecting clips for accompaniment from: {args.dir}')
    filenames = glob(args.dir + '/**/*.mid', recursive=True) \
            + glob(args.dir + '/**/*.midi', recursive=True)
    filenames = sorted(filenames)

    print(f'Saving clips to: {args.output}')
    try:
        os.makedirs(args.output)
    except FileExistsError:
        pass

    try:
        os.makedirs(f'{args.output}/groundtruth')
    except FileExistsError:
        pass

    with open(f'{args.output}/index.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['idx', 'original', 'prompt', 'parts'])

        for i in tqdm(range(args.count)):
            filename, clip, prompt = select_sample(filenames, args.prompt_length, args.clip_length)
            parts = ops.get_instruments(clip).keys()
            writer.writerow([i, filename, f'{i}-conditional.mid', len(parts)])

            mid = events_to_midi(clip)
            mid.save(f'{args.output}/groundtruth/{i}-clip.mid')
            if args.visualize:
                visualize(clip, f'{args.output}/groundtruth/{i}-clip.png')

            mid = events_to_midi(prompt)
            mid.save(f'{args.output}/{i}-conditional.mid')
            if args.visualize:
                visualize(prompt, f'{args.output}/{i}-conditional.png')


if __name__ == '__main__':
    parser = ArgumentParser(description='select prompts for infilling completion human eval')
    parser.add_argument('dir', help='directory containing MIDI files to sample')
    parser.add_argument('-o', '--output', type=str, default='output',
            help='output directory')
    parser.add_argument('-s', '--seed', type=int, default=0,
            help='random seed for sampling')
    parser.add_argument('-c', '--count', type=int, default=10,
            help='number of clips to sample')
    parser.add_argument('-p', '--prompt_length', type=int, default=5,
            help='length of the prompt (in seconds)')
    parser.add_argument('-l', '--clip_length', type=int, default=20,
            help='length of the full clip (in seconds)')
    parser.add_argument('-v', '--visualize', action='store_true',
            help='plot visualizations')
    main(parser.parse_args())
