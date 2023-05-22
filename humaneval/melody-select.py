import os,csv

from argparse import ArgumentParser
from glob import glob

import numpy as np

from tqdm import tqdm

from anticipation import ops
from anticipation.visuals import visualize
from anticipation.tokenize import extract_instruments
from anticipation.convert import midi_to_events, events_to_midi
from anticipation.config import TIME_RESOLUTION
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

        # find an ensemble with a healthy (non-drum / effect) instrument collection
        instruments = [instr for instr in ops.get_instruments(clip).keys() if instr != 128]
        if len(instruments) < 4 or len(instruments) > 10:
            if verbose:
                print(f'  rejected: track instrument count out of bounds: {len(instruments)}')
            continue

        # define melody as the intstrument part with the highest (non-drum, non-piano) pitchj
        pitches = {}
        for instr in ops.get_instruments(clip).keys():
            pitches[instr] = []

        for time, _, note in zip(clip[0::3],clip[1::3],clip[2::3]):
            time -= TIME_OFFSET
            note -= NOTE_OFFSET

            instr = note//2**7
            pitch = note - (2**7)*instr

            pitches[instr].append(pitch)

        melody = None
        high = 0
        for instr in ops.get_instruments(clip).keys():
            if instr in [0,9] + list(range(112,129)):
                continue 

            avg = np.mean(pitches[instr])
            if avg > high:
                melody = instr
                high = avg

        assert melody

        # get clips with at least 20 notes of melody
        if ops.get_instruments(clip)[melody] < 20:
            if verbose:
                print('  rejected: too few melodic notes')
            continue

        # prompt should contain the melody line
        if ops.min_time(clip, seconds=True, instr=melody) > prompt_length:
            if verbose:
                print('  rejected: prompt does not contain the melody')
            continue

        # melody shouldn't end early
        if ops.max_time(clip, seconds=True, instr=melody) < (clip_length-2):
            if verbose:
                print('  rejected: melody ends before the end of the clip')
            continue

        break # found one

    return os.path.basename(filenames[idx]), clip, melody


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
        writer.writerow(['idx', 'original', 'conditional', 'parts', 'melody'])

        for i in tqdm(range(args.count)):
            filename, clip, melody = select_sample(filenames, args.prompt_length, args.clip_length)
            parts = ops.get_instruments(clip).keys()
            writer.writerow([i, filename, f'{i}-conditional.mid', len(parts), melody])

            mid = events_to_midi(clip)
            mid.save(f'{args.output}/groundtruth/{i}-clip.mid')
            if args.visualize:
                visualize(clip, f'{args.output}/groundtruth/{i}-clip.png')

            events, controls = extract_instruments(clip, [melody])
            prompt = ops.clip(events, 0, args.prompt_length, clip_duration=False)

            conditional_events = ops.clip(ops.combine(prompt, controls), 0, args.clip_length)
            mid = events_to_midi(conditional_events)
            mid.save(f'{args.output}/{i}-conditional.mid')
            if args.visualize:
                visualize(conditional_events, f'{args.output}/{i}-conditional.png')


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
