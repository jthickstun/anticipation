import os, csv, time

from argparse import ArgumentParser

import numpy as np

import torch

from transformers import AutoModelForCausalLM

from anticipation import ops
from anticipation.visuals import visualize
from anticipation.convert import midi_to_interarrival, interarrival_to_midi
from anticipation.convert import midi_to_events, events_to_midi
from anticipation.vocab import MIDI_SEPARATOR,MIDI_START_OFFSET,MIDI_END_OFFSET


def main(args):
    np.random.seed(args.seed)

    print(f'Prompting using model checkpoint: {args.model}')
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model).cuda()
    print(f'Loaded model ({time.time()-t0} seconds)')

    print(f'Writing outputs to {args.dir}/{args.output}')
    try:
        os.makedirs(f'{args.dir}/{args.output}')
    except FileExistsError:
        pass

    print(f'Prompting with tracks in index : {args.dir}/index.csv')
    with open(f'{args.dir}/index.csv', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            prompt_midi = row[header.index('prompt')]
            idx = int(row[header.index('idx')])

            prompt = midi_to_interarrival(os.path.join(args.dir, prompt_midi))
            # search for the last note onset
            max_idx = 0
            for i,token in enumerate(prompt):
                if MIDI_START_OFFSET <= token < MIDI_START_OFFSET + MIDI_END_OFFSET:
                    max_idx = i

            prompt = prompt[:max_idx+1] # strip trailing offsets
            for j in range(args.multiplicity):
                t0 = time.time()

                input_ids = torch.tensor([prompt]).cuda()
                output = model.generate(input_ids, do_sample=True, max_length=1024, top_p=0.95, pad_token_id=MIDI_SEPARATOR)
                output = output[0].cpu().tolist()

                # most convenient way to operate on this stuff is to round-trip through events
                mid = interarrival_to_midi(output)
                events = midi_to_events(mid)
                output = ops.clip(events, 0, args.clip_length)
                mid = events_to_midi(output)
                mid.save(f'{args.dir}/{args.output}/{idx}-clip-v{j}.mid')
                if args.visualize:
                    visualize(output, f'{args.dir}/{args.output}/{idx}-clip-v{j}.png')

                print(f'Generated completion. Sampling time: {time.time()-t0} seconds')


if __name__ == '__main__':
    parser = ArgumentParser(description='generate prompted completions with an interarrival-time model')
    parser.add_argument('dir', help='directory containing an index of MIDI files')
    parser.add_argument('model', help='directory containing an interarrival model checkpoint')
    parser.add_argument('-o', '--output', type=str, default='model',
            help='model description (the name of the output subdirectory)')
    parser.add_argument('-s', '--seed', type=int, default=0,
            help='random seed')
    parser.add_argument('-c', '--count', type=int, default=10,
            help='number of clips to sample')
    parser.add_argument('-m', '--multiplicity', type=int, default=1,
            help='number of generations per clip')
    parser.add_argument('-l', '--clip_length', type=int, default=20,
            help='length of the full clip (in seconds)')
    parser.add_argument('-v', '--visualize', action='store_true',
            help='plot visualizations')
    main(parser.parse_args())
