import os, csv, time

from argparse import ArgumentParser

import numpy as np

from transformers import AutoModelForCausalLM

from anticipation import ops
from anticipation.visuals import visualize
from anticipation.sample import generate, generate_ar
from anticipation.tokenize import extract_instruments
from anticipation.convert import midi_to_events, events_to_midi
from anticipation.config import TIME_RESOLUTION

np.random.seed(0)

def main(args):
    if args.anticipatory or args.baseline:
        print(f'Accompaniment using model checkpoint: {args.model}')
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(args.model).cuda()
        print(f'Loaded model ({time.time()-t0} seconds)')

    if args.anticipatory:
        print(f'Writing outputs to {args.dir}/anticipatory')
        try:
            os.makedirs(f'{args.dir}/anticipatory')
        except FileExistsError:
            pass

    if args.baseline:
        print(f'Writing outputs to {args.dir}/autoregressive')
        try:
            os.makedirs(f'{args.dir}/autoregressive')
        except FileExistsError:
            pass

    if args.retrieve:
        print(f'Writing outputs to {args.dir}/retrieved')
        try:
            os.makedirs(f'{args.dir}/retrieved')
        except FileExistsError:
            pass

    print(f'Accompanying tracks in index : {args.dir}/index.csv')
    with open(f'{args.dir}/index.csv', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            original = os.path.join(args.midis, row[header.index('original')])
            conditional_midi = row[header.index('conditional')]
            melody = int(row[header.index('melody')])
            idx = int(row[header.index('idx')])

            events = midi_to_events(os.path.join(args.dir, conditional_midi))

            events, controls = extract_instruments(events, [melody])
            prompt = ops.clip(events, 0, args.prompt_length, clip_duration=False)

            for j in range(args.multiplicity):
                t0 = time.time()

                if args.anticipatory:
                    generated_tokens = generate(model, args.prompt_length, args.clip_length, prompt, controls, top_p=0.95)
                    output = ops.clip(ops.combine(generated_tokens, controls), 0, args.clip_length)
                    mid = events_to_midi(output)
                    mid.save(f'{args.dir}/anticipatory/{idx}-clip-v{j}.mid')
                    if args.visualize:
                        visualize(output, f'{args.dir}/anticipatory/{idx}-clip-v{j}.png')

                if args.baseline:
                    generated_tokens = generate_ar(model, args.prompt_length, args.clip_length, prompt, controls, top_p=0.95)
                    output = ops.clip(generated_tokens, 0, args.clip_length)
                    print(len(generated_tokens), len(output))
                    mid = events_to_midi(output)
                    mid.save(f'{args.dir}/autoregressive/{idx}-clip-v{j}.mid')
                    if args.visualize:
                        visualize(output, f'{args.dir}/autoregressive/{idx}-clip-v{j}.png')

                if args.retrieve:
                    original_events = midi_to_events(original)
                    max_time = ops.max_time(original_events) - args.clip_length
                    start_time = max_time*np.random.rand(1)[0] # get a different random clip
                    retrieved = ops.clip(original_events, start_time, start_time+args.clip_length, clip_duration=True)
                    retrieved = ops.translate(retrieved, -int(TIME_RESOLUTION*start_time))
                    events, _ = extract_instruments(retrieved, [melody])
                    generated = prompt + ops.clip(events, args.prompt_length, args.clip_length)
                    output = ops.combine(generated, controls)
                    mid = events_to_midi(output)
                    mid.save(f'{args.dir}/retrieved/{idx}-clip-v{j}.mid')
                    if args.visualize:
                        visualize(output, f'{args.dir}/retrieved/{idx}-clip-v{j}.png')


                print(f'Accompanied with instrument {melody}. Sampling time: {time.time()-t0} seconds')


if __name__ == '__main__':
    parser = ArgumentParser(description='generate infilling completions')
    parser.add_argument('dir', help='directory containing an index of MIDI files')
    parser.add_argument('--model', type=str, default='',
            help='directory containing an anticipatory model checkpoint')
    parser.add_argument('-c', '--count', type=int, default=10,
            help='number of clips to sample')
    parser.add_argument('-m', '--multiplicity', type=int, default=1,
            help='number of generations per clip')
    parser.add_argument('-p', '--prompt_length', type=int, default=5,
            help='length of the prompt (in seconds)')
    parser.add_argument('-l', '--clip_length', type=int, default=20,
            help='length of the full clip (in seconds)')
    parser.add_argument('-a', '--anticipatory', action='store_true',
            help='generate anticipatory results')
    parser.add_argument('-b', '--baseline', action='store_true',
            help='generate autoregressive (baseline) results')
    parser.add_argument('-r', '--retrieve', action='store_true',
            help='generate the retrieval baseline')
    parser.add_argument('-d', '--midis', type=str, default='',
            help='directory containing the reference MIDI files (for retrieval)')
    parser.add_argument('-v', '--visualize', action='store_true',
            help='plot visualizations')
    main(parser.parse_args())
