import os, csv, time

from argparse import ArgumentParser

import numpy as np

from transformers import AutoModelForCausalLM

from anticipation import ops
from anticipation.visuals import visualize
from anticipation.sample import generate
from anticipation.convert import midi_to_events, events_to_midi

np.random.seed(0)

def main(args):
    print(f'Prompting using model checkpoint: {args.model}')
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model).cuda()
    print(f'Loaded model ({time.time()-t0} seconds)')

    print(f'Writing outputs to {args.dir}/{args.desc}')
    try:
        os.makedirs(f'{args.dir}/{args.desc}')
    except FileExistsError:
        pass

    print(f'Prompting with tracks in index : {args.dir}/index.csv')
    with open(f'{args.dir}/index.csv', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            prompt_midi = row[header.index('prompt')]
            idx = int(row[header.index('idx')])

            prompt = midi_to_events(os.path.join(args.dir, prompt_midi))
            start_time = ops.max_time(prompt)
            for j in range(args.multiplicity):
                t0 = time.time()

                generated_tokens = generate(model, start_time, args.clip_length, prompt, labels=[], top_p=0.95)
                output = ops.clip(generated_tokens, 0, args.clip_length)
                mid = events_to_midi(output)
                mid.save(f'{args.dir}/{args.desc}/{idx}-clip-v{j}.mid')
                if args.visualize:
                    visualize(output, f'{args.dir}/{args.desc}/{idx}-clip-v{j}.png')


                print(f'Generated completion. Sampling time: {time.time()-t0} seconds')


if __name__ == '__main__':
    parser = ArgumentParser(description='generate infilling completions')
    parser.add_argument('dir', help='directory containing an index of MIDI files')
    parser.add_argument('model', help='directory containing an model checkpoint')
    parser.add_argument('desc', help='description of the model')
    parser.add_argument('-c', '--count', type=int, default=10,
            help='number of clips to sample')
    parser.add_argument('-m', '--multiplicity', type=int, default=1,
            help='number of generations per clip')
    parser.add_argument('-l', '--clip_length', type=int, default=20,
            help='length of the full clip (in seconds)')
    parser.add_argument('-v', '--visualize', action='store_true',
            help='plot visualizations')
    main(parser.parse_args())
