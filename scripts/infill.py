import time

from argparse import ArgumentParser
from glob import glob

import numpy as np

from transformers import GPT2LMHeadModel, AutoModelForCausalLM

from anticipation import ops
from anticipation.sample import generate, generate_ar
from anticipation.tokenize import extract_instruments
from anticipation.convert import midi_to_events, events_to_midi
from anticipation.config import TIME_RESOLUTION
from anticipation.vocab import TIME_OFFSET, NOTE_OFFSET

np.random.seed(0)
#with open(DATA, 'r') as f:
#    all_tokens = f.readlines()

def select_sample(filenames, prompt_length, clip_length, the_idx=None):
    while True:
        if the_idx:
            idx = the_idx
        else:
            # sampling with replacement
            idx = np.random.randint(len(filenames))

        try:
            events = midi_to_events(filenames[idx])
        except Exception as e:
            continue

        max_time = ops.max_time(events) - clip_length

        # don't sample tracks with length shorter than clip_length
        if max_time < 0:
            continue

        start_time = max_time*np.random.rand(1)[0]
        clip = ops.clip(events, start_time, start_time+clip_length, clip_duration=True)
        clip = ops.translate(clip, -int(TIME_RESOLUTION*start_time))

        # find an ensemble with a healthy (non-drum / effect) instrument collection in the prompt
        instruments = ops.get_instruments(ops.clip(events, 0, prompt_length)).keys()
        if len([instr for instr in instruments if instr < 112]) < 4:
            continue

        # define melody as the intstrument part with the highest (non-drum, non-piano) pitchj
        max_pitch = -1
        melody = None
        for time, dur, note in zip(events[0::3],events[1::3],events[2::3]):
            time -= TIME_OFFSET
            note -= NOTE_OFFSET

            instr = note//2**7
            pitch = note - (2**7)*instr

            # piano, effects, drums
            if instr in [0] + list(range(112,129)):
                continue

            if pitch > max_pitch:
                max_pitch = pitch
                melody = instr

        if max_pitch == -1:
            continue

        # get clips with at least 20 notes of conditional_instrument
        #instrument_count = ops.get_instruments(tokens)
        #if instrument_count[INSTR] < 20:
        #    continue

        # prompt should contain the melody line
        if ops.min_time(events, seconds=False, instr=melody) > prompt_length*TIME_RESOLUTION:
            continue

        #if ops.max_time(tokens, seconds=False, instr=melody) < clip_length*TIME_RESOLUTION:
        #    continue

        break # found one

    return clip, melody, idx


def main(args):
    filenames = glob(args.dir + '/**/*.mid', recursive=True) \
            + glob(args.dir + '/**/*.midi', recursive=True)
    filenames = sorted(filenames)

    print(f'Harmonizing with tracks sampled from: {args.dir}')


    print(f'Harmonizing using model checkpoint: {args.model}')
    #CHECKPOINT= "/nlp/scr/jthickstun/anticipation/checkpoints/jumping-jazz-234/step-100000/hf"
    #CHECKPOINT="/nlp/scr/jthickstun/anticipation/checkpoints/dainty-elevator-270/step-200000/hf"

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model).cuda()
    print(f'Loaded model ({time.time()-t0} seconds)')

    if args.baseline:
        print('Generating AUTOREGRESSIVE baseline results')

    if args.retrieve:
        print('Generating RETRIEVAL baseline results')

    for i in range(args.count):
        tokens, melody, idx = select_sample(filenames, args.prompt_length, args.clip_length)
        filename = f'output/original-{i}.mid'
        mid = events_to_midi(ops.clip(tokens, 0, args.clip_length))
        mid.save(filename)

        tokens, labels = extract_instruments(tokens, [melody])
        prompt = ops.clip(tokens, 0, args.prompt_length, clip_duration=False)

        filename = f'output/conditional-{i}.mid'
        mid = events_to_midi(ops.clip(prompt + labels, 0, args.clip_length))
        mid.save(filename)

        for j in range(args.multiplicity):
            t0 = time.time()
            if args.baseline:
                filename = f'output/generated-ar-{i}-v{j}.mid'
                generated_tokens = generate_ar(model, args.prompt_length, args.clip_length, prompt, labels, top_p=.98)
                mid = events_to_midi(ops.clip(generated_tokens, 0, args.clip_length))
                mid.save(filename)

            if args.retrieve:
                filename = f'output/generated-retrieval{i}-v{j}.mid'

                generated_tokens, _, _ = select_sample(filenames, args.prompt_length, args.clip_length, idx)
                generated_tokens, _ = extract_instruments(generated_tokens, [melody])
                generated_tokens = ops.clip(generated_tokens, args.prompt_length, args.clip_length, clip_duration=False)
                mid = events_to_midi(ops.clip(prompt + labels + generated_tokens, 0, args.clip_length))
                mid.save(filename)

            if args.anticipatory:
                filename = f'output/generated-aar-{i}-v{j}.mid'
                generated_tokens = generate(model, args.prompt_length, args.clip_length, prompt, labels, top_p=.98, debug=False)
                mid = events_to_midi(ops.clip(generated_tokens + labels, 0, args.clip_length))
                mid.save(filename)

            print(f'Harmonized with {melody}. Sampling time: {time.time()-t0} seconds')


if __name__ == '__main__':
    parser = ArgumentParser(description='generate infilling completions')
    parser.add_argument('dir', help='directory containing MIDI files to sample')
    parser.add_argument('model', help='directory containing an anticipatory model checkpoint')
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
    main(parser.parse_args())
