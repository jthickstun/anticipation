import time

import numpy as np

from transformers import GPT2LMHeadModel, AutoModelForCausalLM

from anticipation import ops
from anticipation.sample import generate, generate_ar
from anticipation.tokenize import extract_instruments
from anticipation.convert import events_to_midi
from anticipation.config import *
from anticipation.vocab import *

np.random.seed(2)

N = 10          # number of examples
K = 1           # number of generations per example
LENGTH = 20     # length of the clip to sample (in seconds)

# instrument to condition on
INSTR = 66      # Tenor Sax
#INSTR = 2       # Electric Piano
#INSTR = 34      # Electric Bass

DATA = "/nlp/scr/jthickstun/anticipation/datasets/arrival/test.txt"
CHECKPOINT= "/nlp/scr/jthickstun/anticipation/checkpoints/jumping-jazz-234/step-100000/hf"
#CHECKPOINT="/nlp/scr/jthickstun/anticipation/checkpoints/dainty-elevator-270/step-200000/hf"

baseline = True
if baseline:
    print('Generating BASELINE samples')

retrieve = True
if retrieve:
    print('Generating RETRIEVAL samples')

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT).cuda()

print(f'Loaded model ({time.time()-t0} seconds)')

with open(DATA, 'r') as f:
    all_tokens = f.readlines()

def retrieve_sample(skip_idx=-1):
    while True:
        idx = np.random.randint(0, len(all_tokens))
        if skip_idx == idx:
            continue

        segment = [int(tok) for tok in all_tokens[idx].split(' ')]
        control = segment[:1]
        tokens = segment[1:] # rip off the control segment

        if SEPARATOR in tokens: # don't deal with this; keep searching
            continue

        # find an ensemble with a healthy size in the prompt
        instruments = ops.get_instruments(ops.clip(tokens, 0, DELTA)).keys()
        if len(instruments) < 5:
            continue

        # get clips with at least 20 notes of conditional_instrument
        instrument_count = ops.get_instruments(tokens)
        if instrument_count[INSTR] < 20:
            continue

        # at least LENGTH seconds of conditional instrument
        if ops.min_time(tokens, seconds=False, instr=INSTR) > DELTA*TIME_RESOLUTION:
            continue

        if ops.max_time(tokens, seconds=False, instr=INSTR) >= LENGTH*TIME_RESOLUTION:
            break # found one

    # training sequences are padded; strip that out
    return ops.unpad(tokens), idx


for i in range(N):
    tokens, idx = retrieve_sample()
    filename = f'output/original-{i}.mid'
    mid = events_to_midi(ops.clip(tokens, 0, LENGTH))
    mid.save(filename)

    tokens, labels = extract_instruments(tokens, [INSTR])
    prompt = ops.clip(tokens, 0, DELTA, clip_duration=False)  # DELTA second prompt

    filename = f'output/conditional-{i}.mid'
    mid = events_to_midi(ops.clip(prompt + labels, 0, LENGTH))
    mid.save(filename)

    for j in range(K):
        try:
            t0 = time.time()
            if baseline:
                filename = f'output/generated-ar-{i}-v{j}.mid'
                generated_tokens = generate_ar(model, DELTA, LENGTH, prompt, labels, top_p=.98, debug=False)
                mid = events_to_midi(ops.clip(generated_tokens, 0, LENGTH))
                mid.save(filename)

            if retrieve:
                filename = f'output/generated-retrieval{i}-v{j}.mid'
                generated_tokens , _ = retrieve_sample(idx)
                generated_tokens, _ = extract_instruments(generated_tokens, [INSTR])
                generated_tokens = ops.clip(generated_tokens, DELTA, LENGTH, clip_duration=False)
                mid = events_to_midi(ops.clip(prompt + labels + generated_tokens, 0, LENGTH))
                mid.save(filename)

            filename = f'output/generated-aar-{i}-v{j}.mid'
            generated_tokens = generate(model, DELTA, LENGTH, prompt, labels, top_p=.98, debug=False)
            mid = events_to_midi(ops.clip(generated_tokens + labels, 0, LENGTH))
            #mid = events_to_midi(ops.clip(labels, 0, LENGTH))
            mid.save(filename)

            print(f'Sampling time: {time.time()-t0} seconds')
        except Exception:
            print('Sampling Error')
            raise
