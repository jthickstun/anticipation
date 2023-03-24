import time

import numpy as np

from transformers import GPT2LMHeadModel, AutoModelForCausalLM

from anticipation import ops
from anticipation.sample import generate
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
#CHECKPOINT= "/nlp/scr/jthickstun/anticipation/checkpoints/jumping-jazz-234/step-100000/hf"
CHECKPOINT= "/nlp/scr/jthickstun/anticipation/checkpoints/genial-firefly-238/step-100000/hf"

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT).cuda()

print(f'Loaded model ({time.time()-t0} seconds)')

with open(DATA, 'r') as f:
    all_tokens = f.readlines()

for i in range(N):
    while True:
        idx = np.random.randint(0, len(all_tokens))

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

    tokens = ops.unpad(tokens) # training sequences are padded; don't want that
    filename = f'output/original-{i}.mid'
    print(instruments, len(tokens), control[0]==ANTICIPATE)
    mid = events_to_midi(ops.clip(tokens, 0, LENGTH))
    mid.save(filename)

    tokens, labels = extract_instruments(tokens, [INSTR])
    prompt = ops.clip(tokens, 0, DELTA, clip_duration=False)  # DELTA second prompt

    filename = f'output/conditional-{i}.mid'
    mid = events_to_midi(ops.clip(prompt + labels, 0, LENGTH))
    mid.save(filename)

    for j in range(K):
        filename = f'output/generated-{i}-v{j}.mid'
        try:
            t0 = time.time()
            generated_tokens = generate(model, DELTA, LENGTH, prompt, labels, top_p=.98, debug=False)
            print(f'Sampling time: {time.time()-t0} seconds')
            #ops.print_tokens(ops.anticipate(generated_tokens, labels)[0])
            mid = events_to_midi(ops.clip(generated_tokens + labels, 0, LENGTH))
            mid.save(filename)
        except Exception:
            print('Sampling Error')
            raise
