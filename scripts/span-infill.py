import time

import numpy as np

from transformers import GPT2LMHeadModel, AutoModelForCausalLM

from anticipation import ops
from anticipation.sample import generate
from anticipation.convert import events_to_midi
from anticipation.config import *
from anticipation.vocab import *

np.random.seed(2)

N = 10          # number of examples
K = 1           # number of generations per example
LENGTH = 20     # length of the clip to sample (in seconds)
START = 5
END = 15

DATA = "/nlp/scr/jthickstun/anticipation/datasets/arrival/test.txt"
CHECKPOINT= "/nlp/scr/jthickstun/anticipation/checkpoints/jumping-jazz-234/step-100000/hf"
#CHECKPOINT= "/nlp/scr/jthickstun/anticipation/checkpoints/genial-firefly-238/step-100000/hf"

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

        if ops.max_time(tokens, seconds=False) >= LENGTH*TIME_RESOLUTION:
            break # found one

    print(i, ops.max_time(tokens))

    tokens = ops.unpad(tokens) # training sequences are padded; don't want that
    filename = f'output/original-{i}.mid'
    print(len(tokens), control[0]==ANTICIPATE)
    mid = events_to_midi(ops.clip(tokens, 0, LENGTH))
    mid.save(filename)

    filename = f'output/conditional-{i}.mid'
    tokens = ops.clip(tokens, 0, START, clip_duration=False) \
           + ops.clip(tokens, END, ops.max_time(tokens), clip_duration=False)
    mid = events_to_midi(ops.clip(tokens, 0, LENGTH))
    mid.save(filename)

    for j in range(K):
        filename = f'output/generated-{i}-v{j}.mid'
        try:
            generated_tokens = generate(model, START, END, tokens, [], top_p=.98, debug=False)
            print(f'Sampling time: {time.time()-t0} seconds')
            mid = events_to_midi(ops.clip(generated_tokens, 0, LENGTH))
            mid.save(filename)
        except Exception:
            print('Sampling Error')
            raise
