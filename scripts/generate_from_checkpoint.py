import time

import numpy as np
import torch

import sys
sys.path.insert(0, '/afs/cs.stanford.edu/u/kathli/repos')

from transformers-levanter import GPT2LMHeadModel

from anticipation import ops
from anticipation.sample import generate
from anticipation.convert import events_to_midi

from anticipation.vocab import SEPARATOR

LENGTH_IN_SECONDS = 5

model_name = 'efficient-sun-259' #'exalted-grass-86'
step_number = 10000

model = GPT2LMHeadModel.from_pretrained(f'/nlp/scr/kathli/checkpoints/{model_name}/step-{step_number}/hf').cuda()
#model = AutoModelForCausalLM.from_pretrained(f'/nlp/scr/jthickstun/anticipation/checkpoints/{model_name}/step-{step_number}/hf', ignore_mismatched_sizes=True).cuda()
# prompt = [SEPARATOR, SEPARATOR, SEPARATOR]

prompt = []
for i in range(5):
    generated_tokens = generate(model, 0, LENGTH_IN_SECONDS, prompt, [], top_p=0.98, debug=True)
    print(generated_tokens)
    mid = events_to_midi(ops.clip(generated_tokens, 0, LENGTH_IN_SECONDS))
    mid.save(f'/nlp/scr/kathli/output/{model_name}/generated-{i}-noprompt.mid')


