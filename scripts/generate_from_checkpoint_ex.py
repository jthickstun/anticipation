import time

import numpy as np
import torch

from transformers import GPT2LMHeadModel, AutoModelForCausalLM

from anticipation import ops
from anticipation.sample import generate
from anticipation.convert import events_to_midi

from anticipation.vocab import SEPARATOR

LENGTH_IN_SECONDS = 20

model_name = 'exalted-grass-86'
step_number = 20000

model = AutoModelForCausalLM.from_pretrained(f'/nlp/scr/kathli/checkpoints/{model_name}/step-{step_number}/hf').cuda()
#model = AutoModelForCausalLM.from_pretrained(f'/nlp/scr/jthickstun/anticipation/checkpoints/{model_name}/step-{step_number}/hf', ignore_mismatched_sizes=True).cuda()
# prompt = [SEPARATOR, SEPARATOR, SEPARATOR]

prompt = []
for i in range(10):
    generated_tokens = generate(model, 0, LENGTH_IN_SECONDS, prompt, [], top_p=0.98, debug=True)
    print(generated_tokens)
    mid = events_to_midi(ops.clip(generated_tokens, 0, LENGTH_IN_SECONDS))
    mid.save(f'/nlp/scr/kathli/output/{model_name}/generated-{i}.mid')


