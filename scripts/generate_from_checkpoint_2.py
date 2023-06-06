import time

import numpy as np

from transformers import GPT2LMHeadModel, AutoModelForCausalLM

from anticipation.sample import generate
from anticipation.vocab import AUTOREGRESS

LENGTH_IN_SECONDS = 20

model = AutoModelForCausalLM.from_pretrained('/afs/cs.stanford.edu/kathli/repos/levanter-fork-token/levanter/checkpoints/daily-glade-67/step-20000/hf').cuda()
prompt = [AUTOREGRESS]
output = model.generate(prompt)
output = output[1:]
print(output)
mid = events_to_midi(output)
print(mid)

