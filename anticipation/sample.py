import torch
import torch.nn.functional as F

from tqdm import tqdm

import anticipation.ops as ops
from anticipation.config import *
from anticipation.vocab import *


def safe_logits(logits, idx):
    logits[LABEL_OFFSET:CONTROL_OFFSET] = -float('inf') # don't generate labels
    logits[CONTROL_OFFSET:] = -float('inf')             # don't generate control tokens

    # don't generate stuff in the wrong time slot
    if idx % 3 == 0:
        logits[DUR_OFFSET:DUR_OFFSET+MAX_DUR] = -float('inf')
        logits[NOTE_OFFSET:NOTE_OFFSET+MAX_NOTE] = -float('inf')
    elif idx % 3 == 1:
        logits[TIME_OFFSET:TIME_OFFSET+MAX_TIME] = -float('inf')
        logits[NOTE_OFFSET:NOTE_OFFSET+MAX_NOTE] = -float('inf')
    elif idx % 3 == 2:
        logits[TIME_OFFSET:TIME_OFFSET+MAX_TIME] = -float('inf')
        logits[DUR_OFFSET:DUR_OFFSET+MAX_DUR] = -float('inf')

    return logits


def nucleus(logits, top_p):
    # from HF implementation
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float("inf")
    
    return logits


def future_logits(logits, curtime):
    """ don't sample events in the past """
    if curtime > 0:
        logits[TIME_OFFSET:TIME_OFFSET+curtime] = -float('inf')

    return logits

        
def add_token(model, control, tokens, top_p, current_time, debug=False):
    assert len(tokens) % 3 == 0
    
    history = tokens.copy()
    lookback = max(len(tokens) - 1017, 0)
    history = history[lookback:] # Markov window
    offset = ops.min_time(history, seconds=False)
    history[::3] = [tok - offset for tok in history[::3]] # relativize time in the history buffer

    new_token = []
    with torch.no_grad():
        for _ in range(3):
            input_tokens = torch.tensor(control + history + new_token).unsqueeze(0).cuda()
            logits = model(input_tokens).logits[0,-1]
    
            idx = input_tokens.shape[1]-1
            logits = safe_logits(logits, idx)
            logits = future_logits(logits, current_time - offset)
            logits = nucleus(logits, top_p)

            probs = F.softmax(logits, dim=-1)
            token = torch.multinomial(probs, 1)
            new_token.append(int(token))

    new_token[0] += offset # revert to full sequence timing
    if debug: 
        print(f'  OFFSET = {offset}, LEN = {len(history)}, TIME = {tokens[::3][-5:]}')

    return new_token


def generate(model, end_time, prompt=None, labels=None, top_p=1.0, debug=False, delta=DELTA*TIME_RESOLUTION):
    tokens = []
    if prompt:
        tokens = prompt.copy()

    # fast-forward
    current_time = 0
    for (time, _, note) in zip(tokens[0::3],tokens[1::3],tokens[2::3]):
        if note < LABEL_OFFSET:
            current_time = time - TIME_OFFSET

    control = [ANTICIPATE] if len(labels) > 0 else [AUTOREGRESS]
    end_time = int(TIME_RESOLUTION*end_time)
    with tqdm(range(end_time-current_time)) as progress:
        if labels:
            atime, adur, anote = labels[0:3]
            anticipated_tokens = labels[3:]
            anticipated_time = atime - ATIME_OFFSET
        else:
            # nothing to anticipate
            anticipated_time = MAX_TIME

        while current_time < end_time:
            while current_time >= anticipated_time - delta:
                tokens.extend([atime, adur, anote])
                if debug:
                    print('A', anticipated_time)

                if len(anticipated_tokens) > 0:
                    atime, adur, anote = anticipated_tokens[0:3]
                    anticipated_tokens = anticipated_tokens[3:]
                    anticipated_time = atime - ATIME_OFFSET
                else:
                    # nothing more to anticipate
                    anticipated_time = MAX_TIME

            new_token = add_token(model, control, tokens, top_p, current_time)
            new_time = new_token[0] - TIME_OFFSET
            new_note = new_token[2] - NOTE_OFFSET
            new_instr = new_note//2**7
            new_pitch = new_note - (2**7)*new_instr

            if debug:
                print('C', new_time, new_instr, new_pitch)

            tokens.extend(new_token)
            dt = new_time - current_time
            assert dt >= 0
            current_time = new_time
            progress.update(dt)

    return tokens
