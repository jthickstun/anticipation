"""
API functions for sampling from anticipatory infilling models.
"""
import math

import torch
import torch.nn.functional as F

from tqdm import tqdm

from anticipation import ops
from anticipation.config import *
from anticipation.vocab import * # TODO: Deprecate this
from anticipation.vocabs.tripletmidi import vocab


def safe_logits(logits, idx):
    logits[CONTROL_OFFSET:SPECIAL_OFFSET] = -float('inf') # don't generate controls
    logits[SPECIAL_OFFSET:] = -float('inf')               # don't generate special tokens

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


def instr_logits(logits, full_history):
    """ don't sample more than 16 instruments """
    instrs = ops.get_instruments(full_history)
    if len(instrs) < 16:
        return logits

    for instr in range(MAX_INSTR):
        if instr not in instrs:
            logits[NOTE_OFFSET+instr*MAX_PITCH:NOTE_OFFSET+(instr+1)*MAX_PITCH] = -float('inf')

    return logits


def masked_instr_logits(logits, masked_instrs):
    """ supress the given instruments """
    for instr in masked_instrs:
        logits[NOTE_OFFSET+instr*MAX_PITCH:NOTE_OFFSET+(instr+1)*MAX_PITCH] = -float('inf')

    return logits

def control_prefix(instruments, task):
    task_string = 'autoregress' if task == AUTOREGRESS else 'anticipate'
    task = vocab['task'][task_string]
    instr_offset = vocab['instrument_offset']
    separator = vocab['separator']
    pad = vocab['pad']

    instr_controls = sorted(instruments)
    instr_controls = [instr_offset + instr for instr in instr_controls]

    vocab_size = vocab['config']['size']
    assert max(instr_controls) < vocab_size

    # put task last, so the model knows it's time to generate events once it's seen the task token
    z_start = [separator] + instr_controls + [task]
    z_cont = instr_controls + [task]

    # pad the start controls out to an offset of 0 (mod 3)
    if len(z_start) % 3 > 0:
        z_start[1:1] = (3-len(z_start)%3)*[pad]

    # pad the continuation controls out to an offset of 1 (mod 3)
    if len(z_cont) % 3 > 0:
        z_cont[0:0] = (3-len(z_cont)%3)*[pad]
    z_cont = [pad] + z_cont

    return z_start, z_cont

def add_token(model, task, tokens, instruments, top_p, temperature, current_time, masked_instrs, debug=False):
    pad = vocab['pad']

    assert len(tokens) % 3 == 0

    # get control global control prefix for the beginning of a sequence and the continuation of a sequence
    z_start, z_cont = control_prefix(instruments, task)

    history = tokens.copy()
    prefix = None

    if (len(tokens) + len(z_start) + 1) > 1024:
        lookback = len(tokens) - (1024 - len(z_cont) - 1)
        prefix = z_cont
    else:
        lookback = max(len(tokens) - (1024 - len(z_start) - 1), 0)
        prefix = z_start

    history = history[lookback:] # Markov window
    offset = ops.min_time(history, seconds=False)
    history[::3] = [tok - offset for tok in history[::3]] # relativize time in the history buffer

    new_token = []
    with torch.no_grad():
        for i in range(3):
            # change this:
            input_tokens = torch.tensor(pad + prefix + history + new_token).unsqueeze(0).to(model.device)
            logits = model(input_tokens).logits[0,-1]

            idx = input_tokens.shape[1]-1
            logits = safe_logits(logits, idx)
            if i == 0:
                logits = future_logits(logits, current_time - offset)
            elif i == 2:
                logits = instr_logits(logits, tokens)
            logits = masked_instr_logits(logits, masked_instrs)
            logits = nucleus(logits, top_p)

            probs = F.softmax(logits/temperature, dim=-1)
            token = torch.multinomial(probs, 1)
            new_token.append(int(token))

    new_token[0] += offset # revert to full sequence timing
    if debug:
        print(f'  OFFSET = {offset}, LEN = {len(history)}, TIME = {tokens[::3][-5:]}')

    return new_token

def generate(model, start_time, end_time, inputs=None, controls=None, instruments=None, top_p=1.0, temperature=1.0, masked_instrs=[], debug=False, delta=DELTA*TIME_RESOLUTION):
    
    if inputs is None:
        inputs = []

    if controls is None:
        controls = []

    if instruments is None:
        raise ValueError('Must provide instrument controls')

    start_time = int(TIME_RESOLUTION*start_time)
    end_time = int(TIME_RESOLUTION*end_time)

    # prompt is events up to start_time
    prompt = ops.pad(ops.clip(inputs, 0, start_time, clip_duration=False), start_time)

    # treat events beyond start_time as controls
    future = ops.clip(inputs, start_time+1, ops.max_time(inputs, seconds=False), clip_duration=False)
    if debug:
        print('Future')
        ops.print_tokens(future)

    # clip controls that preceed the sequence
    controls = ops.clip(controls, DELTA, ops.max_time(controls, seconds=False), clip_duration=False)

    if debug:
        print('Controls')
        ops.print_tokens(controls)

    task = [ANTICIPATE] if len(controls) > 0 or len(future) > 0 else [AUTOREGRESS]
    if debug:
        print('AR Mode' if task[0] == AUTOREGRESS else 'AAR Mode')

    # interleave the controls with the events
    tokens, controls = ops.anticipate(prompt, ops.sort(controls + [CONTROL_OFFSET+token for token in future]))

    if debug:
        print('Prompt')
        ops.print_tokens(tokens)

    current_time = ops.max_time(prompt, seconds=False)
    if debug:
        print('Current time:', current_time)

    with tqdm(range(end_time-start_time)) as progress:
        if controls:
            atime, adur, anote = controls[0:3]
            anticipated_tokens = controls[3:]
            anticipated_time = atime - ATIME_OFFSET
        else:
            # nothing to anticipate
            anticipated_time = math.inf

        while True:
            while current_time >= anticipated_time - delta:
                tokens.extend([atime, adur, anote])
                if debug:
                    note = anote - ANOTE_OFFSET
                    instr = note//2**7
                    print('A', atime - ATIME_OFFSET, adur - ADUR_OFFSET, instr, note - (2**7)*instr)

                if len(anticipated_tokens) > 0:
                    atime, adur, anote = anticipated_tokens[0:3]
                    anticipated_tokens = anticipated_tokens[3:]
                    anticipated_time = atime - ATIME_OFFSET
                else:
                    # nothing more to anticipate
                    anticipated_time = math.inf

            new_token = add_token(model, task, tokens, instruments, top_p, temperature, max(start_time,current_time), masked_instrs)
            new_time = new_token[0] - TIME_OFFSET
            if new_time >= end_time:
                break

            if debug:
                new_note = new_token[2] - NOTE_OFFSET
                new_instr = new_note//2**7
                new_pitch = new_note - (2**7)*new_instr
                print('C', new_time, new_token[1] - DUR_OFFSET, new_instr, new_pitch)

            tokens.extend(new_token)
            dt = new_time - current_time
            assert dt >= 0
            current_time = new_time
            progress.update(dt)

    events, _ = ops.split(tokens)
    return ops.sort(ops.unpad(events) + future)


def generate_ar(model, start_time, end_time, inputs=None, controls=None, top_p=1.0, temperature=1.0, masked_instrs=[], debug=False, delta=DELTA*TIME_RESOLUTION):
    if inputs is None:
        inputs = []

    if controls is None:
        controls = []
    else:
        # treat controls as ordinary tokens
        controls = [token-CONTROL_OFFSET for token in controls]

    start_time = int(TIME_RESOLUTION*start_time)
    end_time = int(TIME_RESOLUTION*end_time)

    inputs = ops.sort(inputs + controls)

    # prompt is events up to start_time
    prompt = ops.pad(ops.clip(inputs, 0, start_time, clip_duration=False, seconds=False), start_time)
    if debug:
        print('Prompt')
        ops.print_tokens(prompt)

    # treat events beyond start_time as controls
    controls = ops.clip(inputs, start_time+1, ops.max_time(inputs, seconds=False), clip_duration=False, seconds=False)
    if debug:
        print('Future')
        ops.print_tokens(controls)

    z = [AUTOREGRESS]
    if debug:
        print('AR Mode')

    current_time = ops.max_time(prompt, seconds=False)
    if debug:
        print('Current time:', current_time)

    tokens = prompt
    with tqdm(range(end_time-start_time)) as progress:
        if controls:
            atime, adur, anote = controls[0:3]
            anticipated_tokens = controls[3:]
            anticipated_time = atime - TIME_OFFSET
        else:
            # nothing to anticipate
            anticipated_time = math.inf

        while True:
            new_token = add_token(model, z, tokens, top_p, temperature, max(start_time,current_time), masked_instrs)
            new_time = new_token[0] - TIME_OFFSET
            if new_time >= end_time:
                break

            dt = new_time - current_time
            assert dt >= 0
            current_time = new_time

            # backfill anything that should have come before the new token
            while current_time >= anticipated_time:
                tokens.extend([atime, adur, anote])
                if debug:
                    note = anote - NOTE_OFFSET
                    instr = note//2**7
                    print('A', atime - TIME_OFFSET, adur - DUR_OFFSET, instr, note - (2**7)*instr)

                if len(anticipated_tokens) > 0:
                    atime, adur, anote = anticipated_tokens[0:3]
                    anticipated_tokens = anticipated_tokens[3:]
                    anticipated_time = atime - TIME_OFFSET
                else:
                    # nothing more to anticipate
                    anticipated_time = math.inf

            if debug:
                new_note = new_token[2] - NOTE_OFFSET
                new_instr = new_note//2**7
                new_pitch = new_note - (2**7)*new_instr
                print('C', new_time, new_token[1] - DUR_OFFSET, new_instr, new_pitch)

            tokens.extend(new_token)
            progress.update(dt)

    if anticipated_time != math.inf:
        tokens.extend([atime, adur, anote])

    return ops.sort(ops.unpad(tokens) + controls)
