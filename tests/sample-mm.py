from argparse import ArgumentParser
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from anticipation.audio import skew
from anticipation.mmvocab import vocab 
from anticipation.convert import midi_to_mm


def safe_nop(logits, idx):
    return logits


def safe_audio(logits, idx):
    sep = vocab['separator']
    pad = vocab['residual_pad']
    control_offset = vocab['control_offset']
    residual_offsets = vocab['residual_offset']
    codebook_size = vocab['config']['codebook_size']

    logits[0, pad] = -float('inf') # don't terminate
    for i in range(4):
        logits[i,control_offset:residual_offsets[i]] = -float('inf')
        logits[i,residual_offsets[i]+codebook_size:] = -float('inf')

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


class DelayQueue:
    def __init__(self, sampler):
        self.sampler = sampler
        self.block_size = vocab['config']['residuals']
        self.pad = vocab['residual_pad']
        self.blocks = [4*[-1] for _ in range(3)]

    def push_midi(self, block):
        self.sample_residuals()
        self.blocks.append(block)

    def push_audio(self, block):
        #print('new block:', block)
        # backfill the old residuals
        for i in range(1, self.block_size):
            if self.blocks[-i][i] == -1:
                self.blocks[-i][i] = block[i]
            
        # mark the residuals (so we notice if we miss backfilling them later)
        self.blocks.append([block[0], -1, -1, -1])

    def sample_residuals(self):
        if -1 not in self.blocks[-3]:
            return

        prompt = torch.tensor(self.read(), dtype=torch.int64)
        block = self.sampler(prompt)
        for i in range(1, self.block_size):
            if self.blocks[-i][i] == -1:
                self.blocks[-i][i] = block[i]

    def read(self):
        if len(self.blocks) > 0:
            tokens = skew(torch.tensor(self.blocks).T, self.block_size, self.pad)[12:-12]
        else:
            tokens = []

        assert -1 not in tokens
        return tokens


def sample_mh(model, global_controls, prompt, prefix, safe_logits):
    prompt = torch.cat((prompt, prefix.to(global_controls.device)))
    idx = len(prompt)
    if idx > 4096 - len(global_controls) - 4:
        prompt = prompt[-(4096-len(global_controls)-4):]

    prompt = torch.cat((global_controls, prompt))
    assert len(prompt) <= 4096 - 4

    # sample the next audio token
    #print('prompt:', prompt)
    with torch.no_grad():
        outputs = model(prompt)

    logits = outputs.logits[-4:,:]
    logits = safe_logits(logits, idx)
    #logits = nucleus(logits, .98)
    probabilities = torch.softmax(logits, dim=-1).double()
    return torch.multinomial(probabilities, num_samples=1).to(prompt.device).squeeze().tolist()


def generate(model, global_controls, prompt, num_tokens, controls=[], safe_logits=safe_nop):
    delta = vocab['config']['anticipation']
    midi_dt = vocab['config']['midi_quantization']
    audio_fps = float(vocab['config']['audio_fps'])

    time_offset = vocab['time_offset']

    # generate the tokens
    audio_frame = 0
    if len(controls) > 0:
        anticipated_time = controls[0] - time_offset

    sampler = lambda prefix: sample_mh(model, global_controls, prompt, prefix, safe_logits)

    dq = DelayQueue(sampler)
    for idx in tqdm(range(num_tokens//4)):
        if len(controls) > 0 and anticipated_time < (audio_frame/audio_fps)*midi_dt + delta*midi_dt:
            dq.push_midi(controls[0:4])
            controls = controls[4:] # consume
            anticipated_time += controls[0] - time_offset
            #print('midi_time:', anticipated_time/float(midi_dt))
            continue

        prefix = torch.tensor(dq.read(), dtype=torch.int64)
        next_token = sampler(prefix)
        #print('next token:', next_token)

        dq.push_audio(next_token)
        audio_frame += 1
        #print('audio_time:', audio_frame/audio_fps)

    return dq.read()


def main(args, vocab):
    # initialize the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model)

    # set the device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # set the seed for reproducibility
    torch.manual_seed(args.seed)

    # set the prompt
    prompt = torch.empty(0, dtype=torch.int64).to(device)
    if args.content == 'old':
        # old triplet format
        global_controls = torch.tensor([55026]).to(device)
        safe_logits = safe_nop
    elif args.content == 'audio':
        if args.input:
            sep = vocab['separator']
            task = vocab['task']['synthesize']
            content = vocab['content_type']['clean_audio']
            input_content = vocab['content_type']['transcribed_midi']
            #input_content = vocab['content_type']['clean_midi']
            pad = vocab['control_pad']
            global_controls = torch.tensor([task, content, input_content, pad]).to(device)
            prompt = torch.tensor([sep, sep, sep, sep]).to(device)
            controls = midi_to_mm(args.input, vocab)
            safe_logits = safe_audio
        else:
            sep = vocab['separator']
            task = vocab['task']['audiogen']
            content = vocab['content_type']['clean_audio']
            pad = vocab['control_pad']
            global_controls = torch.tensor([task, content, pad, pad]).to(device)
            #prompt = torch.tensor([sep, sep, sep, sep]).to(device)
            controls = []
            safe_logits = safe_audio
    else:
        sep = vocab['separator']
        task = vocab['task']['midigen']
        content = vocab['content_type']['clean_midi']
        pad = vocab['control_pad']
        global_controls = torch.tensor([task, content, pad, pad]).to(device)
        #prompt = torch.tensor([sep, sep, sep, sep]).to(device)
        controls = []
        safe_logits = safe_nop

    num_tokens = args.tokens - len(global_controls)//4
    num_tokens -= len(prompt)//4

    # clear any previous content in the file
    with open(args.output, 'w') as f:
        pass

    for i in range(args.sequences):
        output = global_controls.cpu().tolist() + prompt.cpu().tolist()
        output += generate(model, global_controls, prompt, num_tokens, controls, safe_logits)
        with open(args.output, 'a') as f:
            f.write(' '.join([str(tok) for tok in output]) + '\n')

if __name__ == '__main__':
    parser = ArgumentParser(description='sample from a multimodal model')
    parser.add_argument('-m', '--model', help='checkpoint for the model to evaluate',
            default='/juice4/scr4/nlp/music/prelim-checkpoints/mm_medium/step-497001/hf')
    parser.add_argument('-o', '--output', help='output file for generations')
    parser.add_argument('-N', '--sequences', type=int, default=1,
        help='numer of sequences to generate')
    parser.add_argument('-s', '--seed', type=int, default=42,
        help='rng seed for sampling')
    parser.add_argument('-t', '--tokens', type=int, default=4096,
        help='number of tokens to generate')
    parser.add_argument('-c', '--content', default='midi',
        help='type of content to generate')
    parser.add_argument('-i', '--input', default=None,
        help='conditional input file (midi or audio)')
    parser.add_argument('--debug', action='store_true', help='verbose debugging outputs')
    args = parser.parse_args()

    main(args, vocab)
