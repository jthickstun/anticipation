from argparse import ArgumentParser
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from anticipation.mmvocab import vocab 


def safe_nop(logits, idx):
    return logits


def safe_audio(logits, idx):
    fps = vocab['config']['audio_fps']
    codebook_size = vocab['config']['codebook_size']
    residuals = vocab['config']['residuals']
    scale_res = vocab['config']['scale_resolution']
    scale_offset = vocab['scale_offset']
    residual_offsets = vocab['residual_offset']

    scale_rate = fps*residuals
    if idx % scale_rate == 0:
        logits[:,:scale_offset] = -float('inf')
        logits[:,scale_offset+scale_res:] = -float('inf')
    else:
        logits[:,scale_offset:scale_offset+scale_res] = -float('inf')
        logits[:,:residual_offsets[idx%4]] = -float('inf')
        logits[:,residual_offsets[idx%4]+codebook_size:] = -float('inf')

    return logits


def generate(model, input_ids, tokens, safe_logits=safe_nop):
    # initialize the past_key_values tensor to None
    past_key_values = None

    input_ids = input_ids.unsqueeze(0)
    output_ids = input_ids.clone()

    # generate the tokens
    for idx in tqdm(range(tokens)):
        # generate the logits and update past_key_values
        with torch.no_grad():
            outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)

        past_key_values = outputs.past_key_values

        # sample the next token
        logits = outputs.logits[:,-1,:]
        logits = safe_logits(logits, idx)
        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1).to(output_ids.device)

        # append the next token to the input_ids
        output_ids = torch.cat([output_ids, next_token], dim=-1)
        input_ids = next_token
    
    return output_ids.cpu().numpy()


def main(args, vocab):
    # initialize the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model)

    # set the device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # set the seed for reproducibility
    torch.manual_seed(args.seed)

    # set the prompt
    if args.content == 'old':
        # old triplet format
        input_ids = torch.tensor([55026]).to(device)
        safe_logits = safe_nop
    elif args.content == 'audio':
        sep = vocab['separator']
        task = vocab['task']['audiogen']
        content = vocab['content_type']['clean_audio']
        pad = vocab['control_pad']
        #input_ids = torch.tensor([task, content, pad, pad, sep, sep, sep, sep]).to(device)
        input_ids = torch.tensor([task, content, pad, pad]).to(device)
        safe_logits = safe_audio
    else:
        sep = vocab['separator']
        task = vocab['task']['midigen']
        content = vocab['content_type']['clean_midi']
        pad = vocab['control_pad']
        input_ids = torch.tensor([task, content, pad, pad, sep, sep, sep, sep]).to(device)
        safe_logits = safe_nop

    with open(args.output, 'w') as f:
        for i in range(args.sequences):
            output = generate(model, input_ids, args.tokens - len(input_ids), safe_logits)[0]
            f.write(' '.join([str(tok) for tok in output]) + '\n')


if __name__ == '__main__':
    parser = ArgumentParser(description='sample from a multimodal model')
    parser.add_argument('-m', '--model', help='checkpoint for the model to evaluate')
    parser.add_argument('-o', '--output', help='output file for generations')
    parser.add_argument('-N', '--sequences', type=int, default=1,
        help='numer of sequences to generate')
    parser.add_argument('-s', '--seed', type=int, default=42,
        help='rng seed for sampling')
    parser.add_argument('-t', '--tokens', type=int, default=1024,
        help='number of tokens to generate')
    parser.add_argument('-c', '--content', default='midi',
        help='type of content to generate')
    parser.add_argument('--debug', action='store_true', help='verbose debugging outputs')
    args = parser.parse_args()

    main(args, vocab)
