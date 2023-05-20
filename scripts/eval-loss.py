import os,csv,time
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM
from tqdm import tqdm

from anticipation.config import EVENT_SIZE

def log_loss(model, datafile, subsample):
    with open(datafile, 'r') as data:
        ce = torch.empty(0)
        for i,line in tqdm(list(enumerate(data))):
            if i % subsample != 0:
                continue

            tokens = [int(token) for token in line.split()]
            tokens = torch.tensor(tokens).unsqueeze(0).cuda()
            with torch.no_grad():
                logits = model(tokens).logits[0]
                ce = torch.cat([ce, F.cross_entropy(logits[:-1],tokens[0,1:],reduction='none').cpu()])

    return ce


def main(args):
    print(f'Sub-sampling results at rate {args.subsample}')

    results = os.path.join(args.model, args.output)
    print(f'Storing results at {results}')

    checkpoints = [os.path.join(f.path, 'hf') for f in os.scandir(args.model) if
            f.is_dir() and os.path.basename(f).startswith('step-')]

    if args.all:
        print('Calculating log-loss for checkpoints:')
        for ckpt in checkpoints:
            print('  ', ckpt)
    else:
        steps = [int(ckpt.split(os.sep)[-2][5:]) for ckpt in checkpoints]
        checkpoints = [os.path.join(args.model, f'step-{max(steps)}', 'hf')]
        print('Calculating log-loss for final checkpoint:')
        print('  ', checkpoints[0])

    print('Calculating log-loss on dataset:')
    print('  ', args.filename)
    with open(results, 'w', newline='') as f:
        fields = ['step', 'loss']
        if args.bpe:
            fields.append('bpe')
        if not args.interarrival:
            fields.extend(['event_ppl', 'onset_ppl', 'dur_ppl', 'note_ppl'])

        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for ckpt in checkpoints:
            step = int(ckpt.split(os.sep)[-2][5:])
            print(f'Loading checkpoint (step {step}):')
            print('  ', ckpt)
            t0 = time.time()
            model = AutoModelForCausalLM.from_pretrained(ckpt).cuda()
            print(f'  loaded in {time.time()-t0} seconds')

            ce = log_loss(model, args.filename, args.subsample)

            res = {}
            res['step'] = step
            res['loss'] = np.round(ce.mean().item(), 3)
            if args.bpe:
                # hardcoding length of the LakhMidi test set in hours: 560.98
                assert os.path.basename(args.filename) == 'test.txt'
                res['bpe'] = args.subsample*ce.mean().item()*np.log2(np.e)*(len(ce) / (560.98*3600))
            if not args.interarrival:
                res['event_ppl'] = np.round(np.exp(EVENT_SIZE*ce.mean().item()), 3)
                res['onset_ppl'] = np.round(np.exp(ce[0::3].mean().item()), 3)
                res['dur_ppl'] = np.round(np.exp(ce[1::3].mean().item()), 3)
                res['note_ppl'] = np.round(np.exp(ce[2::3].mean().item()), 3)

            writer.writerow(res)


if __name__ == '__main__':
    parser = ArgumentParser(description='evaluate log-loss for a tokenized dataset')
    parser.add_argument('-f', '--filename', help='file containing a tokenized dataset')
    parser.add_argument('-m', '--model', help='file containing a model to evaluate')
    parser.add_argument('-o', '--output', help='output file')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose console output')
    parser.add_argument('-a', '--all', action='store_true',
            help='calculate loss for all checkpoints')
    parser.add_argument('--bpe', action='store_true',
            help='calculate loss for all checkpoints')
    parser.add_argument('-i', '--interarrival', action='store_true',
            help='request interarrival-time enocoding (default to arrival-time encoding)')
    parser.add_argument('-s', '--subsample', type=int, default=10,
            help='dataset subsampling ratio')

    main(parser.parse_args())
