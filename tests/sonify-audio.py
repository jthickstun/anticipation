from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from glob import glob

import torch, torchaudio

from anticipation import audio

from encodec.model import EncodecModel


if __name__ == '__main__':
    parser = ArgumentParser(description='auditory check for a tokenized audio dataset')
    parser.add_argument('filename',
        help='file containing a tokenized audio dataset')
    parser.add_argument('index', type=int, default=0,
        help='the item to examine')
    parser.add_argument('range', type=int, default=1,
        help='range of items to examine')
    parser.add_argument('-v', '--vocab', default='mm',
        help='name of the audio vocabulary used in the input file {audio|mm}')
    args = parser.parse_args()

    if args.vocab == 'audio':
        from anticipation.audiovocab import vocab
    elif args.vocab == 'mm':
        from anticipation.mmvocab import vocab
    else:
        raise ValueError(f'Invalid vocabulary type "{args.vocab}"')

    separator = vocab['separator']
    scale_offset = vocab['scale_offset']
    scale_res = vocab['config']['scale_resolution']
    model = EncodecModel.encodec_model_48khz()
    with open(args.filename, 'r') as f:
        for i, line in enumerate(f):
            if i < args.index:
                continue

            if i == args.index+args.range:
                break

            tokens = [int(token) for token in line.split()]
            print(len(tokens), tokens[0:20])

            # seek for the first complete frame
            for seek, tok in enumerate(tokens):
                if scale_offset <= tok < scale_offset + scale_res:
                    break

            tokens = tokens[seek:]
            print('Seek index:', seek)
            print(len(tokens), tokens[0:20])

            # stop sonifying at EOS
            try:
                eos = tokens.index(separator)
            except ValueError:
                pass
            else:
                tokens = tokens[:eos]
                print('EOS truncation:', eos)

            frames, scales = audio.detokenize(tokens, vocab)
            print(len(frames), len(scales), scales)
            print(frames[0].shape, frames[-1].shape)
            print(frames[0].min(), frames[0].max(), frames[-1].min(), frames[-1].max())
            with torch.no_grad():
                wav = model.decode(zip(frames, [torch.tensor(s/100.).view(1) for s in scales]))[0]
            torchaudio.save(f'output/{Path(args.filename).stem}-{i}.wav', wav, model.sample_rate)
