from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from glob import glob

import torch, torchaudio

from anticipation import audio
from anticipation.audio import SEPARATOR, SCALE_OFFSET

from encodec.model import EncodecModel


if __name__ == '__main__':
    parser = ArgumentParser(description='auditory check for a tokenized audio dataset')
    parser.add_argument('filename',
        help='file containing a tokenized MIDI dataset')
    parser.add_argument('index', type=int, default=0,
        help='the item to examine')
    parser.add_argument('range', type=int, default=1,
        help='range of items to examine')
    args = parser.parse_args()

    model = EncodecModel.encodec_model_48khz()
    with open(args.filename, 'r') as f:
        for i, line in enumerate(f):
            if i < args.index:
                continue

            if i == args.index+args.range:
                break

            tokens = [int(token) for token in line.split()]

            # seek for the first complete frame
            for seek, tok in enumerate(tokens):
                if SCALE_OFFSET < tok < SCALE_OFFSET + 100:
                    break

            tokens = tokens[seek:]

            # stop sonifying at EOS
            try:
                eos = tokens.index(SEPARATOR)
            except ValueError:
                pass
            else:
                tokens = tokens[:eos]

            frames, scales = audio.detokenize(tokens)
            with torch.no_grad():
                wav = model.decode(zip(frames, [torch.tensor(s/100.).view(1) for s in scales]))[0]
            torchaudio.save(f'output/{Path(args.filename).stem}-{i}.wav', wav, model.sample_rate)
