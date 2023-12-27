from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from glob import glob

import torch, torchaudio

from anticipation import audio
from anticipation.convert import compound_to_midi

from transformers import EncodecModel as EncodecModel32k


SAMPLE_RATE = 32000


def detokenize(blocks, vocab):
    residuals = vocab['config']['residuals']
    offsets = torch.tensor(vocab['residual_offset'])[:,None]
    assert residuals > 0

    # remove offsets for the residual vocabularies
    for i in range(residuals):
        blocks[i] = blocks[i] - offsets[i]

    return blocks.view(1,1,4,-1)


def mm_to_compound(blocks, vocab, debug=False):
    time_offset = vocab['time_offset']
    pitch_offset = vocab['pitch_offset']
    instr_offset = vocab['instrument_offset']
    dur_offset = vocab['duration_offset']

    rest = vocab['rest'] - pitch_offset

    tokens = blocks.T.flatten().tolist()

    out = 5*(len(tokens)//4)*[None]
    out[0::5] = [tok - time_offset for tok in tokens[0::4]]
    out[1::5] = [tok - dur_offset for tok in tokens[3::4]]
    out[2::5] = [tok - pitch_offset for tok in tokens[2::4]]
    out[3::5] = [tok - instr_offset for tok in tokens[1::4]]
    out[4::5] = (len(tokens)//4)*[72] # default velocity

    # convert interarrival times to arrival times
    time = 0
    out_norest = []
    for idx in range(len(out)//5):
        time += out[5*idx]
        if out[5*idx+2] == rest:
            continue

        out_norest.extend(out[5*idx:5*(idx+1)])
        out_norest[-5] = time

    return out_norest


def split(blocks, vocab, debug=False):
    """ split token blocks into midi and audio"""

    midi_offset = vocab['midi_offset']

    audio = torch.zeros([4,0], dtype=blocks.dtype)
    midi = torch.zeros([4,0], dtype=blocks.dtype)
    time = 0
    for i, block in enumerate(blocks.T):
        if block[0] < midi_offset:
            audio = torch.cat((audio, block.unsqueeze(1)), dim=1)
        else:
            if debug:
                print('MIDI event at sequence position', i)
                print('  MIDI sequence interrarival time is', )


            midi = torch.cat((midi, block.unsqueeze(1)), dim=1)

    return audio, midi 


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
    parser.add_argument('--debug', action='store_true', help='verbose debugging outputs')
    args = parser.parse_args()

    if args.vocab == 'audio':
        from anticipation.audiovocab import vocab
    elif args.vocab == 'mm':
        from anticipation.mmvocab import vocab
    else:
        raise ValueError(f'Invalid vocabulary type "{args.vocab}"')

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    separator = vocab['separator']
    pad = vocab['residual_pad']
    skew = vocab['config']['skew']
    print(device, separator, skew)
    model = EncodecModel32k.from_pretrained("facebook/encodec_32khz").to(device)
    with open(args.filename, 'r') as f:
        for i, line in enumerate(f):
            if i < args.index:
                continue

            if i == args.index+args.range:
                break

            tokens = [int(token) for token in line.split()]

            # strip the control block
            tokens = tokens[4:]

            # strip sequence separators
            tokens = [token for token in tokens if token != separator]

            if skew:
                blocks = audio.deskew(tokens, 4)
            else:
                blocks = torch.tensor(tokens).reshape(-1, 4).T

            # strip residual pads?
            #tokens = [token for token in tokens if token != pad]

            if args.vocab == 'mm':
                blocks, midi_blocks = split(blocks, vocab, args.debug)
                if midi_blocks.shape[1] > 0:
                    mid = compound_to_midi(mm_to_compound(midi_blocks, vocab), vocab)
                    mid.save(f'output/{Path(args.filename).stem}-{i}.mid')

            if blocks.shape[1] == 0:
                continue

            audio_codes = detokenize(blocks, vocab).to(device)
            print(audio_codes.shape, audio_codes.min(), audio_codes.max())
            with torch.no_grad():
                wav = model.decode(audio_codes, [None]).audio_values.cpu()[0]
                print(wav.shape)
            torchaudio.save(f'output/{Path(args.filename).stem}-{i}.wav', wav, SAMPLE_RATE)

