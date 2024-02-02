from argparse import ArgumentParser
from pathlib import Path

from anticipation.convert import events_to_midi
from anticipation.vocabs.tripletmidi import vocab

if __name__ == '__main__':
    parser = ArgumentParser(description='auditory check for a tokenized dataset')
    parser.add_argument('filename',
        help='file containing a tokenized MIDI dataset')
    parser.add_argument('index', type=int, default=0,
        help='the item to examine')
    parser.add_argument('range', type=int, default=1,
        help='range of items to examine')
    args = parser.parse_args()

    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    with open(args.filename, 'r') as f:
        for i, line in enumerate(f):
            if i < args.index:
                continue

            if i == args.index+args.range:
                break

            tokens = [int(token) for token in line.split()]
            tokens = [tok for tok in tokens if tok < vocab['special_offset']]
            assert(len(tokens) % 3 == 0)

            chord_instr_offset = vocab['control_offset'] + vocab['note_offset'] + 128 * (vocab['chord_instrument'] - vocab['instrument_offset'])

            for j in range(2, len(tokens), 3):
                if chord_instr_offset <= tokens[j] < chord_instr_offset + 128:
                    tokens[j] -= 97 * 128 # change to rhodes piano

            mid = events_to_midi(tokens, vocab, debug=True)
            mid.save(f'output/{Path(args.filename).stem}{i}.mid')
