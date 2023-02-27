from argparse import ArgumentParser

from anticipation.vocab import SEPARATOR

if __name__ == '__main__':
    parser = ArgumentParser(description='inspect a MIDI dataset')
    parser.add_argument('filename',
        help='file containing a tokenized MIDI dataset')
    parser.add_argument('index', type=int, default=0,
        help='start index of items to examine')
    parser.add_argument('range', type=int, default=1,
        help='number of items to examine')
    args = parser.parse_args()

    with open(args.filename, 'r') as f:
        for i, line in enumerate(f):
            if i == args.index+args.range:
                break

            if i >= args.index:
                tokens = [int(token) for token in line.split()]

                if SEPARATOR in tokens[1:]:
                    print(f'Sequence boundary in line {i}. Control codes {tokens[:1]}')
