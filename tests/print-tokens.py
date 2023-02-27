from argparse import ArgumentParser

from anticipation.ops import print_tokens

if __name__ == '__main__':
    parser = ArgumentParser(description='inspect a MIDI dataset')
    parser.add_argument('filename',
        help='file containing a tokenized MIDI dataset')
    parser.add_argument('index', type=int, default=0,
        help='the item to examine')
    args = parser.parse_args()

    with open(args.filename, 'r') as f:
        for i, line in enumerate(f):
            if i < args.index:
                continue 

            tokens = [int(token) for token in line.split()]
            control = tokens[:1]
            tokens = tokens[1:]
            print('Control tokens:', control)
            print_tokens(tokens)
            break
