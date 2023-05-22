from argparse import ArgumentParser
from tqdm import tqdm

from anticipation.vocab import *
from anticipation.ops import get_instruments

if __name__ == '__main__':
    parser = ArgumentParser(description='inspect a MIDI dataset')
    parser.add_argument('filename',
        help='file containing a tokenized MIDI dataset')
    args = parser.parse_args()

    with open(args.filename, 'r') as f:
        for line in tqdm(list(f)):
            tokens = [int(token) for token in line.split()]
            tokens = tokens[1:] # strip control codes
            assert(len([tok for tok in tokens if tok == SEPARATOR]) % 3 == 0)

            num_instruments = len(get_instruments(tokens))
            assert num_instruments <= MAX_TRACK_INSTR

            # check the ordering
            previous_time = TIME_OFFSET+0
            anticipation_time = ATIME_OFFSET+0
            check = False
            for time in tokens[0::3]:
                if time == SEPARATOR:
                    # reset the time counters for new sequence
                    previous_time = TIME_OFFSET+0
                    anticipation_time = ATIME_OFFSET+0
                    continue

                if time < CONTROL_OFFSET: # event token
                    assert(previous_time <= time) # events should come in order
                    previous_time = time
                    if check: # if the last token was anticipated
                        # check sequence ordering
                        assert(anticipation_time - CONTROL_OFFSET <= time + DELTA*TIME_RESOLUTION)
                        check = False
                else: # anticipated token
                    assert(anticipation_time <= time)
                    anticipation_time = time
                    check = True

    print('Integrity check passed for', args.filename)

