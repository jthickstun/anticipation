import sys

import mido

from anticipation.mmvocab import vocab


if __name__ == '__main__':
    midifile = sys.argv[1]
    print('Compound file check for ', midifile)

    midi = mido.MidiFile(midifile)

    compound_file = midifile + '.compound.txt'
    with open(compound_file) as f:
        tokens = [int(token) for token in f.read().split()]

    print('Midi length: ', midi.length)
    print('Compound length: ', tokens[-5]/vocab['config']['time_resolution'])
