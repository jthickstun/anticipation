import traceback
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from glob import glob

from tqdm import tqdm

from anticipation.convert import midi_to_compound_new, midi_to_compound
from anticipation.config import PREPROC_WORKERS

from anticipation.vocabs.tripletmidi import vocab
import os


def convert_midi(filename, harmonize, output=None, debug=False):
    try:
        if debug:
            print('Processing file: ', filename)

        tokens, harmonized = midi_to_compound_new(filename, vocab, harmonize, debug=debug)
        # tokens = midi_to_compound(filename, vocab, debug=debug)

        if debug and harmonized == 0:
            print('Failed to harmonize: ', filename)

    except Exception:
        if debug:
            print('Failed to process: ', filename)
            print(traceback.format_exc())

        return 1

    if output:
        output_folder = os.path.join(output, os.path.basename(os.path.dirname(filename)))
        os.makedirs(output_folder, exist_ok=True)
        output_filename = os.path.join(output_folder, os.path.basename(filename) + ".compound.txt")
    else:
        output_filename = f"{filename}.compound.txt"

    with open(output_filename, 'w') as f:
        if debug:
            print(f'Writing {filename} to {output_filename}')

        f.write(' '.join(str(tok) for tok in tokens))

    return 0


def main(args):
    print(f'Midi time quantization is: {vocab["config"]["midi_quantization"]}')
    filenames = glob(args.dir + '/**/*.mid', recursive=True) \
            + glob(args.dir + '/**/*.midi', recursive=True)
    
    harmonize = args.harmonize
    debug = args.debug
    if args.output:
        output = args.output
    else:
        output = None
    print(f'Preprocessing {len(filenames)} files with {1} workers')
    with ProcessPoolExecutor(max_workers=1) as executor:
        partial_convert_midi = partial(convert_midi, harmonize=harmonize, output=output, debug=debug)
        results = list(tqdm(executor.map(partial_convert_midi, filenames), desc='Preprocess', total=len(filenames)))

    discards = round(100*sum(results)/float(len(filenames)),2)
    print(f'Successfully processed {len(filenames) - sum(results)} files (discarded {discards}%)')

if __name__ == '__main__':
    parser = ArgumentParser(description='prepares a MIDI dataset')
    parser.add_argument('dir', help='directory containing .mid files for training')
    parser.add_argument('--output', help='optional output directory, otherwise done in place')
    parser.add_argument('--harmonize', action='store_true', help="harmonize and store chords with program code specified by vocab")
    parser.add_argument('--debug', action='store_true', help="verbose debug messages")
    main(parser.parse_args())
