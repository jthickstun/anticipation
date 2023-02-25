import traceback
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob

from tqdm import tqdm

from anticipation.convert import midi_to_compound
from anticipation.config import PREPROC_WORKERS


def convert_midi(filename, debug=False):
    try:
        tokens = midi_to_compound(filename, debug=debug)
    except Exception:
        if debug:
            print('Failed to process: ', filename)
            print(traceback.format_exc())

        return

    with open(f"{filename}.compound.txt", 'w') as f:
        f.write(' '.join(str(tok) for tok in tokens))


def main(args):
    with ProcessPoolExecutor(max_workers=PREPROC_WORKERS) as executor:
        filenames = glob(args.dir + '/**/*.mid', recursive=True) \
                + glob(args.dir + '/**/*.midi', recursive=True)
        list(tqdm(executor.map(convert_midi, filenames), desc='Preprocess', total=len(filenames)))

if __name__ == '__main__':
    parser = ArgumentParser(description='prepares a MIDI dataset')
    parser.add_argument('dir', help='directory containing .mid files for training')
    main(parser.parse_args())
