import os, traceback
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from tqdm import tqdm
from pathlib import Path

from anticipation.mmvocab import vocab
from anticipation.audio import read_ecdc
from anticipation.audio import tokenize as tokenize_audio

from encodec.model import EncodecModel


PREPROC_WORKERS = 1

model = EncodecModel.encodec_model_48khz()

def prepare_audio(ecdc, debug=True):
    separator = vocab['separator']

    try:
        audio_length, frames, scales = read_ecdc(Path(ecdc), model)
        blocks = tokenize_audio(frames, scales, vocab)
        tokens = blocks.T.flatten().tolist()
    except Exception:
        if debug:
            print('Failed to process: ', ecdc)
            print(traceback.format_exc())

        return 1

    with open(f"{ecdc}.cache.txt", 'w') as f:
        f.write(' '.join(str(tok) for tok in tokens))

    return 0


def main(args):
    filenames = glob(os.path.join(args.datadir, '**/*.ecdc'), recursive=True)

    print(f'Preprocessing {len(filenames)} files with {PREPROC_WORKERS} workers')
    with ProcessPoolExecutor(max_workers=PREPROC_WORKERS) as executor:
        results = list(tqdm(executor.map(prepare_audio, filenames), desc='Preprocess', total=len(filenames)))

    discards = round(100*sum(results)/float(len(filenames)),2)
    print(f'Successfully processed {len(filenames) - sum(results)} files (discarded {discards}%)')


if __name__ == '__main__':
    parser = ArgumentParser(description='tokenizes a multimodal dataset (ecdc audio paired with midi)')
    parser.add_argument('datadir', help='directory containing the dataset to tokenize')

    main(parser.parse_args())
