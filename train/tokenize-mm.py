import os, math
from argparse import ArgumentParser
from multiprocessing import Pool, RLock
from glob import glob
from tqdm import tqdm
from pathlib import Path

import torch

from encodec.model import EncodecModel

from anticipation.mmvocab import vocab
from anticipation.audio import read_ecdc, skew
from anticipation.audio import tokenize as tokenize_audio


PREPROC_WORKERS = 16
SEQ_LEN = 8192


def compound_to_mm(tokens, vocab, stats=False):
    assert len(tokens) % 5 == 0
    tokens = tokens.copy()

    time_offset = vocab['time_offset']
    pitch_offset = vocab['pitch_offset']
    instr_offset = vocab['instrument_offset']
    dur_offset = vocab['duration_offset']

    time_res = vocab['config']['time_resolution']
    max_duration = vocab['config']['max_duration']
    max_interarrival = vocab['config']['max_interarrival']

    rest = [time_offset+max_interarrival, vocab['rest'], vocab['rest'], dur_offset+max_interarrival]

    # remove velocities
    del tokens[4::5]

    mm_tokens = [None] * len(tokens)

    # sanity check and offset
    assert all(-1 <= tok < 2**7 for tok in tokens[2::4])
    assert all(-1 <= tok < 129 for tok in tokens[3::4])
    mm_tokens[1::4] = [pitch_offset + tok for tok in tokens[2::4]]
    mm_tokens[2::4] = [instr_offset + tok for tok in tokens[3::4]]

    # max duration cutoff and set unknown durations to 250ms
    truncations = sum([1 for tok in tokens[1::4] if tok >= max_duration])
    mm_tokens[3::4] = [dur_offset + time_res//4 if tok == -1 else dur_offset + min(tok, max_duration-1)
                       for tok in tokens[1::4]]

    # convert to interarrival times
    assert min(tokens[0::4]) >= 0
    offset = 0
    for idx in range(len(tokens) // 4):
        if idx == 0:
            previous_time = 0

        time = tokens[4*idx]
        ia = time - previous_time
        while ia > max_interarrival:
            # insert a rest
            mm_tokens[4*(idx+offset):4*(idx+offset)] = rest.copy()
            ia -= max_interarrival
            offset += 1

        mm_tokens[4*(idx+offset)] = time_offset + ia
        previous_time = time

    mm_tokens = torch.tensor(mm_tokens).reshape(-1, 4).T

    if stats:
        return mm_tokens, truncations

    return mm_tokens


def anticipate(audio, midi, delta):
    if len(midi) == 0:
        return audio 

    time_resolution = vocab['config']['time_resolution']
    time_offset = vocab['time_offset']
    blocks = audio.clone().T

    time = delta
    for block in midi.T:
        time += block[0] - time_offset

        # audio scale makes this off by one
        # (very annoying; but good news is this should generalize?)
        seqtime = time + math.floor(time/float(time_resolution)) 

        seqpos = max(seqtime, 0) # events in first delta interval go at the start
        seqpos = min(seqtime, len(blocks)) # events after the sequence go at the end
        blocks = torch.cat((blocks[:seqpos], block.unsqueeze(0), blocks[seqpos:]), dim=0)
        time += 1

    return blocks.T


def prepare_mm(ecdc, model, vocab, anticipation):
    separator = vocab['separator']

    audio_length, frames, scales = read_ecdc(Path(ecdc), model)
    midifile = ecdc.replace('.ecdc','.ismir2022_base.mid.compound.txt')
    with open(midifile, 'r') as f:
        compound_tokens = [int(token) for token in f.read().split()]

    midi_blocks = compound_to_mm(compound_tokens, vocab)
    audio_blocks = tokenize_audio(frames, scales, vocab)

    blocks = anticipate(audio_blocks, midi_blocks, anticipation)
    tokens = skew(blocks, 4, pad=vocab['residual_pad'])
    tokens[0:0] = 4*[separator]
    return tokens


def prepare_audio(ecdc, model, vocab):
    separator = vocab['separator']

    audio_length, frames, scales = read_ecdc(Path(ecdc), model)

    blocks = tokenize_audio(frames, scales, vocab)
    tokens = skew(blocks, 4, pad=vocab['residual_pad'])
    tokens[0:0] = 4*[separator]
    return tokens


def prepare_midi(midifile, vocab):
    separator = vocab['separator']

    with open(midifile, 'r') as f:
        compound_tokens = [int(token) for token in f.read().split()]

    blocks = compound_to_mm(compound_tokens, vocab)
    tokens = skew(blocks, 4, pad=vocab['residual_pad'])
    tokens[0:0] = 4*[separator]
    return tokens


def pack_tokens(ecdcs, output, idx, z, prepare, seqlen):
    files = bad_files = seqcount = 0
    with open(output, 'w') as outfile:
        concatenated_tokens = []
        for ecdc in tqdm(ecdcs, desc=f'#{idx}', position=idx+1, leave=True):
            try:
                tokens = prepare(ecdc)
                files += 1
            except Exception as e:
                #print(e)
                bad_files += 1
                continue

            # write out full sequences to file
            concatenated_tokens.extend(tokens)
            while len(concatenated_tokens) >= seqlen-len(z):
                seq = concatenated_tokens[0:seqlen-len(z)]
                seq = z + seq
                concatenated_tokens = concatenated_tokens[seqlen:]

                outfile.write(' '.join([str(tok) for tok in seq]) + '\n')
                seqcount += 1

    return (files, bad_files, seqcount)


def preprocess_transcribe(ecdcs, output, idx):
    task = vocab['task']['transcribe']
    input_content = vocab['content_type']['clean_audio']
    output_content = vocab['content_type']['transcribed_midi']
    control_pad = vocab['control_pad']
    z = [task, input_content, output_content, control_pad]

    anticipation = vocab['config']['anticipation']
    model = EncodecModel.encodec_model_48khz()
    prepare = lambda ecdc: prepare_mm(ecdc, model, vocab, -anticipation)

    return pack_tokens(ecdcs, output, idx, z, prepare, seqlen=SEQ_LEN)


def preprocess_synthesize(ecdcs, output, idx):
    task = vocab['task']['synthesize']
    input_content = vocab['content_type']['transcribed_midi']
    output_content = vocab['content_type']['clean_audio']
    control_pad = vocab['control_pad']
    z = [task, input_content, output_content, control_pad]

    anticipation = vocab['config']['anticipation']
    model = EncodecModel.encodec_model_48khz()
    prepare = lambda ecdc: prepare_mm(ecdc, model, vocab, anticipation)

    return pack_tokens(ecdcs, output, idx, z, prepare, seqlen=SEQ_LEN)


def preprocess_audio(ecdcs, output, idx):
    task = vocab['task']['audiogen']
    input_content = vocab['content_type']['clean_audio']
    control_pad = vocab['control_pad']
    z = [task, input_content, control_pad, control_pad]

    model = EncodecModel.encodec_model_48khz()
    prepare = lambda ecdc: prepare_audio(ecdc, model, vocab)

    return pack_tokens(ecdcs, output, idx, z, prepare, seqlen=SEQ_LEN)


def preprocess_cleanmidi(midifiles, output, idx):
    task = vocab['task']['audiogen']
    input_content = vocab['content_type']['clean_midi']
    control_pad = vocab['control_pad']
    z = [task, input_content, control_pad, control_pad]

    prepare = lambda mid: prepare_midi(mid, vocab)

    return pack_tokens(midifiles, output, idx, z, prepare, seqlen=SEQ_LEN)


preproc_func = {
    'audiogen' : preprocess_audio,
    'synthesize' : preprocess_synthesize,
    'transcribe' : preprocess_transcribe,
    'midigen' : preprocess_cleanmidi
}


def main(args):
    print('Tokenizing a multimodal dataset at:', args.datadir)
    print('Tokenization parameters:')
    print(f"  anticipation interval = {vocab['config']['anticipation']} frames")
    print('Processing...')

    if args.type == 'midigen':
        files = glob(os.path.join(args.datadir, '**/*.compound.txt'), recursive=True)
    else:
        files = glob(os.path.join(args.datadir, '**/*.ecdc'), recursive=True)

    n = len(files) // PREPROC_WORKERS
    shards = [files[i:i+n] for i in range(PREPROC_WORKERS)] # dropping a few tracks (< PREPROC_WORKERS)
    outfiles = args.datadir + '.{t}.shard-{s}.txt'
    print('Outputs to:', outfiles)
    outputs = [outfiles.format(t=args.type, s=s) for s in range(len(shards))]

    if args.debug:
        results = preproc_func[args.type](shards[0], outputs[0], 0)
        results = [results]
    else:
        with Pool(processes=PREPROC_WORKERS, initargs=(RLock(),), initializer=tqdm.set_lock) as pool:
            results = pool.starmap(preproc_func[args.type], zip(shards, outputs, range(PREPROC_WORKERS)))

    files, bad_files, seq_count = (sum(x) for x in zip(*results))

    print('Tokenization complete.')
    print(f'  => Processed {files} input files')
    print(f'  => Processed {seq_count} training sequences')
    print(f'  => Discarded {bad_files} input files (failed to read)')

if __name__ == '__main__':
    parser = ArgumentParser(description='tokenizes a multimodal dataset (ecdc audio paired with midi)')
    parser.add_argument('datadir', help='directory containing the dataset to tokenize')
    parser.add_argument('type', help='{audiogen|synthesize|transcribe|midigen}')
    parser.add_argument('--debug', action='store_true', help='debugging (single shard; non-parallel)')

    main(parser.parse_args())
