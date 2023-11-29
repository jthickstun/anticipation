import os, math
from argparse import ArgumentParser
from multiprocessing import Pool, RLock
from glob import glob
from tqdm import tqdm
from pathlib import Path

import torch

from anticipation.mmvocab import vocab
from anticipation.audio import read_ecdc, skew
from anticipation.audio import tokenize as tokenize_audio
from anticipation.tokenize import maybe_tokenize


def compound_to_mm(tokens, vocab, stats=False):
    assert len(tokens) % 5 == 0

    # bail on bad midi files
    _, _, status = maybe_tokenize(tokens)
    if status > 0:
        raise RuntimeError

    tokens = tokens.copy()

    time_offset = vocab['time_offset']
    pitch_offset = vocab['pitch_offset']
    instr_offset = vocab['instrument_offset']
    dur_offset = vocab['duration_offset']

    time_res = vocab['config']['midi_quantization']
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

    audio_fps = vocab['config']['audio_fps']
    midi_quantization = vocab['config']['midi_quantization']
    time_offset = vocab['time_offset']
    blocks = audio.clone().T

    offset = 0
    time = delta*midi_quantization
    time_ratio = audio_fps / float(midi_quantization)
    for block in midi.T:
        time += block[0] - time_offset

        seqtime = math.floor(time*time_ratio) 
        seqpos = max(seqtime, 0) # events in first delta interval go at the start
        seqpos = min(seqpos, len(blocks)) # events after the sequence go at the end

        blocks = torch.cat((blocks[:seqpos+offset], block.unsqueeze(0), blocks[seqpos+offset:]), dim=0)
        offset += 1

    return blocks.T


def fast_anticipate(audio, midi, delta):
    if len(midi) == 0:
        return audio 

    audio_fps = vocab['config']['audio_fps']
    midi_quantization = vocab['config']['midi_quantization']
    time_offset = vocab['time_offset']
    audio = audio.clone().T

    audio_idx = offset = 0
    time = delta*midi_quantization
    time_ratio = audio_fps / float(midi_quantization)
    max_pos = len(audio)
    blocks = torch.empty(audio.shape[0]+midi.shape[1], audio.shape[1], dtype=audio.dtype)
    for midi_block in midi.T:
        time += midi_block[0] - time_offset

        seqtime = math.floor(time*time_ratio) 
        seqpos = max(seqtime, 0)      # events in first delta interval go at the start
        seqpos = min(seqpos, max_pos) # events after the sequence go at the end

        blocks[audio_idx+offset:seqpos+offset] = audio[audio_idx:seqpos]
        blocks[seqpos+offset] = midi_block
        audio_idx = seqpos
        offset += 1

    blocks[audio_idx+offset:] = audio[audio_idx:]

    return blocks.T


def prepare_mm(ecdc, vocab, anticipation):
    separator = vocab['separator']

    with open(f'{ecdc}.cache.txt', 'r') as f:
        cached_tokens = [int(token) for token in f.read().split()]

    midifile = ecdc.replace('.ecdc','.ismir2022_base.mid.compound.txt')
    with open(midifile, 'r') as f:
        compound_tokens = [int(token) for token in f.read().split()]

    audio_blocks = torch.tensor(cached_tokens).reshape(-1, 4).T
    midi_blocks = compound_to_mm(compound_tokens, vocab)

    blocks = fast_anticipate(audio_blocks, midi_blocks, anticipation)
    if vocab['config']['skew']:
        tokens = skew(blocks, 4, pad=vocab['residual_pad'])
    else:
        tokens = blocks.T.flatten().tolist()

    tokens[0:0] = 4*[separator]
    return tokens


def prepare_audio(ecdc, vocab):
    separator = vocab['separator']

    #audio_length, frames, scales = read_ecdc(Path(ecdc), model)
    #blocks = tokenize_audio(frames, scales, vocab)

    with open(f'{ecdc}.cache.txt', 'r') as f:
        tokens = [int(token) for token in f.read().split()]

    if vocab['config']['skew']:
        blocks = torch.tensor(tokens).reshape(-1, 4).T
        tokens = skew(blocks, 4, pad=vocab['residual_pad'])

    tokens[0:0] = 4*[separator]
    return tokens


def prepare_midi(midifile, vocab):
    separator = vocab['separator']

    with open(midifile, 'r') as f:
        compound_tokens = [int(token) for token in f.read().split()]

    blocks = compound_to_mm(compound_tokens, vocab)

    if vocab['config']['skew']:
        tokens = skew(blocks, 4, pad=vocab['residual_pad'])
    else:
        tokens = blocks.T.flatten().tolist()

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


def preprocess_transcribe(ecdcs, output, seqlen, idx):
    task = vocab['task']['transcribe']
    input_content = vocab['content_type']['clean_audio']
    output_content = vocab['content_type']['transcribed_midi']
    control_pad = vocab['control_pad']
    z = [task, output_content, input_content, control_pad]

    anticipation = vocab['config']['anticipation']
    prepare = lambda ecdc: prepare_mm(ecdc, vocab, anticipation)

    return pack_tokens(ecdcs, output, idx, z, prepare, seqlen=seqlen)


def preprocess_synthesize(ecdcs, output, seqlen, idx):
    task = vocab['task']['synthesize']
    input_content = vocab['content_type']['transcribed_midi']
    output_content = vocab['content_type']['clean_audio']
    control_pad = vocab['control_pad']
    z = [task, output_content, input_content, control_pad]

    anticipation = vocab['config']['anticipation']
    prepare = lambda ecdc: prepare_mm(ecdc, vocab, -anticipation)

    return pack_tokens(ecdcs, output, idx, z, prepare, seqlen=seqlen)


def preprocess_audio(ecdcs, output, seqlen, idx):
    task = vocab['task']['audiogen']
    output_content = vocab['content_type']['clean_audio']
    control_pad = vocab['control_pad']
    z = [task, output_content, control_pad, control_pad]

    prepare = lambda ecdc: prepare_audio(ecdc, vocab)

    return pack_tokens(ecdcs, output, idx, z, prepare, seqlen=seqlen)


def preprocess_cleanmidi(midifiles, output, seqlen, idx):
    task = vocab['task']['midigen']
    output_content = vocab['content_type']['clean_midi']
    control_pad = vocab['control_pad']
    z = [task, output_content, control_pad, control_pad]

    prepare = lambda mid: prepare_midi(mid, vocab)

    return pack_tokens(midifiles, output, idx, z, prepare, seqlen=seqlen)


def preprocess_transmidi(midifiles, output, seqlen, idx):
    task = vocab['task']['midigen']
    output_content = vocab['content_type']['transcribed_midi']
    control_pad = vocab['control_pad']
    z = [task, output_content, control_pad, control_pad]

    prepare = lambda mid: prepare_midi(mid, vocab)

    return pack_tokens(midifiles, output, idx, z, prepare, seqlen=seqlen)


preproc_func = {
    'audiogen' : preprocess_audio,
    'synthesize' : preprocess_synthesize,
    'transcribe' : preprocess_transcribe,
    'midigen' : preprocess_cleanmidi,
    'trans_midigen' : preprocess_transmidi,
}


def main(args):
    print('Tokenizing a multimodal dataset at:', args.datadir)
    print('Tokenization parameters:')
    print(f"  type = {args.type}")
    print(f"  context = {args.context}")
    print(f"  anticipation interval = {vocab['config']['anticipation']} seconds")
    print(f"  skew = {vocab['config']['skew']}")

    if 'midigen' in args.type:
        files = glob(os.path.join(args.datadir, '**/*.compound.txt'), recursive=True)
    else:
        files = glob(os.path.join(args.datadir, '**/*.ecdc'), recursive=True)

    n = len(files) // args.workers
    shards = [files[i:i+n] for i in range(args.workers)] # dropping a few tracks (< args.workers)
    outfiles = os.path.join(args.outdir, os.path.basename(args.datadir) + '.{t}.shard-{s:03}.txt')
    print('Outputs to:', outfiles)
    outputs = [outfiles.format(t=args.type, s=s) for s in range(len(shards))]
    context = args.workers*[args.context]

    print('Processing...')
    if args.debug:
        results = preproc_func[args.type](shards[0], outputs[0], args.context, 0)
        results = [results]
    else:
        with Pool(processes=args.workers, initargs=(RLock(),), initializer=tqdm.set_lock) as pool:
            results = pool.starmap(preproc_func[args.type], zip(shards, outputs, context, range(args.workers)))

    files, bad_files, seq_count = (sum(x) for x in zip(*results))

    print('Tokenization complete.')
    print(f'  => Processed {files} input files')
    print(f'  => Processed {seq_count} training sequences')
    print(f'  => Discarded {bad_files} input files (failed to read)')

if __name__ == '__main__':
    parser = ArgumentParser(description='tokenizes a multimodal dataset (ecdc audio paired with midi)')
    parser.add_argument('datadir', help='directory containing the dataset to tokenize')
    parser.add_argument('outdir', help='location to store the tokenized datafile')
    parser.add_argument('type', help='{audiogen|synthesize|transcribe|midigen}')
    parser.add_argument('context', type=int, help='context length for packing training sequences')
    parser.add_argument('--workers', type=int, default=16, help='number of workers/shards')
    parser.add_argument('--debug', action='store_true', help='debugging (single shard; non-parallel)')

    main(parser.parse_args())
