import os, math, traceback
from argparse import ArgumentParser
from multiprocessing import Pool, RLock
from glob import glob
from tqdm import tqdm
from pathlib import Path

import numpy as np

from anticipation import ops
from anticipation.tokenize import maybe_tokenize
from anticipation.config import HUMAN_DELTA, TIME_RESOLUTION


def extract_instruments(all_events, instruments, vocab):
    events = []
    controls = []

    control_offset = vocab['control_offset']
    note_offset = vocab['note_offset']
    separator = vocab['separator']
    rest = vocab['rest']

    for time, dur, note in zip(all_events[0::3],all_events[1::3],all_events[2::3]):
        assert note < control_offset         # shouldn't be in the sequence yet
        assert note not in [separator, rest] # these shouldn't either

        instr = (note-note_offset)//2**7
        if instr in instruments:
            # mark this event as a control
            controls.extend([control_offset+time, control_offset+dur, control_offset+note])
        else:
            events.extend([time, dur, note])

    return events, controls

def prepare_triplet_midi(midifile, vocab):
    with open(midifile, 'r') as f:
        cmp = [int(token) for token in f.read().split()]
        events, truncations, status = maybe_tokenize(cmp, vocab)

    if status > 0:
        raise ValueError(f'Bad midi sequence (status {status})')

    return events


def control_prefix(instruments, task, vocab):
    task = vocab['task'][task]
    instr_offset = vocab['instrument_offset']
    separator = vocab['separator']
    pad = vocab['pad']

    # get the list of instruments to condition on
    # by convention, let's provide the list sorted by instrument code
    instr_controls = sorted(instruments)
    instr_controls = [instr_offset + instr for instr in instruments]

    vocab_size = vocab['config']['size']
    assert max(instr_controls) < vocab_size

    # put task last, so the model knows it's time to generate events once it's seen the task token
    z_start = [separator] + instr_controls + [task]
    z_cont = instr_controls + [task]

    # pad the start controls out to an offset of 0 (mod 3)
    if len(z_start) % 3 > 0:
        z_start[1:1] = (3-len(z_start)%3)*[pad]

    # pad the continuation controls out to an offset of 1 (mod 3)
    if len(z_cont) % 3 > 0:
        z_cont[0:0] = (3-len(z_cont)%3)*[pad]
    z_cont = [pad] + z_cont

    return z_start, z_cont


def pack_tokens(sequences, output, idx, vocab, prepare, prefix, seqlen):
    vocab_size = vocab['config']['size']
    pad = vocab['pad']

    files = bad_files = seqcount = 0
    with open(output, 'w') as outfile:
        concatenated_tokens = []
        for sequence in tqdm(sequences, desc=f'#{idx}', position=idx+1, leave=True):
            if len(concatenated_tokens) == 0:
                z = [pad]

            try:
                events = prepare(sequence)
                files += 1
            except Exception as e:
                #print(e)
                #print(traceback.format_exc())
                bad_files += 1
                continue

            # record the original end time before extracting control tokens
            end_time = ops.max_time(events, seconds=False)

            # anticipation happens here (extract control tokens)
            #   * extract the chord sequence to anticipate
            #   * extract the (randomly selected) "human" sequence to anti-anticipate
            
            instruments = list(ops.get_instruments(events).keys())

            if len(instruments) < 2:
                continue

            # extract the chord sequence to anticipate
            # check if vocab['chord_instrument'] is in instruments
            if vocab['chord_instrument'] in instruments:
                # extract the chord sequence to anticipate
                events, chord_controls = extract_instruments(events, [vocab['chord_instrument']])
                instruments.remove(vocab['chord_instrument'])
            else:
                continue

            # extract the randomly selected "human" sequence to anti-anticipate
            human = np.random.choice(instruments, 1, replace=False)
            events, human_controls = extract_instruments(events, human)
            
            #
            #
            #
            #

            # get the global control tokens for this sequence
            # do this before padding because some ops don't handle REST properly
            instruments = sorted(ops.get_instruments(events).keys())
            z_start, z_cont = prefix(instruments)

            # add rest tokens to events after extracting control tokens
            # (see Section 3.2 of the paper for why we do this)
            events = ops.pad(events, end_time)

            #
            # interleave the events and anticipated controls
            #    * something like this: tokens, controls = ops.anticipate(events, controls)
            #    * might need some care about how to anti-anticipate
            #

            tokens, chord_controls = ops.anticipate(events, chord_controls)
            tokens, human_controls = ops.anticipate(events, human_controls, delta=(HUMAN_DELTA*TIME_RESOLUTION))

            #
            #
            #
            #

            # write out full contexts to file
            concatenated_tokens.extend(z_start + tokens)
            while len(concatenated_tokens) >= seqlen-len(z):
                seq = concatenated_tokens[0:seqlen-len(z)]
                concatenated_tokens = concatenated_tokens[len(seq):]

                # relativize time to the context 
                try: 
                    seq = ops.translate(seq, -ops.min_time(seq, seconds=False), seconds=False)
                    assert ops.min_time(seq, seconds=False) == 0
                except OverflowError:
                    # TODO: I'm not sure I ever actually check for overflow (max_time(seq) > 10000)
                    #   * did I ever correctly check for this?
                    #   * is this causing vocab overflows?
                    #      - could be that there are overflows, but not bad enough to overflow
                    #      the top of the vocabulary and hit the assertion max(seq) < vocab_size
                    #      because time events are at the bottom of the vocabulary
                    #   * follow up on this: at least check whether this codepath is ever reached
                    continue

                seq = z + seq

                assert max(seq) < vocab_size
                outfile.write(' '.join([str(tok) for tok in seq]) + '\n')
                z = z_cont # update the global control prompt (if it changed)
                seqcount += 1

    return (files, bad_files, seqcount)


def preprocess_midi(midifiles, output, seqlen, task, vocab, idx):
    prefix = lambda instruments: control_prefix(instruments, task, vocab)
    prepare = lambda mid: prepare_triplet_midi(mid, vocab)

    return pack_tokens(midifiles, output, idx, vocab, prepare, prefix, seqlen=seqlen)


preproc_func = {
    'autoregress' : preprocess_midi,
}

def main(args):
    print('Tokenizing a dataset at:', args.datadir)

    if args.vocab == 'triplet-midi':
        from anticipation.vocabs.tripletmidi import vocab
    else:
        raise ValueError(f'Invalid vocabulary type "{args.vocab}"')

    print('Tokenization parameters:')
    print(f"  vocab = {args.vocab}")
    print(f"  task = {args.task}")
    print(f"  context = {args.context}")
    print(f"  anticipation interval = {vocab['config']['anticipation']} seconds")
    print(f"  skew = {vocab['config']['skew']}")

    files = glob(os.path.join(args.datadir, '**/*.compound.txt'), recursive=True)

    n = len(files) // args.workers
    shards = [files[i*n:(i+1)*n] for i in range(args.workers)] # dropping a few tracks (< args.workers)
    outfiles = os.path.join(args.outdir, os.path.basename(args.datadir) + '.{t}.shard-{s:03}.txt')
    print('Outputs to:', outfiles)
    outputs = [outfiles.format(t=args.task, s=s) for s in range(len(shards))]
    context = args.workers*[args.context]
    task = args.workers*[args.task]
    vocab = args.workers*[vocab]

    print('Processing...')
    if args.debug:
        results = preproc_func[args.task](shards[0], outputs[0], args.context, args.task, vocab[0], 0)
        results = [results]
    else:
        with Pool(processes=args.workers, initargs=(RLock(),), initializer=tqdm.set_lock) as pool:
            results = pool.starmap(preproc_func[args.task], zip(shards, outputs, context, task, vocab, range(args.workers)))

    files, bad_files, seq_count = (sum(x) for x in zip(*results))

    print('Tokenization complete.')
    print(f'  => Processed {files} input files')
    print(f'  => Processed {seq_count} training sequences')
    print(f'  => Discarded {bad_files} input files (failed to read)')

if __name__ == '__main__':
    parser = ArgumentParser(description='tokenizes a dataset')
    parser.add_argument('datadir', help='directory containing the dataset to tokenize')
    parser.add_argument('outdir', help='location to store the tokenized datafile')
    parser.add_argument('task', help='task for which we are preparing sequences')
    parser.add_argument('context', type=int, default=1024, help='context length for packing training sequences')
    parser.add_argument('-v', '--vocab', default='triplet-midi', help='name of vocabulary to use for tokenization')
    parser.add_argument('--workers', type=int, default=16, help='number of workers/shards')
    parser.add_argument('--debug', action='store_true', help='debugging (single shard; non-parallel)')

    main(parser.parse_args())
