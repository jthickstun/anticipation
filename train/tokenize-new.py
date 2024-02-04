import os, math, traceback
from argparse import ArgumentParser
from multiprocessing import Pool, RLock
from glob import glob
from tqdm import tqdm
from pathlib import Path

import numpy as np

from anticipation import ops
from anticipation import tokenize


def prepare_triplet_midi(midifile, task):
    with open(midifile, 'r') as f:
        events, truncations, status = tokenize.maybe_tokenize([int(token) for token in f.read().split()])

    if status > 0:
        return events, status

    # record the original end time before extracting control tokens
    end_time = ops.max_time(events, seconds=False)

    if  task == 'autoregress':
        controls = []
    elif task == 'instrument':
        instruments = list(ops.get_instruments(events).keys())
        if len(instruments) < 2:
            # need at least two instruments for instrument anticipation
            return events, 4 # status 4 == too few instruments

        u = 1+np.random.randint(len(instruments)-1)
        subset = np.random.choice(instruments, u, replace=False)
        events, controls = tokenize.extract_instruments(events, subset)
    elif task == 'span':
        events, controls = tokenize.extract_spans(all_events, .05)
    elif task == 'random':
        events, controls = tokenize.extract_random(all_events, 10)

    # add rest tokens to events after extracting control tokens
    # (see Section 3.2 of the paper for why we do this)
    events = ops.pad(events, end_time)

    # interleave the events and anticipated controls
    tokens, controls = ops.anticipate(events, controls)
    assert len(controls) == 0 # should have consumed all controls (because of padding)

    return tokens, status


def pack_tokens(sequences, output, idx, prepare, z, separator, config, seqlen):
    vocab_size = config['size']
    max_arrival = config['max_time']

    seqcount = 0
    stats = 5*[0] # (short, long, too many instruments, too few instruments, inexpressible)
    with open(output, 'w') as outfile:
        concatenated_tokens = []
        for sequence in tqdm(sequences, desc=f'#{idx}', position=idx+1, leave=True):
            tokens, status = prepare(sequence)
            if status > 0:
                stats[status-1] += 1
                continue

            # write out full contexts to file
            concatenated_tokens.extend(separator + tokens)
            while len(concatenated_tokens) >= seqlen-len(z):
                seq = concatenated_tokens[0:seqlen-len(z)]
                concatenated_tokens = concatenated_tokens[len(seq):]

                # relativize time to the context 
                seq = ops.translate(seq, -ops.min_time(seq, seconds=False), seconds=False)
                assert ops.min_time(seq, seconds=False) == 0
                if ops.max_time(seq, seconds=False) >= max_arrival:
                    stats[4] += 1
                    continue

                seq = z + seq

                assert max(seq) < vocab_size
                #if max(seq) >= vocab_size:
                #    print('OUCH\n', max(seq), seq[0], max(seq[1::3]), max(seq[2::3]), max(seq[3::3]), '^^^\n', sequence, '\n')

                outfile.write(' '.join([str(tok) for tok in seq]) + '\n')
                seqcount += 1

    return (seqcount, *stats)


def preprocess_ar(midifiles, output, seqlen, task, factor, vocab, idx):
    assert factor == 1, f'Autoregressive preprocessing has no randomness; cannot apply augmentation factor {factor}'

    z = [vocab['task']['autoregress']]
    separator = [vocab['separator'] for _ in range(3)]

    prepare = lambda midifile: prepare_triplet_midi(midifile, task)

    return pack_tokens(midifiles, output, idx, prepare, z, separator, vocab['config'], seqlen=seqlen)


def preprocess_aar(midifiles, output, seqlen, task, factor, vocab, idx):
    z = [vocab['task']['anticipate']]
    separator = [vocab['separator'] for _ in range(3)]

    prepare = lambda midifile: prepare_triplet_midi(midifile, task)

    results = []
    for i in range(factor):
        results.append(pack_tokens(midifiles, output, idx, prepare, z, separator, vocab['config'], seqlen=seqlen))

    return tuple(map(sum, zip(*results)))


preproc_func = {
    'autoregress' : preprocess_ar,
    'random' : preprocess_aar,
    'span' : preprocess_aar,
    'instrument' : preprocess_aar,
}

def init_worker(lock):
    tqdm.set_lock(lock)
    np.random.seed(os.getpid())

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
    print(f"  augmentation factor = {args.factor}")
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
    factor = args.workers*[args.factor]
    vocab = args.workers*[vocab]

    print('Processing...')
    if args.debug:
        results = preproc_func[args.task](shards[0], outputs[0], args.context, args.task, args.factor, vocab[0], 0)
        results = [results]
    else:
        with Pool(processes=args.workers, initargs=(RLock(),), initializer=init_worker) as pool:
            results = pool.starmap(preproc_func[args.task], zip(shards, outputs, context, task, factor, vocab, range(args.workers)))

    seqcount, too_short, too_long, many_instr, few_instr, discarded_seqs = (sum(x) for x in zip(*results))

    print('Tokenization complete.')
    print(f'  => Processed {seqcount} training sequences')
    print(f'  => Discarded {too_short+too_long+many_instr+few_instr} event sequences')
    print(f'      - {too_short} too short')
    print(f'      - {too_long} too long')
    print(f'      - {many_instr} too many instruments')
    print(f'      - {few_instr} too few instruments')
    print(f'  => Discarded {discarded_seqs} training sequences')


if __name__ == '__main__':
    parser = ArgumentParser(description='tokenizes a dataset')
    parser.add_argument('datadir', help='directory containing the dataset to tokenize')
    parser.add_argument('outdir', help='location to store the tokenized datafile')
    parser.add_argument('task', help='task for which we are preparing sequences')
    parser.add_argument('context', type=int, help='context length for packing training sequences')
    parser.add_argument('-v', '--vocab', default='triplet-midi', help='name of vocabulary to use for tokenization')
    parser.add_argument('-f', '--factor', type=int, default=1, help='augmentation factor')
    parser.add_argument('--workers', type=int, default=16, help='number of workers/shards')
    parser.add_argument('--debug', action='store_true', help='debugging (single shard; non-parallel)')

    main(parser.parse_args())
