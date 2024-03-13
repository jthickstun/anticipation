import os
import datetime
import time
import pickle

from argparse import ArgumentParser
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM

from anticipation.sample import generate
from anticipation import ops

def main(args):
    # initialize the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model)

    # set the device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # set the seed for reproducibility
    torch.manual_seed(args.seed)

    interval = args.interval
    numIntervals = args.numIntervals
    numSequences = args.sequences

    stats = []

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    folder_path = os.path.join(args.output, timestamp)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    log_file_path = os.path.join(folder_path, "log.txt")

    with open(log_file_path, 'a') as f:
        f.write(f'Model: {args.model}\n')
        f.write(f'Timestamp: {timestamp}\n')
        f.write(f'Number of sequences to generate: {numSequences}\n')
        f.write(f'Generation interval length: {interval} sec.\n')
        f.write(f'Number of intervals to generate per sequence: {numIntervals}\n')
        f.write('\n')

    # generate sequences
    for s in range(numSequences):
        times = {}
        tokens = []
        start_time = 0
        end_time = interval
        for i in range(numIntervals):
            start_clock = time.time()
            tokens = generate(model, inputs=tokens, start_time=start_time, end_time=end_time, top_p=.98)
            end_clock = time.time()

            # compute number of instruments
            num_instr = len(list(ops.get_instruments(tokens).keys()))
            
            # track stats
            if (num_instr, interval) in times:
                times[(num_instr, interval)].append(end_clock - start_clock)
            else:
                times[(num_instr, interval)] = [end_clock - start_clock]

            start_time += interval
            end_time += interval
        
        stats.append(times)
        
        with open(log_file_path, 'a') as f:
            f.write(f'Sample {s+1} of {numSequences}. Generation interval length: {interval} sec.\n')

            f.write('\n')
            f.write(f'In this sample, intervals were generated with the following number of instruments: {len(times)}\n')
            f.write('\n')

            f.write('Summary by number of instruments generated:\n')
            for key, value in times.items():
                avg_time = sum(value) / len(value)
                num_instr, interval = key
                f.write(f"{len(value)} generation interval(s) contained {num_instr} instruments. Average Time: {avg_time}, Average Time/Interval: {avg_time/interval}\n")
            
            f.write('\n')

            overall_avg_time = sum([sum(value) / len(value) for value in times.values()]) / len(times)
            f.write(f"Average generation time for {interval} sec intervals across entire sequence: {overall_avg_time}\n")

            f.write('\n')
            f.write('Tokens:\n')
            f.write('\n')

            f.write(' '.join([str(tok) for tok in tokens]) + '\n')
            f.write('\n')
    
    stats_dump_path = os.path.join(folder_path, "stats.pickle")
    with open(stats_dump_path, 'wb') as f:
        pickle.dump(stats, f)

if __name__ == '__main__':
    parser = ArgumentParser(description='benchmark a tripletmidi anticipatory music transformer')
    parser.add_argument('-m', '--model', help='checkpoint for the model to evaluate')
    parser.add_argument('-o', '--output', help='output file for samples and logs')
    parser.add_argument('-N', '--sequences', type=int, default=100,
        help='number of sequences to generate')
    parser.add_argument('-s', '--seed', type=int, default=42,
        help='rng seed for sampling')
    parser.add_argument('-I', '--interval', type=int, default=1,
        help='generation interval in seconds')
    parser.add_argument('-n', '--numIntervals', type=int, default=25,
        help='number of intervals to generate per sequence')
    parser.add_argument('--debug', action='store_true', help='verbose debugging outputs')
    args = parser.parse_args()

    main(args)
