import argparse
import pickle
import matplotlib.pyplot as plt

def main(stats_file):
    with open(stats_file, 'rb') as file:
        stats = pickle.load(file)

    # The imported stats is a list of dicts. Each dict corresponds to a generated sequence.
    # The keys of a dict are (num_instr, interval) tuples. The values of the dict
    # are lists of times taken to generate each interval. 

    times_by_num_instr = {}
    interval = None
    for sample in stats:
        for key, value in sample.items():
            num_instr, interval = key
            if not interval:
                interval = interval
            else:
                assert interval == interval

            if num_instr in times_by_num_instr:
                times_by_num_instr[num_instr].extend(value)
            else:
                times_by_num_instr[num_instr] = value 

    means = []
    stds = []
    for key, value in times_by_num_instr.items():
        mean = sum(value) / len(value)
        std = (sum([(x - mean)**2 for x in value]) / len(value))**.5
        means.append(mean)
        stds.append(std)

    plt.axhline(y=interval, linestyle='--', color='red', label='Real-time')
    plt.errorbar(times_by_num_instr.keys(), means, yerr=stds, fmt='o')
    plt.xlabel('Number of Instruments')
    plt.ylabel('Time (s)')
    plt.title(f'Generation Time of {interval}s Intervals by Number of Instruments')
    plt.legend() 
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot benchmark data')
    parser.add_argument('stats_file', type=str, help='Path to the stats file')
    args = parser.parse_args()
    main(args.stats_file)