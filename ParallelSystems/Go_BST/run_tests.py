import argparse
from subprocess import check_output
import re
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input file")
    args = parser.parse_args()
    return args

NUM_HASH_WORKERS = [1, 2, 4, 8, 16]
NUM_GROUP_WORKERS = [1, 2, 4, 8, 16]
NUM_COMP_WORKERS = [1, 2, 4, 8, 16]
NUM_RUNS = 100

def main():
    args = parse_args()
    inp = args.input
    with open(inp, 'r') as f:
        lines = f.readlines()
        num_workers_n = len(lines)
    
    times = {"hash_workers": {}}
    # NUM_HASH_WORKERS = NUM_HASH_WORKERS + [num_workers_n]
    # NUM_GROUP_WORKERS = NUM_GROUP_WORKERS + [num_workers_n]

    # for num_workers in NUM_GROUP_WORKERS + [num_workers_n]:
    for num_workers in NUM_COMP_WORKERS + [num_workers_n]:
        print(num_workers)
        for _ in tqdm(range(NUM_RUNS)):
            cmd = "./bst_comp -input={} -data-workers=16 -comp-workers={} -hash-workers=16".format(
                        inp, num_workers)
            out = check_output(cmd, shell=True).decode("ascii")
            m = re.search("compareTreeTime: (.*)", out)
            if m is not None:
                time = m.group(1)
            times["hash_workers"].setdefault(num_workers, [])
            times["hash_workers"][num_workers].append(float(time))
            # print(f"{num_workers} : {time}")
    
    # print(times)
    for num_workers, times_worker in times["hash_workers"].items():
        print(f"{num_workers} : {np.mean(times_worker) * 1000} +- {np.std(times_worker) * 1000}")
    
if __name__ == "__main__":
    main()
