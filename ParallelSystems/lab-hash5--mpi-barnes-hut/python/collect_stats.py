import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from subprocess import check_output
from tqdm import tqdm

# n_steps = 500, t = 0.5

seq_10_times = [0.0128301, 0.0115226, 0.0122752, 0.0145769, 0.0122336]
seq_100_times = [0.278481, 0.276322, 0.277863, 0.27479, 0.278924]

seq_10_mean = np.mean(seq_10_times)
seq_10_std = np.std(seq_10_times)
seq_100_mean = np.mean(seq_100_times)
seq_100_std = np.std(seq_100_times)

n_procs = list(range(1, 9))
thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_RUNS = 5

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, default="", help="txt with initial particles coordinates")
    parser.add_argument("--output", type=str, default="output.txt")
    parser.add_argument("--n_steps", type=int, default=500)
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--np", type=int, default=3)
    args = parser.parse_args()

    num_objects = int(args.input.split('/')[1].split('.')[0].split('-')[-1])

    times = []
    # for n_proc in tqdm(n_procs):
    for thr in tqdm(thresholds):
        cmd = f"mpiexec -np {args.np} ./nbody -i {args.input} -o {args.output} -s {args.n_steps} -t {thr} -d {args.dt}"
        times_local = 0
        for _ in range(N_RUNS):
            out = check_output(cmd, shell=True).decode("ascii")
            times_local += float(out)
        times.append(times_local / N_RUNS)
    
    # times = [t / seq_100_mean for t in times]
    plt.scatter(thresholds, times, marker='o', color='red')

    for i in range(len(thresholds) - 1):
        plt.plot([thresholds[i], thresholds[i+1]], [times[i], times[i+1]], color='gray')
    plt.xlabel("Threshold")
    plt.ylabel("Processing time")
    plt.title(f"Processing time, input = {args.input.split('/')[-1]}, n_steps = {args.n_steps}, np = {args.np}")
    plt.grid(True)
    plt.savefig(f"graphs/threshold_n_obj_{num_objects}.png")
    print(times)

if __name__ == "__main__":
    main()
