from subprocess import check_output
import re
import argparse
from time import sleep
import matplotlib.pyplot as plt

THREADS = list(range(2, 33, 2))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str)
    parser.add_argument('--loops', '-l', type=int, default=100000)
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--spin', '-s', action="store_true")
    
    return parser.parse_args()

def main():
    args = parse_args()
    INPUT = args.input
    LOOPS = args.loops
    NUM_RUNS = args.n_runs
    spin = args.spin
    # sequential impl
    seq_time = 0

    cmd = "./bin/prefix_scan -o tmp_out.txt -n 0 -i tests/{} -l {}".format(
             INPUT, LOOPS)
    for _ in range(NUM_RUNS):
        out = check_output(cmd, shell=True).decode("ascii")
        m = re.search("time: (.*)", out)
        if m is not None:
            seq_time += int(m.group(1))
    seq_time /= NUM_RUNS
    
    # parallel
    times = []
    for thr in THREADS:
        print(thr)
        if spin:
            cmd = "./bin/prefix_scan -o tmp_out.txt -n {} -i tests/{} -l {} --spin".format(
                thr, INPUT, LOOPS)
        else:
            cmd = "./bin/prefix_scan -o tmp_out.txt -n {} -i tests/{} -l {}".format(
                thr, INPUT, LOOPS)
        time = 0
        for _ in range(NUM_RUNS):
            out = check_output(cmd, shell=True).decode("ascii")
            m = re.search("time: (.*)", out)
            if m is not None:
                time += int(m.group(1))
        times.append(time / NUM_RUNS)
    
    times = [ t / seq_time for t in times]

    _, ax = plt.subplots()

    # Plot
    ax.scatter(THREADS, times, marker='o', color='red')

    for i in range(len(THREADS) - 1):
        ax.plot([THREADS[i], THREADS[i+1]], [times[i], times[i+1]], color='gray')

    ax.set_xlabel('n_threads')
    ax.set_ylabel('time relative')
    title = f'Parallel prefix sum over sequential, inp={INPUT}, loops={LOOPS}'
    if spin:
        title += f', spin={spin}'
    ax.set_title(title)
    ax.set_xticks(THREADS)
    ax.grid(True)
    out_img_path = f"{INPUT.split('.')[0]}_loops_{LOOPS}_spin.jpg" if spin \
                    else f"{INPUT.split('.')[0]}_loops_{LOOPS}.jpg"
    plt.savefig(out_img_path)

if __name__ == "__main__":
    main()



