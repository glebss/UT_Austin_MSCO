#!/usr/bin/env python3
import os
import subprocess
from subprocess import check_output, run
import re
import time
import argparse
from time import sleep

#
#  Feel free (a.k.a. you have to) to modify this to instrument your code
#

THREADS = list(range(33))
LOOPS = [10**i for i in range(6)]
# LOOPS = [10000]
INPUTS = ["10.txt", "1k.txt", "111.txt", "7777.txt", "8k.txt", "16k.txt"]
# INPUTS = ["32k.txt", "64k.txt", "128k.txt"]
# INPUTS = ["10.txt"]
# INPUTS = [f"{i}.txt" for i in range(1, 11)]
# INPUTS = os.listdir("tests")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spin', '-s', action="store_true")
    return parser.parse_args()

def run_process(cmd, start_time, timeout=10):
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while True:
        if process.poll() is not None:
            break
        if time.time() - start_time > timeout:
            process.terminate()
            process.wait()
            process.kill()
            raise subprocess.TimeoutExpired(cmd, timeout)
    out, _ = process.communicate()
    return out


def main():
    args = parse_args()
    if args.spin:
        print("Testing with spin barrier impl")
    else:
        print("Testing with pthread")
    csvs = []
    for inp in INPUTS:
        for loop in LOOPS:
            csv = ["{}/{}".format(inp, loop)]
            for thr in THREADS:
                print(f"inp {inp}, loop {loop}, thr {thr}")
                output_file = "output_seq.txt" if thr == 0 else "output.txt"
                # Don't know why but sometimes the process called from python hangs. 
                # Need to call it again if this is the case
                cmd = "./bin/prefix_scan -i tests/{} -o {} -n {} -l {}".format(
                    inp, output_file, thr, loop)
                if args.spin:
                    cmd += " -s"
                try:
                    start_time = time.time()
                    out = run_process(cmd, start_time)
                except subprocess.TimeoutExpired:
                    start_time = time.time()
                    out = run_process(cmd, start_time)

                # print(out)
                m = re.search("time: (.*)", out)
                if m is not None:
                    time_passed = m.group(1)
                    csv.append(time_passed)
                if thr != 0:
                    with open("output_seq.txt", 'r') as f:
                        out_seq = [int(l.strip()) for l in f.readlines()]
                    with open(output_file, 'r') as f:
                        out_par = [int(l.strip()) for l in f.readlines()]

                    assert out_seq == out_par, f"{inp}, {loop}, {thr}, {len(out_seq)}, {len(out_par)}"

                # sleep(1)
            csvs.append(csv)
            
    print("All tests passed!!!")

    header = ["microseconds"] + [str(x) for x in THREADS]

    print("\n")
    print(", ".join(header))
    for csv in csvs:
        print (", ".join(csv))


if __name__ == "__main__":
    main()