import argparse
import numpy as np
import matplotlib.pyplot as plt

def read_file(file_path):
    with open(file_path, 'r') as f:
        in_lines = f.readlines()[1:]
        in_lines = [l.strip() for l in in_lines]
    
    res = []
    for l in in_lines:
        idx, x, y, mass, x_vel, y_vel = l.split()
        res.append((idx, x, y, mass, x_vel, y_vel))
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, default="", help="txt with initial particles coordinates")
    parser.add_argument("output", type=str, default="", help="txt with final particles coordinates")
    parser.add_argument("out_img", type=str, default="", help="path where the final image will be saved")

    args = parser.parse_args()

    in_coords = read_file(args.input)
    out_coords = read_file(args.output)
    in_masses = [float(m[3]) * 5 for m in in_coords]
    out_coords = [el for el in out_coords if float(el[3]) > 0]
    out_masses = [float(m[3]) * 5 for m in out_coords]

    plt.scatter([float(x[1]) for x in in_coords], [float(y[2]) for y in in_coords], s=in_masses, c='b', alpha=0.5, label="initial")
    plt.scatter([float(x[1]) for x in out_coords], [float(y[2]) for y in out_coords], s=out_masses, c='r', alpha=0.5, label="final")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig(args.out_img)

if __name__ == "__main__":
    main()
