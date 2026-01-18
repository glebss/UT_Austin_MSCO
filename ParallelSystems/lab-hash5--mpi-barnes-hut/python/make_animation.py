import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import os
import cv2
from tqdm import tqdm

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
    parser.add_argument("input_folder", type=str, default="", help="Folder with output txts")
    parser.add_argument("n_steps", type=int, default=-1, help="Number of steps")
    
    args = parser.parse_args()

    input_files = os.listdir(args.input_folder)
    input_files = [f for f in input_files if f.endswith(".txt")]
    input_files = sorted(input_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    input_files = input_files[:args.n_steps]
    images = []
    video = cv2.VideoWriter(os.path.join(args.input_folder, f"animation_n_steps_{args.n_steps}.mp4"), \
                            cv2.VideoWriter_fourcc(*'mp4v'), 10, (640, 480))
    for i, input_file in enumerate(tqdm(input_files)):
        out_coords = read_file(os.path.join(args.input_folder, input_file))
        out_coords = [el for el in out_coords if float(el[3]) > 0]
        out_masses = [float(m[3]) * 5 for m in out_coords]
        plt.scatter([float(x[1]) for x in out_coords], [float(y[2]) for y in out_coords], s=out_masses, c='r', alpha=0.5, label="final")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        out_img_path = os.path.join(args.input_folder, f"out.png")
        plt.savefig(out_img_path)
        plt.close()
        video.write(cv2.imread(out_img_path))
    
if __name__ == "__main__":
    main()
