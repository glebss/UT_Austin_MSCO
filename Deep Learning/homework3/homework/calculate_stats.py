import os
import cv2
import numpy as np
from tqdm import tqdm

def main(dataset_path):
    images_list = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
    means = []
    stds = []
    for img_path in tqdm(images_list):
        img = cv2.imread(os.path.join(dataset_path, img_path))
        means.append(np.mean(img, axis=(0,1)))
        stds.append(np.std(img, axis=(0,1)))
    means = np.array(means)
    stds = np.array(stds)
    means = np.mean(means, axis=0)
    stds = np.mean(stds, axis=0)
    print(means)
    print(stds)

if __name__ == "__main__":
    main("/home/kadmin/gleb/UTAustin_MSCO/Deep_Learning/homework3/dense_data/train")
