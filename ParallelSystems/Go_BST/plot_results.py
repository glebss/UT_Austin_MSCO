import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    out_plots_folder = "out_plots"
    input_name = "coarse.txt"
    os.makedirs(out_plots_folder, exist_ok=True)
    hash_workers = np.array([1, 2, 4, 8, 16, 32])
    labels_names = hash_workers.copy()
    labels_names[-1] = 100
    hash_workers_times = np.array([1346.14403, 1069.96532, 936.0756200000001, 851.4890299999998,
                                   832.2274900000001, 802.2958000000001])
    hash_workers_std = np.array([197.92182060947474, 96.03540342632816, 94.73546744897395, 101.50370326697002,
                                 79.3644060707941, 111.03261738570338])

    # Create a scatter plot with error bars
    plt.figure(figsize=(8, 6))  # Optional: Set the figure size

    # Scatter plot
    plt.scatter(hash_workers, hash_workers_times, label='mean', color='blue', marker='o')

    # Error bars
    plt.errorbar(hash_workers, hash_workers_times, yerr=hash_workers_std, linestyle='None',
                 color='red', capsize=4, label='std')

    # Customize the plot
    plt.xlabel('Num workers')
    plt.ylabel('Time, ms')
    plt.title(f'Tree compare time, {input_name}')
    plt.xticks(ticks=hash_workers, labels=labels_names)
    plt.legend()

    # Display or save the plot
    plt.grid(True)  # Optional: Add grid lines
    plt.tight_layout()  # Optional: Improve layout
    plt.savefig(os.path.join(out_plots_folder, input_name.split('.')[0] + "_comp_workers.png"))
    # plt.show()

if __name__ == "__main__":
    main()
