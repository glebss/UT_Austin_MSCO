import os
import shutil
from tqdm import tqdm

def main():
    data_path = "drive_data"
    train_folder = os.path.join(data_path, "train")
    val_folder = os.path.join(data_path, "valid")
    data_files = os.listdir(train_folder) + os.listdir(val_folder)
    maps = set(['_'.join(f.split('_')[:-1]) for f in data_files])
    files_train, files_test = [], []
    for map in maps:
        files_map = [f for f in data_files if map in f]
        files_map = sorted(files_map)
        files_map_train = files_map[:-100]
        files_map_val = files_map[-100:]
        files_train += files_map_train
        files_test += files_map_val
    
    out_folder_train = os.path.join("drive_data_new/train")
    out_folder_val = os.path.join("drive_data_new/valid")
    os.makedirs(out_folder_train, exist_ok=True)
    os.makedirs(out_folder_val, exist_ok=True)
    for f in tqdm(files_train):
        src = os.path.join(train_folder, f)
        if not os.path.exists(src):
            src = os.path.join(val_folder, f)
        assert os.path.exists(src)
        dest = os.path.join(out_folder_train, f)
        shutil.copy(src, dest)
    for f in tqdm(files_test):
        src = os.path.join(train_folder, f)
        if not os.path.exists(src):
            src = os.path.join(val_folder, f)
        assert os.path.exists(src)
        dest = os.path.join(out_folder_val, f)
        shutil.copy(src, dest)
    # print(len(val_files))
    # print(len(train_files))


if __name__ == '__main__':
    main()
