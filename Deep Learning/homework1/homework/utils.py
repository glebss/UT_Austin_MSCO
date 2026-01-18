from PIL import Image
import csv
import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv
        

        WARNING: Do not perform data normalization here. 
        """

        super().__init__()
        self.dataset_path = dataset_path
        labels_path = os.path.join(dataset_path, "labels.csv")
        labels2ids = {l : i for i, l in enumerate(LABEL_NAMES)}
        self.to_tensor = transforms.ToTensor()
        self.data = []
        with open(labels_path, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                img_name, label = row[:2]
                self.data.append((img_name, labels2ids[label]))

    
    def __len__(self):
        """
        Your code here
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        img = Image.open(os.path.join(self.dataset_path, self.data[idx][0]))
        img = self.to_tensor(img)
        return img, self.data[idx][1]
        


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()


# if __name__ == "__main__":
#     dataset_path = "/Users/gleb/Documents/UT_Austin_MSCO/Deep Learning/homework1/data/valid"
#     dataset = SuperTuxDataset(dataset_path)
#     img, lbl = dataset[10]
