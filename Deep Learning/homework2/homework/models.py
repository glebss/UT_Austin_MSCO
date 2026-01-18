import torch
from torch import nn

class CNNClassifier(torch.nn.Module):
    def __init__(self):
        """
        Your code here
        """
        super().__init__()
        self.layers = []
        self.layers.append(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(3, 1))
        self.layers.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(3, 1))
        self.layers.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(3, 1))
        # self.layers.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3))
        # self.layers.append(nn.ReLU())
        # self.layers.append(nn.MaxPool2d(3, 1))
        self.net = nn.Sequential(*self.layers)

        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features=48*48, out_features=6)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        # x = x / 255.0
        out = self.net(x)
        out = self.relu(torch.mean(out, dim=1))
        out = torch.flatten(out, start_dim=1)
        out = self.linear(out)
        return out


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r

if __name__ == "__main__":
    classifier = CNNClassifier()
    input = torch.rand(5, 3, 64, 64)
    out = classifier(input)