import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, padding=padding, stride=stride)
        torch.nn.init.xavier_normal_(self.conv1.weight)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, padding=1, stride=1)
        torch.nn.init.xavier_normal_(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.max_pool = nn.MaxPool2d(3, 1)
    
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.bn1(out)
        out2 = self.conv2(out)
        out = self.relu(out + out2)
        out = self.bn2(out)
        out = self.max_pool(out)
        return out

class UpconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, padding=padding, stride=stride)
        torch.nn.init.xavier_normal_(self.conv1.weight)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        self.conv2 = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels,
                                        kernel_size=3, padding=1, stride=1)
        torch.nn.init.xavier_normal_(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv3 = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels,
                                        kernel_size=3, stride=1)
        torch.nn.init.xavier_normal_(self.conv3.weight)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels)
    
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.bn1(out)
        out2 = self.conv2(out)
        out = self.relu(out + out2)
        out = self.bn2(out)
        out = self.relu(self.conv3(out))
        out = self.bn3(out)
        return out


class CNNClassifier(torch.nn.Module):
    def __init__(self):
        """
        Your code here
        """
        super().__init__()
        block1 = []
        block1.append(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, padding=3, stride=2))
        block1.append(nn.ReLU())
        block1.append(nn.BatchNorm2d(num_features=32))
        block1.append(nn.MaxPool2d(3, 1))
        self.block1 = nn.Sequential(*block1)
        self.block2 = Block(32, 64, 3, 3, 2)
        self.block3 = Block(64, 128, 3, 0, 1)
        self.block4 = Block(128, 256, 3, 0, 1)
        for l in self.block1:
            if type(l) != torch.nn.modules.conv.Conv2d:
                continue
            torch.nn.init.xavier_normal_(l.weight)
            
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features=7*7, out_features=6)
        torch.nn.init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        x = (x - x.mean()) / x.std()
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.relu(torch.mean(out, dim=1))
        out = torch.flatten(out, start_dim=1)
        out = self.linear(out)
        return out


class FCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        super().__init__()
        block1 = []
        block1.append(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, padding=3, stride=2))
        block1.append(nn.ReLU())
        block1.append(nn.BatchNorm2d(num_features=32))
        block1.append(nn.MaxPool2d(3, 1))
        self.block1 = nn.Sequential(*block1)
        for l in self.block1:
            if type(l) != torch.nn.modules.conv.Conv2d:
                continue
            torch.nn.init.xavier_normal_(l.weight)

        self.block2 = Block(in_channels=32, out_channels=64, kernel_size=3, padding=3, stride=2)
        self.block3 = Block(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.block4 = Block(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.block5 = Block(in_channels=256, out_channels=320, kernel_size=3, padding=2, stride=1)
        # self.upconv_block00 = UpconvBlock(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.upconv_block0 = UpconvBlock(in_channels=320, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.upconv_block1 = UpconvBlock(in_channels=256, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.upconv_block2 = UpconvBlock(in_channels=128, out_channels=32, kernel_size=3, padding=2, stride=2)
        last_block = []
        last_block.append(nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, padding=0, stride=1))
        last_block.append(nn.ReLU())
        last_block.append(nn.BatchNorm2d(num_features=32))
        last_block.append(nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, padding=0, stride=1))
        last_block.append(nn.ReLU())
        last_block.append(nn.BatchNorm2d(num_features=16))
        last_block.append(nn.ConvTranspose2d(in_channels=16, out_channels=5, kernel_size=5, padding=3, stride=2))
        self.last_block = nn.Sequential(*last_block)
        for l in self.last_block:
            if type(l) != torch.nn.modules.conv.ConvTranspose2d:
                continue
            torch.nn.init.xavier_normal_(l.weight)
        # self.last_conv1 = nn.ConvTranspose2d(in_channels=128, out_channels=5, kernel_size=7, padding=0, stride=2) 
        # self.last_conv = nn.ConvTranspose2d(in_channels=128, out_channels=5, kernel_size=7, padding=0, stride=2)
        self.mean = nn.Parameter(torch.tensor([67.01374067, 67.75714183, 71.10103019]).reshape(1, 3, 1, 1))
        self.std = nn.Parameter(torch.tensor([47.12723543, 42.4388973, 44.88066532]).reshape(1, 3, 1, 1))
        self.min_w = 32
        self.min_h = 32


    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,5,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        h, w = x.shape[-2:]
        x = (x - self.mean) / self.std # 3, 96, 128
        diff_x = self.min_w - x.shape[2]
        diff_y = self.min_h - x.shape[3]
        if diff_x > 0:
            x = nn.functional.pad(x, (0, 0, 0, diff_x), 'constant', 0)
        if diff_y > 0:
            x = nn.functional.pad(x, (0, diff_y, 0, 0), 'constant', 0)
        out1 = self.block1(x) # 64, 46, 62
        out2 = self.block2(out1) # 128, 23, 31
        out3 = self.block3(out2) # 256, 21, 29
        out4 = self.block4(out3) # 512, 19, 27
        out5 = self.block5(out4)
        # out_up00 = self.upconv_block00(out5) 
        out_up0 = self.upconv_block0(out5)
        out_up1 = self.upconv_block1(torch.cat([out_up0, out3], dim=1)) # 64, 23, 31
        out_up2 = self.upconv_block2(torch.cat([out_up1, out2], dim=1)) # 32, 45, 61
        out_up2 = nn.functional.pad(out_up2, (0, 1, 0, 1), 'constant', 0) # 32, 46, 62
        out = self.last_block(torch.cat([out_up2, out1], dim=1))[..., :h, :w]
        return out


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r

# if __name__ == "__main__":
#     model = FCN()
#     shape=[(2**i, 2**i) for i in range(10)] + [(2**(5-i), 2**i) for i in range(5)]
#     # shape = [(32, 1), (16, 2), (2, 16)]
#     for s in shape:
#         input = torch.zeros(1, 3, *s)
#         try:
#             out = model(input)
#             print(out.shape[2:] == input.shape[2:])
#             print('works for ', out.shape[2:])
#         except:
#             print("doesn't work for ", s)