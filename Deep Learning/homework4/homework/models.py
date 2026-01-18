import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image


def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """
    pad_size = max_pool_ks // 2
    ret = F.max_pool2d(heatmap.unsqueeze(0), kernel_size=max_pool_ks, stride=1, padding=pad_size).squeeze()
    mask = (heatmap == ret) & (heatmap > min_score)
    coords = torch.nonzero(mask)
    vals = [(heatmap[c[0], c[1]].item(), c[1].item(), c[0].item()) for c in coords]
    vals = sorted(vals, key=lambda x: x[0], reverse=True)
    return vals[:max_det]


class Block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        net = []
        net.append(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 1), padding=(1, 0)))
        net.append(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 3), padding=(0, 1)))
        net.append(nn.BatchNorm2d(num_features=in_channels))
        net.append(nn.ReLU(inplace=True))
        net.append(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 1), padding=(1, 0)))
        net.append(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 3), padding=(0, 1)))
        net.append(nn.BatchNorm2d(num_features=in_channels))
        net.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*net)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.net(x)
        out = out + x
        return out

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        net = []
        net.append(nn.BatchNorm2d(num_features=in_channels))
        net.append(nn.ReLU(inplace=True))
        net.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.net = nn.Sequential(*net)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.net(x)


class CNNClassifierBlock(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride, bias=False)
            self.c2 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.c3 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.b1 = torch.nn.BatchNorm2d(n_output)
            self.b2 = torch.nn.BatchNorm2d(n_output)
            self.b3 = torch.nn.BatchNorm2d(n_output)
            self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        def forward(self, x):
            return F.relu(self.b3(self.c3(F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)))))))) + self.skip(x))


class Detector(torch.nn.Module):
    class UpBlock(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride, output_padding=1)
            nn.init.normal_(self.c1.weight, mean=0, std=0.001)

        def forward(self, x):
            return F.relu(self.c1(x))

    def __init__(self, layers=[32, 64, 128, 256], n_output_channels=5, kernel_size=3, use_skip=True):
        super().__init__()
        self.input_mean = torch.Tensor([0.2788, 0.2657, 0.2629])
        self.input_std = torch.Tensor([0.2064, 0.1944, 0.2252])

        c = 3
        self.use_skip = use_skip
        self.n_conv = len(layers)
        skip_layer_size = [3] + layers[:-1]
        for i, l in enumerate(layers):
            self.add_module('conv%d' % i, CNNClassifierBlock(c, l, kernel_size, 2))
            c = l
        for i, l in list(enumerate(layers))[::-1]:
            self.add_module('upconv%d' % i, self.UpBlock(c, l, kernel_size, 2))
            c = l
            if self.use_skip:
                c += skip_layer_size[i]
        self.classifier = torch.nn.Conv2d(c, n_output_channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.num_classes = 3

    def forward(self, x):
        z = (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device)
        up_activation = []
        for i in range(self.n_conv):
            # Add all the information required for skip connections
            up_activation.append(z)
            z = self._modules['conv%d'%i](z)

        for i in reversed(range(self.n_conv)):
            z = self._modules['upconv%d'%i](z)
            # Fix the padding
            z = z[:, :, :up_activation[i].size(2), :up_activation[i].size(3)]
            # Add the skip connection
            if self.use_skip:
                z = torch.cat([z, up_activation[i]], dim=1)
        out = self.classifier(z)
        out[:, 3:, ...] = F.relu(out[:, 3:, ...]) 
        return out
    
    def detect(self, image):
        """
           Your code here.
           Implement object detection here.
           @image: 3 x H x W image
           @return: Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
                    return no more than 30 detections per image per class. You only need to predict width and height
                    for extra credit. If you do not predict an object size, return w=0, h=0.
           Hint: Use extract_peak here
           Hint: Make sure to return three python lists of tuples of (float, int, int, float, float) and not a pytorch
                 scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
                 out of memory.
        """
        res = []
        with torch.no_grad():
            image = image.unsqueeze(0)
            out = self.forward(image).squeeze(0)
            out_hms, out_sizes = torch.split(out, [3, 2], dim=0)
            for nc in range(self.num_classes):
                hm = out_hms[nc, ...].sigmoid()
                ret = extract_peak(hm, min_score=0.4, max_det=30)
                ret_with_hw = [(r[0], r[1], r[2], out_sizes[0, r[2], r[1]].item(), out_sizes[1, r[2], r[1]].item()) for r in ret]
                res.append(ret_with_hw)
        return res


def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    """
    Shows detections of your detector
    """
    # from .utils import DetectionSuperTuxDataset
    # dataset = DetectionSuperTuxDataset('dense_data/valid', min_size=0)
    # import torchvision.transforms.functional as TF
    # from pylab import show, subplots
    # import matplotlib.patches as patches

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # fig, axs = subplots(3, 4)
    # model = load_model().eval().to(device)
    # for i, ax in enumerate(axs.flat):
    #     im, kart, bomb, pickup = dataset[i]
    #     ax.imshow(TF.to_pil_image(im), interpolation=None)
    #     for k in kart:
    #         ax.add_patch(
    #             patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='r'))
    #     for k in bomb:
    #         ax.add_patch(
    #             patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='g'))
    #     for k in pickup:
    #         ax.add_patch(
    #             patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='b'))
    #     detections = model.detect(im.to(device))
    #     for c in range(3):
    #         for s, cx, cy, w, h in detections[c]:
    #             ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color='rgb'[c]))
    #     ax.axis('off')
    # show()

    # detector = Detector()
    # input = torch.rand(1, 3, 256, 256)
    # out = detector(input)
    # hm = torch.rand(256, 256)
    # ret = extract_peak(hm, min_score=0.5, max_pool_ks=3)

    # detector = Detector()
    # out = detector.forward(img)
    # out_det = detector.detect(img)
    # print()