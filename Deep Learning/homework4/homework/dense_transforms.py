# Source: https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py
import numpy as np
from PIL import Image
# import cv2
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, *args):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            args = tuple(np.array([(image.width-x1, y0, image.width-x0, y1) for x0, y0, x1, y1 in boxes])
                         for boxes in args)
        return (image,) + args


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, *args):
        for t in self.transforms:
            image, *args = t(image, *args)
        return (image,) + tuple(args)


class Normalize(T.Normalize):
    def __call__(self, image, *args):
        return (super().__call__(image),) + args


class ColorJitter(T.ColorJitter):
    def __call__(self, image, *args):
        return (super().__call__(image),) + args

class GaussianBlur(T.GaussianBlur):
    def __call__(self, image, *args):
        return (super().__call__(image),) + args


class ToTensor(object):
    def __call__(self, image, *args):
        return (F.to_tensor(image),) + args


class ToHeatmap(object):
    def __init__(self, radius=2):
        self.radius = radius

    def __call__(self, image, *dets):
        peak, size = detections_to_heatmap(dets, image.shape[1:], radius=self.radius)
        return image, peak, size

class RandomCropResize:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, *args):
        image = np.array(image)
        h, w = image.shape[:2]
        if np.random.rand() > self.p:
            return (image,) + args

        crop_h_frac, crop_w_frac = np.random.uniform(0.5, 1.0, size=2)
        new_h, new_w = int(np.ceil(crop_h_frac * h)), int(np.ceil(crop_w_frac * w))
        new_h = min(new_h, h)
        new_w = min(new_w, w)
        start_h = np.random.randint(low=0, high=h-new_h + 1)
        start_w = np.random.randint(low=0, high=w-new_w + 1)
        new_image = image[start_h:start_h + new_h, start_w:start_w + new_w, :]
        kh, kw = h / new_h, w / new_w
        new_dets = []
        for det in args:
            new_det = []
            for bbox in det:
                xmin, ymin, xmax, ymax = bbox
                if xmax <= start_w or xmin >= start_w + new_w:
                    continue
                if ymax <= start_h or ymin >= start_h + new_h:
                    continue
                xmin = max(start_w, xmin) - start_w
                xmax = min(xmax, start_w + new_w) - start_w
                ymin = max(ymin, start_h) - start_h
                ymax = min(ymax, start_h + new_h) - start_h
                xmin, xmax = int(kw * xmin), int(kw * xmax)
                ymin, ymax = int(kh *ymin), int(kh * ymax)
                new_det.append([xmin, ymin, xmax, ymax])
            new_det = np.array(new_det, dtype=det.dtype)
            new_dets.append(new_det)
        new_image = Image.fromarray(new_image)
        new_image = new_image.resize((w, h))
        # new_image = cv2.resize(new_image, (w, h))
        # new_image = Image.fromarray(new_image)
        new_dets = tuple(new_dets)
        return (image, ) + new_dets


def detections_to_heatmap(dets, shape, radius=2, device=None):
    with torch.no_grad():
        size = torch.zeros((2, shape[0], shape[1]), device=device)
        peak = torch.zeros((len(dets), shape[0], shape[1]), device=device)
        for i, det in enumerate(dets):
            if len(det):
                det = torch.tensor(det.astype(float), dtype=torch.float32, device=device)
                cx, cy = (det[:, 0] + det[:, 2] - 1) / 2, (det[:, 1] + det[:, 3] - 1) / 2
                x = torch.arange(shape[1], dtype=cx.dtype, device=cx.device)
                y = torch.arange(shape[0], dtype=cy.dtype, device=cy.device)
                gx = (-((x[:, None] - cx[None, :]) / radius)**2).exp()
                gy = (-((y[:, None] - cy[None, :]) / radius)**2).exp()
                gaussian, id = (gx[None] * gy[:, None]).max(dim=-1)
                mask = gaussian > peak.max(dim=0)[0]
                det_size = (det[:, 2:] - det[:, :2]).T / 2
                size[:, mask] = det_size[:, id[mask]]
                peak[i] = gaussian
        return peak, size
