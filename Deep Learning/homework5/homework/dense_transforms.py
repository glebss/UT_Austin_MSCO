# Source: https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py
import numpy as np
import random
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, *args):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            args = tuple(np.array([-point[0], point[1]], dtype=point.dtype) for point in args)
        return (image,) + args

class RandomVerticalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, *args):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            args = tuple(np.array([point[0], -point[1]], dtype=point.dtype) for point in args)
        return (image,) + args

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, *args):
        for t in self.transforms:
            image, *args = t(image, *args)
        return (image,) + tuple(args)


class ColorJitter(T.ColorJitter):
    def __call__(self, image, *args):
        return (super().__call__(image),) + args

class GaussianBlur(T.GaussianBlur):
    def __call__(self, image, *args):
        return (super().__call__(image),) + args

class RandomErasing(T.RandomErasing):
    def __call__(self, image, *args):
        return (super().__call__(image),) + args

class RandomInvert(T.RandomInvert):
    def __call__(self, image, *args):
        return (super().__call__(image),) + args

class RandomShift:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, *args):
        if np.random.rand() > self.p:
            return (image,) + args

        image = np.array(image)
        h, w = image.shape[:2]
        x, y = args[0]
        x, y = (1 + x) / 2, (1 + y) / 2
        x, y = int(x * w), int(y * h)
        if x != 0:
            shift_x = np.random.randint(-x+1, w-x)
        else:
            shift_x = np.random.randint(0, w)
        if y != 0:
            shift_y = np.random.randint(-y+1, h-y)
        else:
            shift_y = np.random.randint(0, h)
        
        # clip shifts to [-10, 10]
        # shift_x = min(20, max(-20, shift_x))
        # shift_y = min(20, max(-20, shift_y))
        
        
        if shift_x < 0:
            image = image[:, -shift_x:, :]
            image = np.pad(image, ((0, 0), (0, -shift_x), (0, 0)), mode='constant', constant_values=0)
        elif shift_x > 0:
            image = image[:, :-shift_x, :]
            image = np.pad(image, ((0, 0), (shift_x, 0), (0, 0)), mode='constant', constant_values=0)
        
        if shift_y < 0:
            image = image[-shift_y:, :, :]
            image = np.pad(image, ((0, -shift_y), (0, 0), (0, 0)), mode='constant', constant_values=0)
        elif shift_y > 0:
            image = image[:-shift_y, :, :]
            image = np.pad(image, ((shift_y, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
        
        new_x = x + shift_x
        new_y = y + shift_y
        # to (-1, 1) range
        new_x = 2 * new_x / w - 1
        new_y = 2 * new_y / h - 1
        new_args = (np.array([new_x, new_y], dtype=args[0].dtype), )
        new_image = Image.fromarray(image)
        return (new_image, ) + new_args


class ToTensor(object):
    def __call__(self, image, *args):
        return (F.to_tensor(image),) + args
