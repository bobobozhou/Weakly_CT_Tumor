import random
import math
import numbers
import collections
import numpy as np
import torch
from skimage.transform import resize

try:
    import accimage
except ImportError:
    accimage = None

__all__ = ["Compose", "ToTensor", "Resize", "MakeNChannel" "Normalize", "RandomCrop"]


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class Resize(object):
    """Resize the input volume to the given size.
    Args:
        size (sequence or int): Desired output size.
    """

    def __init__(self, size, order):
        self.size = size
        self.order = order

    def __call__(self, vol):
        """
        Args:
            tensor (Tensor): Tensor volume of size (C, H, W, D) to be resized.
        Returns:
            tensor (Tensor): Resized Tensor volume
        """

        vol = vol.numpy()

        vol_new = resize(vol, (self.size[0], self.size[1], self.size[2]),
                         order=self.order,
                         preserve_range=True,
                         mode='symmetric')

        vol_new = torch.from_numpy(vol_new).float()

        return vol_new

    def randomize_parameters(self):
        pass


class MakeNChannel(object):
    """Convert from GrayScale to n input"""

    def __init__(self, n):
        self.n = n

    def __call__(self, vol):
        """
        Args:
            tensor (Tensor): Tensor volume to increase channels
        Returns:
            tensor (Tensor): Tensor volume
        """

        vol = vol.numpy()

        vol_new = np.repeat(vol[np.newaxis, :, :, :], self.n, axis=0)

        vol_new = torch.from_numpy(vol_new).float()

        return vol_new

    def randomize_parameters(self):
        pass


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor volume of size (C, H, W, D) to be normalized.
        Returns:
            Tensor: Normalized volume.
        """
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

    def randomize_parameters(self):
        pass