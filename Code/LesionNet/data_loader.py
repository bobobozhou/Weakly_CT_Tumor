import torch
import numpy as np
from torch.utils.data import Dataset
import transforms_3D.transforms_3d as transforms_3d
from PIL import Image
from scipy import ndimage
from scipy.io import loadmat
import os
import ipdb


class CTTumorDataset_FreeSeg(Dataset):
    """"CT Tumor 2D Data loader"""

    def __init__(self, vol_data_dir, mask_data_dir, list_file, transform_vol=None, transform_mask=None):

        """
        Args:
            vol_data_dir (string): Directory with all the volumes.
            list_file: Path to the txt file with: volume file name + label/annotation file name + class label.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.n_classes = 8

        vol_names = []
        mask_names = []
        class_vecs = []

        with open(list_file, "r") as f:
            for line in f:
                items = line.split()

                vol_name = items[0]
                vol_name = os.path.join(vol_data_dir, vol_name)
                vol_names.append(vol_name)

                mask_name = items[1]
                mask_name = os.path.join(mask_data_dir, mask_name)
                mask_names.append(mask_name)

                class_vec = items[2:]
                class_vec = [int(i) for i in class_vec]
                class_vecs.append(class_vec)

        self.vol_names = vol_names
        self.mask_names = mask_names
        self.class_vecs = class_vecs

        self.transform_vol = transform_vol
        self.transform_mask = transform_mask

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and class
        """

        # volume loader
        vol_name = self.vol_names[index]
        vol = np.array(loadmat(vol_name)['vol_crop'], dtype=float) + 1000
        vol = np.transpose(vol, [2, 0, 1])
        vol = torch.from_numpy(vol).float()

        if self.transform_vol is not None:
            vol = self.transform_vol(vol)

        # mask loader
        mask_name = self.mask_names[index]
        mask = np.array(loadmat(mask_name)['mask_crop'], dtype=float)
        mask = np.transpose(mask, [2, 0, 1])
        mask = torch.from_numpy(mask).float()

        if self.transform_mask is not None:
            mask = self.transform_mask(mask)

        # class vector loader
        class_vec = self.class_vecs[index]
        class_vec = torch.FloatTensor(class_vec)
        
        return vol, mask, class_vec

    def __len__(self):
        return len(self.vol_names)
