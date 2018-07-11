import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from scipy import ndimage
import os
import ipdb


class CTTumorDataset_FreeSeg(Dataset):
    """"CT Tumor 2D Data loader"""

    def __init__(self, vol_data_dir, list_file, transform=None, norm=None):

        """
        Args:
            vol_data_dir (string): Directory with all the volumes.
            list_file: Path to the txt file with: volume file name + label/annotation file name + class label.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.n_classes = 3

        vol_names = []
        class_vecs = []

        with open(list_file, "r") as f:
            for line in f:
                items = line.split()

                vol_name = items[0]
                vol_name = os.path.join(vol_data_dir, vol_name)
                vol_names.append(vol_name)

                class_vec = items[1:]
                class_vec = [int(i) for i in class_vec]
                class_vecs.append(class_vec)

        self.vol_names = vol_names
        self.class_vecs = class_vecs

        self.transform = transform
        self.norm = norm

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and class
        """

        # image loader
        vol_name = self.vol_names[index]
        vol = Image.open(vol_name).convert('I')
        vol = vol.numpy()

        # class vector loader
        class_vec = self.class_vecs[index]
        class_vec = torch.FloatTensor(class_vec)

        if self.transform is not None:
            vol = self.transform(vol)
            
        vol = np.repeat(vol, 3, axis=0)
        vol = torch.from_numpy(vol).float();
        
        return vol, class_vec

    def __len__(self):
        return len(self.vol_names)
