from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

class CityScapes(Dataset):
    def __init__(self, root_dir, split = 'train', transform=None):
        super(CityScapes, self).__init__()
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images', split)
        self.label_dir = os.path.join(root_dir, 'gtFine', split)
        self.transform = transform
        self.images = os.listdir(self.image_dir)

    def __getitem__(self, idx):
        image = Image.open(self.image_dir).convert('RGB')
        label = Image.open(self.label_dir)
        #label = torch.cat([label] * 3, dim=0)

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, label

    def __len__(self):
        return len(self.images)
