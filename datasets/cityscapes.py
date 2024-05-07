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
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_name = img_name.replace('leftImg8bit', 'gtFine_labelTrainIds')
        label_path = os.path.join(self.label_dir, label_name)
        
        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)
        #label = torch.cat([label] * 3, dim=0)

        if self.transform is not None:
            image, label = self.transform(image, label)

        return image, label

    def __len__(self):
        return len(self.images)
