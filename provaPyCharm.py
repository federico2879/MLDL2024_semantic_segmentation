import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class CityscapesDataset(Dataset):
    def _init_(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_dir = os.path.join(root_dir, 'leftImg8bit', split)
        self.label_dir = os.path.join(root_dir, 'gtFine', split)
        self.images = os.listdir(self.image_dir)

    def _len_(self):
        return len(self.images)

    def _getitem_(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_name = img_name.replace('leftImg8bit', 'gtFine_labelIds')
        label_path = os.path.join(self.label_dir, label_name)

        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)

        if self.transform:
            image, label = self.transform(image, label)

        return image, label