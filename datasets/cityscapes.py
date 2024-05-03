from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class CityScapes(Dataset):
    def _init_(self, labels_path, root_dir, split = 'train', transform=None, target_transform=None):
        super(CityScapes, self)._init_()
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images', split)
        self.label_dir = os.path.join(root_dir, 'gtFine', split)
        self.transform = transform
        self.target_transform = target_transform
        self.images = os.listdir(self.image_dir)

    def _getitem_(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_name = img_name.replace('leftImg8bit', 'gtFine_labelTrainIds')
        label_path = os.path.join(self.label_dir, label_name)

        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def _len_(self):
        return len(self.images)