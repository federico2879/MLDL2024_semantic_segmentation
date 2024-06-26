from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

class CityScapes(Dataset):
    def __init__(self, root_dir, split = 'train', transform=None, label_transform=None):
        super(CityScapes, self).__init__()
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images', split)
        self.label_dir = os.path.join(root_dir, 'gtFine', split)
        self.transform = transform
        self.label_transform = label_transform
        self.images = os.listdir(self.image_dir)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_name = img_name.replace('leftImg8bit', 'gtFine_labelTrainIds')
        label_path = os.path.join(self.label_dir, label_name)
        
        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)
        

        if self.transform is not None:
            image = self.transform(image)

        if self.label_transform is not None:
            label = self.label_transform(label)

        label_array = np.array(label)
        label_array = label_array.astype(np.int32)
        label_tensor = torch.tensor(label_array)
        
        return image, label_tensor

    def __len__(self):
        return len(self.images)
