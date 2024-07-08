from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import shutil

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


    def __len__(self):
        return len(self.images)


def Modified_CityScapes(start_path):
    # Extract images and copy
    end_path = ['/gtFine/train', '/gtFine/val', '/images/train', '/images/val']
    for str in end_path:
        origin = start_path + str
        for subdir in os.listdir(origin):
            path_subdir = os.path.join(origin, subdir)
            if os.path.isdir(path_subdir):
                for file in os.listdir(path_subdir):
                    path_file_origin = os.path.join(path_subdir, file)
                    shutil.copy(path_file_origin, origin)

        # Delete subdirectory
        for subdir in os.listdir(origin):
            path_subdir = os.path.join(origin, subdir)
            if os.path.isdir(path_subdir):
                shutil.rmtree(path_subdir)
        
        return image, label_tensor
