import os
import numpy as np
from PIL import Image
from abc import ABCMeta
from dataclasses import dataclass
from typing import Tuple


'''
CLASS FOR GENERATION OF LABELED IMAGES OF GTA V DATASET

This class can be used to transform segmentated images of GTAV into labeled images, where each label is 
represented by a different hue of grey. Classes colors and id are taken from the class GTA5Labels_TaskCV2017.
'''

class BaseGTALabels(metaclass=ABCMeta):
    pass


@dataclass
class GTA5Label:
    ID: int
    color: Tuple[int, int, int]


class GTA5Labels_TaskCV2017(BaseGTALabels):
    road = GTA5Label(ID=0, color=(128, 64, 128))
    sidewalk = GTA5Label(ID=1, color=(244, 35, 232))
    building = GTA5Label(ID=2, color=(70, 70, 70))
    wall = GTA5Label(ID=3, color=(102, 102, 156))
    fence = GTA5Label(ID=4, color=(190, 153, 153))
    pole = GTA5Label(ID=5, color=(153, 153, 153))
    light = GTA5Label(ID=6, color=(250, 170, 30))
    sign = GTA5Label(ID=7, color=(220, 220, 0))
    vegetation = GTA5Label(ID=8, color=(107, 142, 35))
    terrain = GTA5Label(ID=9, color=(152, 251, 152))
    sky = GTA5Label(ID=10, color=(70, 130, 180))
    person = GTA5Label(ID=11, color=(220, 20, 60))
    rider = GTA5Label(ID=12, color=(255, 0, 0))
    car = GTA5Label(ID=13, color=(0, 0, 142))
    truck = GTA5Label(ID=14, color=(0, 0, 70))
    bus = GTA5Label(ID=15, color=(0, 60, 100))
    train = GTA5Label(ID=16, color=(0, 80, 100))
    motocycle = GTA5Label(ID=17, color=(0, 0, 230))
    bicycle = GTA5Label(ID=18, color=(119, 11, 32))

    list_ = [
        road,
        sidewalk,
        building,
        wall,
        fence,
        pole,
        light,
        sign,
        vegetation,
        terrain,
        sky,
        person,
        rider,
        car,
        truck,
        bus,
        train,
        motocycle,
        bicycle,
    ]



def create_label_map(image_rgb, label_class):
    labels = label_class.list_
    label_map = np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=int)

    for label in labels:
        mask = np.all(image_rgb == np.array(label.color).reshape(1, 1, 3), axis=-1)
        label_map[mask] = label.ID

    label_map = label_map[:, :, np.newaxis]
    return label_map


def process_images(input_folder, output_folder, label_class):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            image = Image.open(img_path)
            image_rgb = np.array(image.convert('RGB'))

            label_map = create_label_map(image_rgb, label_class)
            label_map_image = Image.fromarray(label_map.squeeze().astype(np.uint8))

            output_path = os.path.join(output_folder, filename)
            label_map_image.save(output_path)
            print(f"Output file: {filename}")


'''
if __name__ == '__main__':
    in_folder = ""  # Replace with the path to your input folder
    out_folder = ""  # Replace with the path to your output folder
    GTA5Labels = GTA5Labels_TaskCV2017()
    process_images(in_folder, out_folder, GTA5Labels_TaskCV2017)
'''