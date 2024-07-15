# Real-time Domain Adaptation in Semantic Segmentation

`Abstract`: One of the biggest challenge in Semantic Segmentation is labeling real-world datasets, since every pixel requires a label. In order to solve this problem we used a synthetic dataset from the famous videogame “GTA V”, whose images have already been labeled by the creators of the game. Training deep learning models on this dataset and using them in real situations introduces the problem of domain shift. In this work, we firstly analyzed performances of two different neural networks, 
Deeplabv2 and BiSeNet, on a real-world dataset, "Cityscapes”. Then we trained the BiSeNet network on the GTA V dataset and evaluated the training on Cityscapes. In conclusion , we evaluated the problem of domain shift and we tried to improve performances using data augmentations and the adversarial approach, which consists in trying to fool a neural network trained to distinguish real and synthetic images.


## Datasets

**Cityscapes**
- Download from https://drive.google.com/file/d/1Qb4UrNsjvlU-wEsR9d7rckB0YS_LXgb2/view?usp=sharing and unzip
- To obtain an usable dataset apply *Modified_CityScapes* (delete city folders)

**GTA5**
- Download from https://drive.google.com/file/d/1xYxlcMR2WFCpayNrW2-Rb7N-950vvl23/view?usp=sharing and unzip
- To obtain an usable dataset apply *...* (produce labels)

## Models

**DeepLabV2**: *get_deeplab_v2* in deeplabv2.py (take the pretrainde ResNet 101)

**BiSeNet**: *BiSeNet* in build_bisenet.py (set the context path)

## Training and test

- `training function`: *train* in train.py
- `test function`: *test* in train.py
- `adversarial training function`: *train_adv* in train_adv.py
- `adapting learning rate function`: *poly_lr_scheduler* in utils.py
- `resumption training function`: *load_checkpoint* in load_checkpoint.py
- `resumption adversarial training function`: *load_checkpoint_adversarial* in load_checkpoint.py

## Other information

- Before training we modified GTAV dataset with *_*
- There is some cells in notebooks where we upload a file, due to contrains regarding run-time we downloaded the checkpoint and we uploaded it in the input section of Kaggle to start the run again.
