# Real-time Domain Adaptation in Semantic Segmentation

`Abstract`: One of the biggest challenge in Semantic Segmentation is labeling real-world datasets, since every pixel requires a label. In order to solve this problem we used a synthetic dataset from the famous videogame “GTA V”, whose images have already been labeled by the creators of the game. Training deep learning models on this dataset and using them in real situations introduces the problem of domain shift. In this work, we firstly analyzed performances of two different neural networks, 
Deeplabv2 and BiSeNet, on a real-world dataset, "Cityscapes”. Then we trained the BiSeNet network on the GTA V dataset and evaluated the training on Cityscapes. In conclusion , we evaluated the problem of domain shift and we tried to improve performances using data augmentations and the adversarial approach, which consists in trying to fool a neural network trained to distinguish real and synthetic images.


## Datasets

To download the dataset use the following download links.

**Cityscapes**: https://drive.google.com/file/d/1Qb4UrNsjvlU-wEsR9d7rckB0YS_LXgb2/view?usp=sharing

**GTA5**: https://drive.google.com/file/d/1xYxlcMR2WFCpayNrW2-Rb7N-950vvl23/view?usp=sharing
