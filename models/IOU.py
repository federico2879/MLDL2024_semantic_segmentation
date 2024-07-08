import numpy as np

def fast_hist(pred, target, num_classes):
    k = (target >= 0) & (target < num_classes)
    return np.bincount(num_classes * target[k].astype(int) + pred[k],
                       minlength = num_classes**2).reshape(num_classes, num_classes)

def per_class_iou(hist):
    epsilon = 1e-5
    return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)

