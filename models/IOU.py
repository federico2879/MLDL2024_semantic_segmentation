import numpy as np

def fast_hist(pred, target, num_classes):
    k = (target >= 0) & (target < num_classes)
    return np.bincount(num_classes * target[k].astype(int) + pred[k],
                       minlength = num_classes**2).reshape(num_classes, num_classes)

def per_class_iou(hist):
    epsilon = 1e-5
    return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)

def meanIOU(num_classes, pred, target):
  mIOU = 0
  IOU_classes = np.zeros([1,num_classes])  
  for i in range(len(pred)):    
      hist = fast_hist(pred[i].cpu().numpy(), target[i].cpu().numpy(), num_classes)
      IOU = per_class_iou(hist)
      IOU_classes = IOU_classes + IOU
      mIOU = mIOU + sum(IOU)/num_classes 
  return mIOU, IOU_classes
