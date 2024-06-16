import torch
import numpy as np

def load_checkpoint(filename="checkpoint.pth.tar"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    meanIOU_tr = checkpoint['meanIOU_tr']
    IOU_tr = checkpoint['IOU_tr']
    loss_tr = checkpoint['loss_tr']
    meanIOU_val = checkpoint['meanIOU_val']
    IOU_val = checkpoint['IOU_val']
    loss_val = checkpoint['loss_val']
    return model, optimizer, start_epoch, meanIOU_tr, IOU_tr, loss_tr, meanIOU_val, IOU_val, loss_val

#model, optimizer, start_epoch, meanIOU_tr, IOU_tr, loss_tr, meanIOU_val, IOU_val, loss_val  = load_checkpoint()
