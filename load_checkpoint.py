import torch
import numpy as np

def load_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
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

def load_checkpoint_adversarial(generator, discriminator, optimizer_G, optimizer_D, filename="checkpoint.pth.tar"):
    checkpoint = torch.load(filename)
    generator.load_state_dict(checkpoint['state_dict_gen'])
    discriminator.load_state_dict(checkpoint['state_dict_dis'])
    optimizer_G.load_state_dict(checkpoint['optimizer_gen'])
    optimizer_D.load_state_dict(checkpoint['optimizer_dis'])
    start_epoch = checkpoint['epoch']
    meanIOU = checkpoint['meanIOU']
    IOU = checkpoint['IOU']
    loss = checkpoint['loss']	
    return generator, discriminator, optimizer_G, optimizer_D, start_epoch, meanIOU, IOU, loss
    
