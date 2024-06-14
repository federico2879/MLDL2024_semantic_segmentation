import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import random
from tensorboardX import SummaryWriter

from model.deeplab_multi import DeeplabMulti
from model.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet

def train_adv(model, discr):
    model.train()  
    discr.train()
  
    # labels for adversarial training
    source_label = 0
    target_label = 1

    # Inizialize loss
    loss_seg = 0
    loss_adv_target = 0
    loss_discr = 0
      
    for iter in len(train_loader):

        optimizer.zero_grad()
        opt_discr.zero_grad()

        ##### train G #####

        # don't accumulate grads in D
        for param in discr.parameters():
            param.requires_grad = False

        # train with source

        _, batch = sourceloader.__next__()

        images, labels, _, _ = batch
        images = images.to(device)
        labels = labels.squeeze(dim=1)
        labels = labels.long().to(device)

        pred = model(images)
        #pred = interp(pred)
        
        loss_seg = seg_loss(pred, labels)

        # proper normalization
        loss = loss / args.iter_size
        loss.backward()
        loss_seg_value1 += loss_seg1.item() / args.iter_size
        loss_seg_value2 += loss_seg2.item() / args.iter_size

        # train with target

        _, batch = targetloader_iter.__next__()
        images, _, _ = batch
        images = images.to(device)

        pred_target1, pred_target2 = model(images)
        pred_target1 = interp_target(pred_target1)
        pred_target2 = interp_target(pred_target2)

        D_out1 = model_D1(F.softmax(pred_target1))
        D_out2 = model_D2(F.softmax(pred_target2))

        loss_adv_target1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).to(device))

        loss_adv_target2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(source_label).to(device))

        loss = args.lambda_adv_target1 * loss_adv_target1 + args.lambda_adv_target2 * loss_adv_target2
        loss = loss / args.iter_size
        loss.backward()
        loss_adv_target_value1 += loss_adv_target1.item() / args.iter_size
        loss_adv_target_value2 += loss_adv_target2.item() / args.iter_size

        # train D

        # bring back requires_grad
        for param in model_D1.parameters():
            param.requires_grad = True

        for param in model_D2.parameters():
            param.requires_grad = True

        # train with source
        pred1 = pred1.detach()
        pred2 = pred2.detach()

        D_out1 = model_D1(F.softmax(pred1))
        D_out2 = model_D2(F.softmax(pred2))

        loss_D1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).to(device))

        loss_D2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(source_label).to(device))

        loss_D1 = loss_D1 / args.iter_size / 2
        loss_D2 = loss_D2 / args.iter_size / 2

        loss_D1.backward()
        loss_D2.backward()

        loss_D_value1 += loss_D1.item()
        loss_D_value2 += loss_D2.item()

        # train with target
        pred_target1 = pred_target1.detach()
        pred_target2 = pred_target2.detach()

        D_out1 = model_D1(F.softmax(pred_target1))
        D_out2 = model_D2(F.softmax(pred_target2))

        loss_D1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(target_label).to(device))

        loss_D2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(target_label).to(device))

        loss_D1 = loss_D1 / args.iter_size / 2
        loss_D2 = loss_D2 / args.iter_size / 2

        loss_D1.backward()
        loss_D2.backward()

        loss_D_value1 += loss_D1.item()
        loss_D_value2 += loss_D2.item()

        optimizer.step()
        opt_discr.step()
