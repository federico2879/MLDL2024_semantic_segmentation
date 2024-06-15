import torch
from torch.autograd import Variable
import torch.nn.functional as F

def train_adv(model, discr, seg_loss, bce_loss, targetloader, sourceloader, num_classes):
    model.train()  
    discr.train()
  
    # labels for adversarial training
    source_label = 0
    target_label = 1

    # Inizialize loss
    loss_seg_value = 0
    loss_adv_target_value = 0
    loss_D_value = 0
      
    for iter in len(train_loader):

        optimizer.zero_grad()
        opt_discr.zero_grad()

        ##### train G #####

        # don't accumulate grads in D
        for param in discr.parameters():
            param.requires_grad = False

        # train with source

        _, batch = sourceloader_iter.__next__()

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
        loss_seg_value += loss_seg1.item() / args.iter_size

        # train with target

        _, batch = targetloader_iter.__next__()
        images, _, _ = batch
        images = images.to(device)

        pred_target = model(images)
        #pred_target = interp_target(pred_target)

        D_out = discr(F.softmax(pred_target))

        loss_adv_target = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))

        loss_adv_target = loss_adv_target / args.iter_size
        loss_adv_target.backward()
        loss_adv_target_value += loss_adv_target.item() / args.iter_size

        # train D

        # bring back requires_grad
        for param in discr.parameters():
            param.requires_grad = True

        # train with source
        pred = pred.detach()

        D_out = discr(F.softmax(pred))

        loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))
        loss_D = loss_D / args.iter_size / 2
        loss_D.backward()

        loss_D_value += loss_D.item()

        # train with target
        pred_target = pred_target.detach()
        
        D_out = discr(F.softmax(pred_target))

        loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(target_label).to(device))

        loss_D = loss_D / args.iter_size / 2
        loss_D.backward()
        loss_D_value += loss_D.item()

        optimizer.step()
        opt_discr.step()
