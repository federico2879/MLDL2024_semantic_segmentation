import torch
import torch.nn.functional as F

def train_adv(model, discr, seg_loss, bce_loss, targetloader, sourceloader, optimizer, opt_discr, device, num_classes):
    model.train()
    discr.train()
  
    # Labels for adversarial training
    source_label = 0
    target_label = 1
    
    # Create iterators
    sourceloader_iter = iter(sourceloader)
    targetloader_iter = iter(targetloader)
    
    max_iterations = min(len(targetloader), len(sourceloader))
    for idx in range(max_iterations):

        optimizer.zero_grad()
        opt_discr.zero_grad()

        ##### Train G #####

        # Don't accumulate grads in D
        for param in discr.parameters():
            param.requires_grad = False

        # Train with source
        batch = next(sourceloader_iter)
        images, labels = batch
        images = images.to(device)
        labels = labels.squeeze(dim=1).long().to(device)

        pred = model(images)[0]
        
        loss_seg = seg_loss(pred, labels)
        loss_seg.backward()

        # Train with target
        batch = next(targetloader_iter)
        images, _ = batch
        images = images.to(device)

        pred_target = model(images)[0]

        D_out = discr(F.softmax(pred_target, dim=1))

        loss_adv_target = bce_loss(D_out, torch.full(D_out.shape, source_label, device=device, dtype=torch.float))

        loss_adv_target.backward()

        # Train D

        # Bring back requires_grad
        for param in discr.parameters():
            param.requires_grad = True

        # Train with source
        pred = pred.detach()

        D_out = discr(F.softmax(pred, dim=1))

        loss_D = bce_loss(D_out, torch.full(D_out.shape, source_label, device=device, dtype=torch.float))
        loss_D.backward()

        # Train with target
        pred_target = pred_target.detach()
        
        D_out = discr(F.softmax(pred_target, dim=1))

        loss_D = bce_loss(D_out, torch.full(D_out.shape, target_label, device=device, dtype=torch.float))
        loss_D.backward()

        optimizer.step()
        opt_discr.step()
