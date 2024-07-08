import torch
import torchvision
import numpy as np
from MLDL2024_semantic_segmentation.models.IOU import fast_hist, per_class_iou

def train(model, optimizer, train_loader, loss_fn, num_classes):
    model.train()
    running_loss = 0.0
    confmat = np.zeros([num_classes,num_classes])

    for batch_idx, (inputs, targets) in enumerate(train_loader):

        inputs, targets = inputs.cuda(), targets.cuda()

        # Compute prediction and loss
        outputs =  model(inputs)
       
        # Reshaping tensor
        targets = targets.squeeze(dim=1)
        targets = targets.long()

        loss = loss_fn(outputs[0], targets)

        # Backpropagation
        optimizer.zero_grad() # reset gradients of parameters
        loss.backward()  # backpropagate the prediction loss
        optimizer.step() # update model

        running_loss += loss.item()
        _, predicted = outputs[0].max(1)

        # Compute Confusion matrix
        for i in range(len(predicted)):    
            confmat += fast_hist(predicted[i].cpu().numpy(), targets[i].cpu().numpy(), num_classes)

    # Compute metrics
    iou_class = per_class_iou(confmat)
    miou = sum(iou_class)/num_classes
    train_loss = running_loss / len(train_loader)
    
    return miou, iou_class, train_loss

def test(model, test_loader, loss_fn, num_classes):
    model.eval()
    test_loss = 0
    confmat = np.zeros([num_classes,num_classes])
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            
            # Reshaping tensor
            targets = targets.squeeze(dim=1)
            targets = targets.long()
            
            loss = loss_fn(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)

            # Compute Confusion matrix
            for i in range(len(predicted)):    
                confmat += fast_hist(predicted[i].cpu().numpy(), targets[i].cpu().numpy(), num_classes)
            

    # Compute metrics
    iou_class = per_class_iou(confmat)
    miou = sum(iou_class)/num_classes
    test_loss = test_loss / len(test_loader)
    
    return miou, iou_class, test_loss
