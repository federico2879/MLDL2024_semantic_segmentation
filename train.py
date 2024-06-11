import torch
import torchvision
import gc
import numpy as np
from MLDL2024_semantic_segmentation.models.IOU import meanIOU
#from MLDL2024_semantic_segmentation.models.metrics import fast_hist
#from MLDL2024_semantic_segmentation.models.metrics import per_class_iou


# Function to clear GPU memory
def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

def train(model, optimizer, train_loader, loss_fn, num_classes, clear_memory_every):
    model.train()
    running_loss = 0.0
    correct = 0
    total_iou = 0
    total_iou_cl = np.zeros([1,num_classes])
    total_images = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):

        first_image = inputs[0]

        inputs, targets = inputs.cuda(), targets.cuda()

        # Compute prediction and loss
        outputs =  model(inputs)
       
        #Ridimensioning tensor
        targets = targets.squeeze(dim=1)
        #print(f'targets size: {targets.size()}')

        targets = targets.long()

        loss = loss_fn(outputs[0], targets)

        # Backpropagation
        optimizer.zero_grad() # reset gradients of parameters
        loss.backward()  # backpropagate the prediction loss
        optimizer.step() # update model

        running_loss += loss.item()
        _, predicted = outputs[0].max(1)
        iou, iou_cl = meanIOU(num_classes, predicted, targets) #sum of meanIOU over classes di tutte le immagini nel batch
        total_iou += iou
        total_iou_cl += iou_cl
        total_images += len(targets)
        
        # Clear GPU memory periodically
        if clear_memory_every!=0 and batch_idx % clear_memory_every == 0:
            clear_gpu_memory()

    miou = total_iou/total_images
    iou_class = total_iou_cl/total_images
    return miou, iou_class

def test(model, test_loader, loss_fn, num_classes, clear_memory_every):
    model.eval()
    test_loss = 0
    correct = 0
    total_images = 0
    total_iou = 0
    total_iou_cl = np.zeros([1,num_classes])
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            #Ridimensioning tensor
            targets = targets.squeeze(dim=1)
            targets = targets.long()
            
            loss = loss_fn(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            
            iou, iou_cl = meanIOU(num_classes, predicted, targets) #sum of meanIOU over classes di tutte le immagini nel batch
            total_iou += iou
            total_iou_cl += iou_cl
            total_images += len(targets)
            
            # Clear GPU memory periodically
            if clear_memory_every!=0 and batch_idx % clear_memory_every == 0:
                clear_gpu_memory()

    miou = total_iou/total_images
    iou_class = total_iou_cl/total_images
    return miou, iou_class
