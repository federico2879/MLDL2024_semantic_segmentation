import torch
import torchvision

def train(model, optimizer, train_loader, loss_fn):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):


        #print(f'batch id: {batch_idx}')
        #print(f'(inputs, targets): {(inputs.size(), targets.size())}')
        first_image = inputs[0]

        # Stampiamo le dimensioni della prima immagine nel batch
        #print("Dimensioni della prima immagine nel batch:", first_image.size())
        inputs, targets = inputs.cuda(), targets.cuda()

        # Compute prediction and loss
        outputs =  model(inputs)
        '''
       # print(f'outputs[0]: {outputs[0]}')
        print(f'outputs[0] type: {outputs[0].type()}')
        print(f'outputs[0] size: {outputs[0].size()}')


        #print(f'targets: {targets}')
        print(f'targets type: {targets.type()}')
        print(f'targets size: {targets.size()}')
        '''
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
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    return train_accuracy

def test(model, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    total_images = 0
    total_iou = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            #Ridimensioning tensor+
            '''
            print(f'outputs: {outputs}')
            print(f'outputs type: {outputs.type()}')
            print(f'outputs size: {outputs.size()}')


            print(f'outputs[0]: {outputs[0]}')

            print(f'outputs[0] type: {outputs[0].type()}')
            print(f'outputs[0] size: {outputs[0].size()}')


            #pri nt(f'targets: {targets}')
            print(f'targets type: {targets.type()}')
            print(f'targets size: {targets.size()}')
            '''
            targets = targets.squeeze(dim=1)

            #print(f'targets size: {targets.size()}')

            targets = targets.long()
            #print(f'targets type: {targets.type()}')
            #print(f'targets size: {targets.size()}')
            loss = loss_fn(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            #print(f'predicted: {predicted}')
            iou = meanIOU(outputs.size()[1], predicted, targets) #sum of meanIOU over classes di tutte le immagini nel batch
            #total += targets.size(0)
            #correct += predicted.eq(targets).sum().item()
            total_iou += iou.sum().item()  #somma di tytte le singole iou calcolate in precedenza

            #print(f'len di targets (=batch_size?): {len(targets)}')
            total_images += len(targets)

    result= total_iou/total_images
    #test_loss = test_loss / len(test_loader)
    #test_accuracy = 100. * correct / total
    return result
