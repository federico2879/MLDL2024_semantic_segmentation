# TODO: Define here your training and validation loops.
import torch
import torchvision

def train(model, optimizer, train_loader, loss_fn):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        # Compute prediction and loss
        outputs =  model(inputs)
        loss = loss_fn(outputs, targets)

        # Backpropagation
        optimizer.zero_grad() # reset gradients of parameters
        loss.backward()  # backpropagate the prediction loss
        optimizer.step() # update model

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    return train_accuracy

def test(model, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss = test_loss / len(test_loader)
    test_accuracy = 100. * correct / total
    return test_accuracy