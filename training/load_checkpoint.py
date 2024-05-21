def load_checkpoint(filename="checkpoint.pth.tar"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    return model, optimizer, start_epoch, val_loss

model, optimizer, start_epoch, val_loss = load_checkpoint()
