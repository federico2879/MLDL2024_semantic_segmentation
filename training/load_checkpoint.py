def load_checkpoint(filename="checkpoint.pth.tar"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    val_IOU = checkpoint['val_IOU']
    return model, optimizer, start_epoch, val_IOU

#model, optimizer, start_epoch, val_IOU = load_checkpoint()
