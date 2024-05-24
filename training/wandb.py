!pip install -q torchinfo torchmetrics wandb
import wandb
import torch
from torch import optim
from functools import partial

def create_name(config, param_list):
    name_str = ""
    for p_name in param_list:
        name_str = name_str + p_name + ": " + str(getattr(config, p_name))
        if p_name != param_list[-1]:
           name_str = name_str + ", "
    return name_str

def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate)
    return optimizer

def train(config=None, network = None, loss = None, train_epoch = None, val_epoch = None,
          train_dataset = None, val_dataset = None, param_list = None):
  
    import wandb
    # Initialize a new wandb run
    with wandb.init(config=config) as run:

        config = wandb.config
        
        if param_list is not None:
            nm = create_name(config, param_list)
            run.name = nm

        optimizer = build_optimizer(network, config.optimizer, config.learning_rate)
        dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle = True)
        dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle = False)

        for epoch in range(config.epochs):
            train_epoch(network, dataloader_train, optimizer, loss, 0)
            mIOU = val_epoch(network, dataloader_val, loss)
            wandb.log({"Mean IOU": mIOU, "epoch": epoch+1}) 

def wandb(network = None, loss = None, sweep_config = None, train_epoch = None, 
          val_epoch = None, train_dataset = None, val_dataset = None, param_list = None):
    
    import wandb

    # Access learning rate from sweep configuration
    config = wandb.login()

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="optimizer_sweep")

    partial_training_function = partial(train, network = network, loss = loss, 
                                        train_epoch = train_epoch, val_epoch = val_epoch,
                                        train_dataset = train_dataset, val_dataset = val_dataset, 
                                        param_list = param_list)

    # Esecute sweep    
    wandb.agent(sweep_id, function=partial_training_function)
              
