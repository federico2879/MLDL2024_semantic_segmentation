!pip install -q torchinfo torchmetrics wandb
import wandb
import torch
from torch import optim

def create_name(config, param_list):
    for p_name in param_list:
        name_str = p_name + " " + str(getattr(config, p_name))
    return name_str

def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate)
    return optimizer

def train(config=None, train_epoch, dataset, network, param_list):
    # Initialize a new wandb run
    with wandb.init(config=config):

        config = wandb.config
        
        if param_list is not None:
            nm = create_name(config, param_list)
            run.name = nm

        optimizer = build_optimizer(network, config.optimizer, config.learning_rate)
        loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size)

        for epoch in range(config.epochs):
            acc = train_epoch(network, loader, optimizer, loss)
            wandb.log({"loss": acc, "epoch": epoch+1}) 

def wandb(loss, sweep_config, train_epoch, dataset, param_list)
    
    # Access learning rate from sweep configuration
    config = wandb.login()

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="optimizer_sweep")
    
    wandb.agent(sweep_id, function=train)
