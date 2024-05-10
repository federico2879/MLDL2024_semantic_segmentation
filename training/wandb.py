!pip install -q torchinfo torchmetrics wandb
import wandb


def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate)
    return optimizer
  
def build_dataset(batch_size):
   
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    # download MNIST training dataset
    dataset = datasets.MNIST(".", train=True, download=True,
                             transform=transform)
    sub_dataset = torch.utils.data.Subset(
        dataset, indices=range(0, len(dataset), 5))
    loader = torch.utils.data.DataLoader(sub_dataset, batch_size=batch_size)

    return loader

def train(config=None, train_epoch, dataset, network):
    # Initialize a new wandb run
    with wandb.init(config=config):

        config = wandb.config

        optimizer = build_optimizer(network, config.optimizer, config.learning_rate)
        loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size)

        for epoch in range(config.epochs):
            acc = train_epoch(network, loader, optimizer, loss)
            wandb.log({"loss": acc, "epoch": epoch+1}) 

def wandb(loss, sweep_config, train_epoch, dataset)
    
    # Access learning rate from sweep configuration
    config = wandb.login()

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="optimizer_sweep")
    
    wandb.agent(sweep_id, function=train)

