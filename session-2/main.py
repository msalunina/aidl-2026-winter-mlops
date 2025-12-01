import torch

from dataset import MyDataset
from model import MyModel
from utils import accuracy
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision


device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_single_epoch(
        epoch: int,
        train_loader: torch.utils.data.DataLoader,
        network: torch.nn.Module,
        optimizer: torch.optim,
        criterion: torch.nn.functional,
        config: dict,
):

    # Activate the train=True flag inside the model
    network.train()

    avg_loss = []
    acc = 0.
    for batch_idx, (data, target) in enumerate(train_loader):

        # print type of data and target
        print(type(data), type(target))

        # Move input data and labels to the device
        data, target = data.to(device), target.to(device)

        # Set network gradients to 0.
        optimizer.zero_grad()

        # Forward batch of images through the network
        output = network(data)

        # Compute loss
        loss = criterion(output, target)

        # Compute backpropagation
        loss.backward()

        # Update parameters of the network
        optimizer.step()

        # Compute metrics
        acc += accuracy(target, output)
        avg_loss.append(loss.item())

        if batch_idx % config["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    avg_acc = 100. * acc / len(train_loader)

    return np.mean(avg_loss), avg_acc


@torch.no_grad()
def eval_single_epoch(
        test_loader: torch.utils.data.DataLoader,
        network: torch.nn.Module,
        criterion: torch.nn.functional,
):

    # Dectivate the train=True flag inside the model
    network.eval()

    test_loss = []
    acc = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        output = network(data)

        # Apply the loss criterion and accumulate the loss
        test_loss.append(criterion(output, target).item())  # sum up batch loss

        # compute number of correct predictions in the batch
        acc += accuracy(target, output)

    # Average accuracy across all batches
    test_acc = 100. * acc / len(test_loader)
    test_loss = np.mean(test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.1f}%\n'.format(
        test_loss, test_acc,
    ))
    return test_loss, test_acc


def train_model(config):

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    my_dataset = MyDataset(images_path=config["images_path"],
                           labels_path=config["labels_path"],
                           transform=transforms)

    # Split dataset into train, val, and test
    train_size = int(0.7 * len(my_dataset))
    val_size = int(0.15 * len(my_dataset))
    test_size = len(my_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        my_dataset, [train_size, val_size, test_size]
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config["train_batch_size"],
        shuffle=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=config["val_batch_size"],
        shuffle=False,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=config["test_batch_size"],
        shuffle=False,
        drop_last=True,
    )

    my_model = MyModel().to(device)
    criterion = nn.NLLLoss(reduction='mean')
    optimizer = torch.optim.RMSprop(
        my_model.parameters(), lr=config["learning_rate"])

    for epoch in range(config["epochs"]):
        train_single_epoch(epoch, train_loader, my_model,
                           optimizer, criterion, config)
        eval_single_epoch(val_loader, my_model, criterion)

    return my_model


if __name__ == "__main__":

    config = {
        "images_path": "./session-2/data/images",
        "labels_path": "./session-2/data/chinese_mnist.csv",
        "learning_rate": 0.001,
        "epochs": 10,
        "log_interval": 10,
        "train_batch_size": 64,
        "test_batch_size": 64,
        "val_batch_size": 64,
    }
    train_model(config)
