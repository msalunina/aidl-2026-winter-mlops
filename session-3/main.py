import torch
from torch.utils.data import DataLoader

from model import MyModel
from utils import binary_accuracy
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_single_epoch(model, train_loader, optimizer):
    criterion = torch.nn.BCELoss()
    model.train()
    accs, losses = [], []
    for x, y in train_loader:
        # You will need to do y = y.unsqueeze(1).float() to add an output dimension to the labels and cast to the correct type

        # Move input data and labels to the device
        data, target = x.to(device), y.to(device)
        target = target.unsqueeze(1).float()

        # Set network gradients to 0.
        optimizer.zero_grad()

        # Forward batch of images through the network
        output = model(data)

        # Compute loss
        loss = criterion(output, target)

        # Compute backpropagation
        loss.backward()

        # Update parameters of the network
        optimizer.step()
        acc = binary_accuracy(target, output)

        losses.append(loss.item())
        accs.append(acc.item())
    return np.mean(losses), np.mean(accs)


def eval_single_epoch(model, val_loader):
    criterion = torch.nn.BCELoss()
    accs, losses = [], []
    with torch.no_grad():
        model.eval()
        for x, y in val_loader:
            data, target = x.to(device), y.to(device)
            target = target.unsqueeze(1).float()

            output = model(data)

            # Apply the loss criterion and accumulate the loss
            loss = criterion(output, target)

            # compute number of correct predictions in the batch
            acc = binary_accuracy(target, output)

            losses.append(loss.item())
            accs.append(acc.item())
    return np.mean(losses), np.mean(accs)


def train_model(config):

    data_transforms = transforms.Compose([
        transforms.Resize(150), # Resize the short side of the image to 150 keeping aspect ratio
        transforms.CenterCrop(150), # Crop a square in the center of the image
        transforms.ToTensor(), # Convert the image to a tensor with pixels in the range [0, 1]
    ])
    train_dataset = ImageFolder(config["train_dir"], transform=data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_dataset = ImageFolder(config["test_dir"], transform=data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    my_model = MyModel().to(device)

    optimizer = optim.Adam(my_model.parameters(), config["lr"])
    for epoch in range(config["epochs"]):
        loss, acc = train_single_epoch(my_model, train_loader, optimizer)
        print(f"Train Epoch {epoch} loss={loss:.2f} acc={acc:.2f}")
        loss, acc = eval_single_epoch(my_model, test_loader)
        print(f"Eval Epoch {epoch} loss={loss:.2f} acc={acc:.2f}")
    
    return my_model


if __name__ == "__main__":

    config = {
        "lr": 1e-3,
        "batch_size": 64,
        "epochs": 5,
        "train_dir": "./session-3/cars_vs_flowers/training_set",
        "test_dir": "./session-3/cars_vs_flowers/test_set"
    }
    my_model = train_model(config)

    
