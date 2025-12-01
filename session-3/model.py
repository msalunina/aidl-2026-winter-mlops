import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 1
IN_CHANNELS = 3

class ConvBlock(nn.Module):

    def __init__(
            self,
            num_inp_channels: int,
            num_out_fmaps: int,
            kernel_size: int,
            pool_size: int=2) -> None:

        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=num_inp_channels,
            out_channels=num_out_fmaps,
            kernel_size=(kernel_size, kernel_size)
        )
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(pool_size, pool_size))

    def forward(self, x):
        return self.maxpool(self.relu(self.conv(x)))


class MyModel(nn.Module):

    def __init__(self, kernel_size: int=5) -> None:
        super().__init__()

        self.pad = nn.ConstantPad2d(2, 0)
        self.conv1 = ConvBlock(num_inp_channels=IN_CHANNELS, num_out_fmaps=6, kernel_size=kernel_size)
        self.conv2 = ConvBlock(num_inp_channels=6, num_out_fmaps=16, kernel_size=kernel_size)
        self.mlp = nn.Sequential(
            nn.LazyLinear(
                out_features=120
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=120,
                out_features=84
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=84,
                out_features=NUM_CLASSES
            ),
            # nn.LogSoftmax(dim=-1)
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.pad(x)
        x = self.conv1(x)
        x = self.conv2(x)
        bsz, nch, height, width = x.shape
        x = x.reshape(bsz, -1)
        y = self.mlp(x)
        return y
