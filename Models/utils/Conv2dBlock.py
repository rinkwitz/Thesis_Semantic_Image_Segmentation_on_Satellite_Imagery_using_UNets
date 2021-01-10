import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm=True):
        super(Conv2dBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batchnorm = batchnorm
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1,
                               padding=0)
        self.conv2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1,
                               padding=0)
        if self.batchnorm:
            self.batchnorm1 = nn.BatchNorm2d(self.out_channels)
            self.batchnorm2 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        if self.batchnorm:
            out = F.relu(self.batchnorm1(self.conv1(x)))
            out = F.relu(self.batchnorm2(self.conv2(out)))
        else:
            out = F.relu(self.conv1(x))
            out = F.relu(self.conv2(out))
        # torch.save(out, f'tmp/conv2d_activation_{out.shape[-3]}-{out.shape[-2]}-{out.shape[-1]}.pt')
        return out

    def initialize_weights(self):
        for layer in [self.conv1, self.conv2]:
            nn.init.normal_(layer.weight.data, mean=0.0, std=.02)
            nn.init.constant_(layer.bias.data, 0.0)
        if self.batchnorm:
            for layer in [self.batchnorm1, self.batchnorm2]:
                nn.init.normal_(layer.weight.data, mean=0.0, std=.02)
                nn.init.constant_(layer.bias.data, 0.0)
