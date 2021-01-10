import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class RandInputImage(nn.Module):
    def __init__(self, shape=(406, 438, 3)):
        super(RandInputImage, self).__init__()
        self.shape = shape
        self.im = nn.Parameter(torch.rand(self.shape, dtype=torch.float32), requires_grad=True)
        self.mean = torch.from_numpy(np.load(Path('tmp/mean_SN1_Buildings.npy'))).type(torch.float32)
        self.std = torch.from_numpy(np.load(Path('tmp/std_SN1_Buildings.npy'))).type(torch.float32)

    def forward(self):
        return torch.div(torch.sub(self.im, self.mean), self.std).permute(2, 0, 1).view(
            (1, self.shape[2], self.shape[0], self.shape[1]))
