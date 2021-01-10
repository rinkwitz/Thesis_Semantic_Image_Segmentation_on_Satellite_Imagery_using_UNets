import torch
import torch.nn as nn
import torch.nn.functional as F


class Concat2d(nn.Module):
    def __init__(self):
        super(Concat2d, self).__init__()

    def forward(self, x_down, x_enc):
        if x_down.shape[-1] > x_enc.shape[-1]:
            p = (x_down.shape[-1] - x_enc.shape[-1]) // 2
            if (x_down.shape[-1] - x_enc.shape[-1]) % 2 != 0:
                p += 1
            x_enc = F.pad(x_enc, (p, p, p, p))
        start = [(x_enc.shape[-2] - x_down.shape[-2]) // 2, (x_enc.shape[-1] - x_down.shape[-1]) // 2]
        length = [x_down.shape[-2], x_down.shape[-1]]
        crop = torch.narrow(torch.narrow(x_enc, dim=2, start=start[0], length=length[0]), dim=3, start=start[1], length=length[1])
        cat = torch.cat(tensors=(x_down, crop), dim=1)
        return cat
