import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSupervisionModule(nn.Module):
    def __init__(self, up_sampling_factors=(2, 2, 2)):
        super(DeepSupervisionModule, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up_sampling_factors = up_sampling_factors

    def forward(self, dec4, dec3, dec2, dec1):
        out = self.up(dec4)
        if self.up_sampling_factors[0] == 4:
            out = self.up(out)

        start = [(out.shape[-2] - dec3.shape[-2]) // 2, (out.shape[-1] - dec3.shape[-1]) // 2]
        length = [dec3.shape[-2], dec3.shape[-1]]
        out = torch.narrow(torch.narrow(out, dim=2, start=start[0], length=length[0]), dim=3, start=start[1], length=length[1])
        out = self.up(torch.cat(tensors=(dec3, out), dim=1))
        if self.up_sampling_factors[1] == 4:
            out = self.up(out)

        start = [(out.shape[-2] - dec2.shape[-2]) // 2, (out.shape[-1] - dec2.shape[-1]) // 2]
        length = [dec2.shape[-2], dec2.shape[-1]]
        out = torch.narrow(torch.narrow(out, dim=2, start=start[0], length=length[0]), dim=3, start=start[1], length=length[1])
        out = self.up(torch.cat(tensors=(dec2, out), dim=1))
        if self.up_sampling_factors[2] == 4:
            out = self.up(out)

        start = [(out.shape[-2] - dec1.shape[-2]) // 2, (out.shape[-1] - dec1.shape[-1]) // 2]
        length = [dec1.shape[-2], dec1.shape[-1]]
        out = torch.narrow(torch.narrow(out, dim=2, start=start[0], length=length[0]), dim=3, start=start[1], length=length[1])
        out = torch.cat(tensors=(dec1, out), dim=1)

        return out


if __name__ == '__main__':
    model = DeepSupervisionModule()
    for param in model.parameters():
        print(param)
