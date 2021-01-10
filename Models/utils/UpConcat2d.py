import torch
import torch.nn as nn
import torch.nn.functional as F


class UpConcat2d(nn.Module):
    def __init__(self, in_channels_conv, out_channels_conv, scale_factor=2):
        super(UpConcat2d, self).__init__()
        self.in_channels_conv = in_channels_conv
        self.out_channels_conv = out_channels_conv
        self.scale_factor = scale_factor
        self.up = nn.ConvTranspose2d(in_channels=self.in_channels_conv, out_channels=self.out_channels_conv,
                                     kernel_size=2, stride=2, padding=0)
        if scale_factor == 4:
            self.up2 = nn.ConvTranspose2d(in_channels=self.out_channels_conv, out_channels=self.out_channels_conv,
                                          kernel_size=2, stride=2, padding=0)

    def forward(self, x_down, x_enc):
        up = F.relu(self.up(x_down))
        if self.scale_factor == 4:
            up = F.relu(self.up2(up))
        if up.shape[-1] > x_enc.shape[-1]:
            p = (up.shape[-1] - x_enc.shape[-1]) // 2
            if (up.shape[-1] - x_enc.shape[-1]) % 2 != 0:
                p += 1
            x_enc = F.pad(x_enc, (p, p, p, p))
        start = [(x_enc.shape[-2] - up.shape[-2]) // 2, (x_enc.shape[-1] - up.shape[-1]) // 2]
        length = [up.shape[-2], up.shape[-1]]
        crop = torch.narrow(torch.narrow(x_enc, dim=2, start=start[0], length=length[0]), dim=3, start=start[1], length=length[1])
        cat = torch.cat(tensors=(up, crop), dim=1)
        return cat

    def initialize_weights(self):
        nn.init.normal_(self.up.weight.data, mean=0.0, std=.02)
        nn.init.constant_(self.up.bias.data, 0.0)
        if self.scale_factor == 4:
            nn.init.normal_(self.up2.weight.data, mean=0.0, std=.02)
            nn.init.constant_(self.up2.bias.data, 0.0)


if __name__ == '__main__':
    x_down = torch.randn((1, 128, 56, 56))
    x_enc = torch.randn((1, 64, 111, 111))
    upconcat = UpConcat2d(in_channels_conv=128, out_channels_conv=64)
    y = upconcat(x_down, x_enc)
    print(y.shape)
