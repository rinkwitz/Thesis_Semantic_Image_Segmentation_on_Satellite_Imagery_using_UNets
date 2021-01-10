import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    def __init__(self, in_channels_in, in_channels_gate):
        super(AttentionGate, self).__init__()
        self.in_channels_in = in_channels_in
        self.in_channels_gate = in_channels_gate
        self.intermediate_channels = self.in_channels_in // 8
        self.up_gate = nn.UpsamplingBilinear2d(scale_factor=2)
        self.linear_in = nn.Conv2d(in_channels=self.in_channels_in, out_channels=self.intermediate_channels,
                                   kernel_size=1, stride=1, padding=0, bias=False)
        self.linear_gate = nn.Conv2d(in_channels=self.in_channels_gate, out_channels=self.intermediate_channels,
                                     kernel_size=1, stride=1, padding=0)
        self.linear_att = nn.Conv2d(in_channels=self.intermediate_channels, out_channels=1, kernel_size=1, stride=1,
                                    padding=0)

    def forward(self, x_in, x_gate, save_attention=False):
        start = [(x_in.shape[-2] - 2 * x_gate.shape[-2]) // 2, (x_in.shape[-1] - 2 * x_gate.shape[-1]) // 2]
        length = [2 * x_gate.shape[-2], 2 * x_gate.shape[-1]]
        crop = torch.narrow(torch.narrow(x_in, dim=2, start=start[0], length=length[0]), dim=3, start=start[1], length=length[1])
        alpha = torch.sigmoid(self.linear_att(F.relu(self.linear_in(crop) + self.linear_gate(self.up_gate(x_gate)))))
        if save_attention:
            torch.save(alpha, f'tmp/attention-gate_alpha_{alpha.shape[-2]}-{alpha.shape[-1]}.pt')
        return crop * alpha

    def initialize_weights(self):
        nn.init.normal_(self.linear_in.weight.data, mean=0.0, std=.02)
        for layer in [self.linear_gate, self.linear_att]:
            nn.init.normal_(layer.weight.data, mean=0.0, std=.02)
            nn.init.constant_(layer.bias.data, 0.0)
