import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.utils.AttentionGate import AttentionGate
from Models.utils.Conv2dBlock import Conv2dBlock
from Models.utils.UpConcat2d import UpConcat2d
from Models.utils.DeepSuperVisionModule import DeepSupervisionModule


class ResNeXt_Block(nn.Module):

    def __init__(self, in_channels, out_channels, batchnorm=True):
        super(ResNeXt_Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batchnorm = batchnorm

        self.conv_in = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1,
                               padding=0)
        self.conv2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1,
                               padding=0)
        self.conv3 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1,
                               padding=0)
        self.conv4 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1,
                               padding=0)
        self.conv5 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1,
                               padding=0)
        self.conv6 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1,
                               padding=0)
        if self.batchnorm:
            self.batchnorm1 = nn.BatchNorm2d(self.out_channels)
            self.batchnorm3 = nn.BatchNorm2d(self.out_channels)
            self.batchnorm5 = nn.BatchNorm2d(self.out_channels)
            self.batchnorm_res = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):

        x = self.conv_in(x)

        res1 = self.conv1(x)
        if self.batchnorm:
            res1 = self.batchnorm1(res1)
        res1 = self.conv2(F.relu(res1))

        res2 = self.conv3(x)
        if self.batchnorm:
            res2 = self.batchnorm3(res2)
        res2 = self.conv4(F.relu(res2))

        res3 = self.conv5(x)
        if self.batchnorm:
            res3 = self.batchnorm5(res3)
        res3 = self.conv6(F.relu(res3))

        res = res1 + res2 + res3
        if self.batchnorm:
            res = self.batchnorm_res(res)
        res = F.relu(res)

        start = [(x.shape[-2] - res.shape[-2]) // 2, (x.shape[-1] - res.shape[-1]) // 2]
        length = [res.shape[-2], res.shape[-1]]
        x = torch.narrow(torch.narrow(x, dim=2, start=start[0], length=length[0]), dim=3, start=start[1], length=length[1])
        out = x + res

        return out

    def initialize_weights(self):
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6]:
            nn.init.normal_(layer.weight.data, mean=0.0, std=.02)
            nn.init.constant_(layer.bias.data, 0.0)
        if self.batchnorm:
            for layer in [self.batchnorm1, self.batchnorm3, self.batchnorm5,
                          self.batchnorm_res]:
                nn.init.normal_(layer.weight.data, mean=0.0, std=.02)
                nn.init.constant_(layer.bias.data, 0.0)


class ResidualAttentionUNet(nn.Module):
    def __init__(self, in_channels, num_categories=2, filter_sizes=(64, 128, 256, 512, 1024), deep_supervision=True):
        super(ResidualAttentionUNet, self).__init__()
        self.in_channels = in_channels
        self.num_categories = num_categories
        self.filter_sizes = filter_sizes
        self.deep_supervision = deep_supervision
        self.main_model_name = 'residualattention_unet'

        # Encoder:
        self.enc1_ResNeXt_Block = ResNeXt_Block(self.in_channels, self.filter_sizes[0])
        self.enc2_ResNeXt_Block = ResNeXt_Block(self.filter_sizes[0], self.filter_sizes[1])
        self.enc3_ResNeXt_Block = ResNeXt_Block(self.filter_sizes[1], self.filter_sizes[2])
        self.enc4_ResNeXt_Block = ResNeXt_Block(self.filter_sizes[2], self.filter_sizes[3])
        self.enc5_ResNeXt_Block = ResNeXt_Block(self.filter_sizes[3], self.filter_sizes[4])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Decoder:
        self.dec4_att_gate = AttentionGate(self.filter_sizes[3], self.filter_sizes[4])
        self.dec4_UpConcat2d = UpConcat2d(self.filter_sizes[4], self.filter_sizes[3])
        self.dec4_ResNeXt_Block = ResNeXt_Block(self.filter_sizes[4], self.filter_sizes[3])
        self.dec3_att_gate = AttentionGate(self.filter_sizes[2], self.filter_sizes[3])
        self.dec3_UpConcat2d = UpConcat2d(self.filter_sizes[3], self.filter_sizes[2])
        self.dec3_ResNeXt_Block = ResNeXt_Block(self.filter_sizes[3], self.filter_sizes[2])
        self.dec2_att_gate = AttentionGate(self.filter_sizes[1], self.filter_sizes[2])
        self.dec2_UpConcat2d = UpConcat2d(self.filter_sizes[2], self.filter_sizes[1])
        self.dec2_ResNeXt_Block = ResNeXt_Block(self.filter_sizes[2], self.filter_sizes[1])
        self.dec1_att_gate = AttentionGate(self.filter_sizes[0], self.filter_sizes[1])
        self.dec1_UpConcat2d = UpConcat2d(self.filter_sizes[1], self.filter_sizes[0])
        self.dec1_ResNeXt_Block = ResNeXt_Block(self.filter_sizes[1], self.filter_sizes[0])
        if self.deep_supervision:
            aggr_channels = self.filter_sizes[0] + self.filter_sizes[1] + self.filter_sizes[2] + self.filter_sizes[3]
            self.dec1_out = nn.Conv2d(in_channels=aggr_channels, out_channels=self.num_categories, kernel_size=1,
                                      stride=1, padding=0)
            self.deep_sup_module = DeepSupervisionModule()
        else:
            self.dec1_out = nn.Conv2d(in_channels=self.filter_sizes[0], out_channels=self.num_categories, kernel_size=1,
                                      stride=1, padding=0)

    def forward(self, x, save_attention=False):
        # Encoder:
        enc1 = self.enc1_ResNeXt_Block(x)
        enc2 = self.pool(enc1)
        enc2 = self.enc2_ResNeXt_Block(enc2)
        enc3 = self.pool(enc2)
        enc3 = self.enc3_ResNeXt_Block(enc3)
        enc4 = self.pool(enc3)
        enc4 = self.enc4_ResNeXt_Block(enc4)
        enc5 = self.pool(enc4)
        enc5 = self.enc5_ResNeXt_Block(enc5)

        # Decoder:
        dec4_gate = self.dec4_att_gate(enc4, enc5, save_attention)
        dec4 = self.dec4_UpConcat2d(enc5, dec4_gate)
        dec4 = self.dec4_ResNeXt_Block(dec4)
        dec3_gate = self.dec3_att_gate(enc3, dec4, save_attention)
        dec3 = self.dec3_UpConcat2d(dec4, dec3_gate)
        dec3 = self.dec3_ResNeXt_Block(dec3)
        dec2_gate = self.dec2_att_gate(enc2, dec3, save_attention)
        dec2 = self.dec2_UpConcat2d(dec3, dec2_gate)
        dec2 = self.dec2_ResNeXt_Block(dec2)
        dec1_gate = self.dec1_att_gate(enc1, dec2, save_attention)
        dec1 = self.dec1_UpConcat2d(dec2, dec1_gate)
        dec1 = self.dec1_ResNeXt_Block(dec1)
        if self.deep_supervision:
            dec1 = self.deep_sup_module(dec4, dec3, dec2, dec1)
        out = F.softmax(self.dec1_out(dec1), dim=1)

        return out

    def initialize_weights(self):
        for layer in [self.enc1_ResNeXt_Block, self.enc2_ResNeXt_Block, self.enc3_ResNeXt_Block,
                      self.enc4_ResNeXt_Block, self.enc5_ResNeXt_Block, self.dec4_att_gate, self.dec4_UpConcat2d,
                      self.dec4_ResNeXt_Block, self.dec3_att_gate, self.dec3_UpConcat2d, self.dec3_ResNeXt_Block,
                      self.dec2_att_gate, self.dec2_UpConcat2d, self.dec2_ResNeXt_Block, self.dec1_att_gate,
                      self.dec1_UpConcat2d, self.dec1_ResNeXt_Block]:
            layer.initialize_weights()
        nn.init.normal_(self.dec1_out.weight.data, mean=0.0, std=.02)
        nn.init.constant_(self.dec1_out.bias.data, 0.0)

    def freeze_encoder(self):
        for enc in [self.enc1_ResNeXt_Block, self.enc2_ResNeXt_Block, self.enc3_ResNeXt_Block, self.enc4_ResNeXt_Block, self.enc5_ResNeXt_Block]:
            enc.requires_grad_(False)

    def unfreeze_encoder(self):
        for enc in [self.enc1_ResNeXt_Block, self.enc2_ResNeXt_Block, self.enc3_ResNeXt_Block, self.enc4_ResNeXt_Block, self.enc5_ResNeXt_Block]:
            enc.requires_grad_(True)


if __name__ == '__main__':
    device = 'cuda:0'
    model = ResidualAttentionUNet(in_channels=3, num_categories=2, filter_sizes=(32, 64, 128, 256, 512),
                                  deep_supervision=False)
    model.to(device)
    model.initialize_weights()

    for param in model.parameters():
        print(param.shape, param.dtype, param.device)
    print('')

    for shape in [(1, 3, 407, 439), (1, 3, 406, 438), (1, 3, 406, 439)]:
        x = torch.randn(shape).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
        optimizer.zero_grad()
        y = model(x)
        loss = (torch.sum((y - torch.randn(y.shape, device=device)) ** 2))
        loss.backward()
        optimizer.step()
        print(y.size())
        print()
        print(y[0, 0, 0:5, 0:5])
        print()
        print(y[0, 1, 0:5, 0:5])

    num_trainable_params = 0
    for param in model.parameters(recurse=True):
        if param.requires_grad:
            x = 1
            for s in param.data.shape:
                x *= s
            num_trainable_params += x
    print(f'\ntrainable params: {num_trainable_params}')
