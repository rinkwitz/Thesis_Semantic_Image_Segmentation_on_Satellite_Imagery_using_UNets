import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.utils.Conv2dBlock import Conv2dBlock
from Models.utils.DeepSuperVisionModule import DeepSupervisionModule
from Models.utils.UpConcat2d import UpConcat2d


class DualAttentionModule(nn.Module):
    def __init__(self, in_channels, batchnorm):
        super(DualAttentionModule, self).__init__()
        self.in_channels = in_channels
        self.intermediate_channels = self.in_channels // 8
        self.out_channels = self.in_channels
        self.batchnorm = batchnorm
        self.pos_att_mod = PositionalAttentionModule(in_channels=self.intermediate_channels)
        self.ch_att_mod = ChannelAttentionModule()
        self.conv_in_pos = nn.Conv2d(in_channels=self.in_channels, out_channels=self.intermediate_channels,
                                     kernel_size=3, stride=1, padding=1)
        self.conv_in_ch = nn.Conv2d(in_channels=self.in_channels, out_channels=self.intermediate_channels,
                                    kernel_size=3, stride=1, padding=1)
        self.conv_out_pos = nn.Conv2d(in_channels=self.intermediate_channels, out_channels=self.intermediate_channels,
                                      kernel_size=3, stride=1, padding=1)
        self.conv_out_ch = nn.Conv2d(in_channels=self.intermediate_channels, out_channels=self.intermediate_channels,
                                     kernel_size=3, stride=1, padding=1)
        self.conv_fusion = nn.Conv2d(in_channels=self.intermediate_channels, out_channels=self.out_channels,
                                     kernel_size=1, stride=1, padding=0)
        if self.batchnorm:
            self.batchnorm_in_pos = nn.BatchNorm2d(self.intermediate_channels)
            self.batchnorm_in_ch = nn.BatchNorm2d(self.intermediate_channels)
            self.batchnorm_out_pos = nn.BatchNorm2d(self.intermediate_channels)
            self.batchnorm_out_ch = nn.BatchNorm2d(self.intermediate_channels)
            self.batchnorm_fusion = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        # positional path:
        pos = self.conv_in_pos(x)
        if self.batchnorm:
            self.batchnorm_in_pos(pos)
        pos = F.relu(pos)
        pos = self.pos_att_mod(pos)
        pos = self.conv_out_pos(pos)
        if self.batchnorm:
            pos = self.batchnorm_out_pos(pos)
        pos = F.relu(pos)

        # channel path:
        ch = self.conv_in_ch(x)
        if self.batchnorm:
            ch = self.batchnorm_in_ch(ch)
        ch = F.relu(ch)
        ch = self.ch_att_mod(ch)
        ch = self.conv_out_ch(ch)
        if self.batchnorm:
            ch = self.batchnorm_out_ch(ch)
        ch = F.relu(ch)

        # fusion:
        fusion = pos + ch
        fusion = self.conv_fusion(fusion)
        if self.batchnorm:
            fusion = self.batchnorm_fusion(fusion)
        out = F.relu(fusion)
        return out

    def initialize_weights(self):
        for module in [self.pos_att_mod, self.ch_att_mod]:
            module.initialize_weights()
        for layer in [self.conv_in_pos, self.conv_in_ch,
                      self.conv_out_pos, self.conv_out_ch, self.conv_fusion]:
            nn.init.normal_(layer.weight.data, mean=0.0, std=.02)
            nn.init.constant_(layer.bias.data, 0.0)
        if self.batchnorm:
            for layer in [self.batchnorm_in_pos, self.batchnorm_in_ch,
                          self.batchnorm_out_pos, self.batchnorm_out_ch,
                          self.batchnorm_fusion]:
                nn.init.normal_(layer.weight.data, mean=0.0, std=.02)
                nn.init.constant_(layer.bias.data, 0.0)


class PositionalAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(PositionalAttentionModule, self).__init__()
        self.in_channels = in_channels
        self.conv_B = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                                padding=0)
        self.conv_C = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                                padding=0)
        self.conv_D = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                                padding=0)
        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, A):
        batchsize, num_channels, height, width = A.shape
        N = height * width
        B = self.conv_B(A).view((batchsize, num_channels, N))
        C = self.conv_C(A).view((batchsize, num_channels, N))
        D = self.conv_D(A).view((batchsize, num_channels, N))
        S = F.softmax(torch.bmm(C.permute(0, 2, 1), B), dim=-1)
        DS = torch.bmm(D, S.permute(0, 2, 1)).view((batchsize, num_channels, height, width))
        E = self.alpha * DS + A
        return E

    def initialize_weights(self):
        for layer in [self.conv_B, self.conv_C, self.conv_D]:
            nn.init.normal_(layer.weight.data, mean=0.0, std=.02)
            nn.init.constant_(layer.bias.data, 0.0)
        nn.init.constant_(self.alpha.data, 0.0)


class ChannelAttentionModule(nn.Module):
    def __init__(self):
        super(ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, A):
        batchsize, num_channels, height, width = A.shape
        N = height * width
        A1 = A.view((batchsize, num_channels, N))
        X = F.softmax(torch.bmm(A1, A1.permute(0, 2, 1)), dim=-1)
        XA1 = torch.bmm(X.permute(0, 2, 1), A1).view((batchsize, num_channels, height, width))
        E = self.beta * XA1 + A
        return E

    def initialize_weights(self):
        nn.init.constant_(self.beta.data, 0.0)


class DualAttentionUNet(nn.Module):
    def __init__(self, in_channels, num_categories=2, filter_sizes=(64, 128, 256, 512, 1024), deep_supervision=False):
        super(DualAttentionUNet, self).__init__()
        self.in_channels = in_channels
        self.num_categories = num_categories
        self.filter_sizes = filter_sizes
        self.deep_supervision = deep_supervision
        self.main_model_name = 'dualattention_unet'

        # Encoder:
        self.enc1_Conv2dBlock = Conv2dBlock(self.in_channels, self.filter_sizes[0], True)
        self.enc2_Conv2dBlock = Conv2dBlock(self.filter_sizes[0], self.filter_sizes[1], True)
        self.enc3_Conv2dBlock = Conv2dBlock(self.filter_sizes[1], self.filter_sizes[2], True)
        self.enc4_Conv2dBlock = Conv2dBlock(self.filter_sizes[2], self.filter_sizes[3], True)
        self.enc5_Conv2dBlock = Conv2dBlock(self.filter_sizes[3], self.filter_sizes[4], True)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)

        # Decoder:
        self.dec4_dual_att_mod = DualAttentionModule(self.filter_sizes[3], True)
        self.dec4_UpConcat2d = UpConcat2d(self.filter_sizes[4], self.filter_sizes[3])
        self.dec4_Conv2dBlock = Conv2dBlock(self.filter_sizes[4], self.filter_sizes[3], True)
        self.dec3_dual_att_mod = DualAttentionModule(self.filter_sizes[2], True)
        self.dec3_UpConcat2d = UpConcat2d(self.filter_sizes[3], self.filter_sizes[2])
        self.dec3_Conv2dBlock = Conv2dBlock(self.filter_sizes[3], self.filter_sizes[2], True)
        self.dec2_dual_att_mod = DualAttentionModule(self.filter_sizes[1], True)
        self.dec2_UpConcat2d = UpConcat2d(self.filter_sizes[2], self.filter_sizes[1])
        self.dec2_Conv2dBlock = Conv2dBlock(self.filter_sizes[2], self.filter_sizes[1], True)
        self.dec1_dual_att_mod = DualAttentionModule(self.filter_sizes[0], True)
        self.dec1_UpConcat2d = UpConcat2d(self.filter_sizes[1], self.filter_sizes[0])
        self.dec1_Conv2dBlock = Conv2dBlock(self.filter_sizes[1], self.filter_sizes[0], True)
        if self.deep_supervision:
            aggr_channels = self.filter_sizes[0] + self.filter_sizes[1] + self.filter_sizes[2] + self.filter_sizes[3]
            self.dec1_out = nn.Conv2d(in_channels=aggr_channels, out_channels=self.num_categories, kernel_size=1,
                                      stride=1, padding=0)
            self.deep_sup_module = DeepSupervisionModule()
        else:
            self.dec1_out = nn.Conv2d(in_channels=self.filter_sizes[0], out_channels=self.num_categories, kernel_size=1,
                                      stride=1, padding=0)

    def forward(self, x):
        # Encoder:
        enc1 = self.enc1_Conv2dBlock(x)
        enc2 = self.pool(enc1)
        enc2 = self.enc2_Conv2dBlock(enc2)
        enc3 = self.pool(enc2)
        enc3 = self.enc3_Conv2dBlock(enc3)
        enc4 = self.pool(enc3)
        enc4 = self.enc4_Conv2dBlock(enc4)
        enc5 = self.pool(enc4)
        enc5 = self.enc5_Conv2dBlock(enc5)

        # Decoder:
        dec4_att = self.dec4_dual_att_mod(enc4)
        dec4 = self.dec4_UpConcat2d(enc5, dec4_att)
        dec4 = self.dec4_Conv2dBlock(dec4)
        dec3_att = self.dec3_dual_att_mod(enc3)
        dec3 = self.dec3_UpConcat2d(dec4, dec3_att)
        dec3 = self.dec3_Conv2dBlock(dec3)
        dec2_att = self.dec2_dual_att_mod(enc2)
        dec2 = self.dec2_UpConcat2d(dec3, dec2_att)
        dec2 = self.dec2_Conv2dBlock(dec2)
        dec1_att = self.dec1_dual_att_mod(enc1)
        dec1 = self.dec1_UpConcat2d(dec2, dec1_att)
        dec1 = self.dec1_Conv2dBlock(dec1)
        if self.deep_supervision:
            dec1 = self.deep_sup_module(dec4, dec3, dec2, dec1)
        out = F.softmax(self.dec1_out(dec1), dim=1)

        return out

    def initialize_weights(self):
        for layer in [self.enc1_Conv2dBlock, self.enc2_Conv2dBlock, self.enc3_Conv2dBlock, self.enc4_Conv2dBlock,
                      self.enc5_Conv2dBlock, self.dec4_dual_att_mod, self.dec4_UpConcat2d, self.dec4_Conv2dBlock,
                      self.dec3_dual_att_mod, self.dec3_UpConcat2d, self.dec3_Conv2dBlock, self.dec2_dual_att_mod,
                      self.dec2_UpConcat2d, self.dec2_Conv2dBlock, self.dec1_dual_att_mod, self.dec1_UpConcat2d,
                      self.dec1_Conv2dBlock]:
            layer.initialize_weights()
        nn.init.normal_(self.dec1_out.weight.data, mean=0.0, std=.02)
        nn.init.constant_(self.dec1_out.bias.data, 0.0)

    def freeze_encoder(self):
        for enc in [self.enc1_Conv2dBlock, self.enc2_Conv2dBlock, self.enc3_Conv2dBlock, self.enc4_Conv2dBlock, self.enc5_Conv2dBlock]:
            enc.requires_grad_(False)

    def unfreeze_encoder(self):
        for enc in [self.enc1_Conv2dBlock, self.enc2_Conv2dBlock, self.enc3_Conv2dBlock, self.enc4_Conv2dBlock, self.enc5_Conv2dBlock]:
            enc.requires_grad_(True)


if __name__ == '__main__':
    device = 'cuda:0'
    model = DualAttentionUNet(in_channels=3, num_categories=2, filter_sizes=(32, 64, 128, 256, 512),
                              deep_supervision=True)
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
            num = 1
            for size in param.data.shape:
                num *= size
            num_trainable_params += num
    print(f'\ntrainable params: {num_trainable_params}\n')
