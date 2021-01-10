import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

from Models.utils.Concat2D import Concat2d
from Models.utils.Conv2dBlock import Conv2dBlock
from Models.utils.DeepSuperVisionModule import DeepSupervisionModule
from Models.utils.UpConcat2d import UpConcat2d


class ResNet34_UNet(torch.nn.Module):
    def __init__(self, in_channels, num_categories=2, deep_supervision=False, pretrained=True):
        super(ResNet34_UNet, self).__init__()
        self.in_channels = in_channels
        self.num_categories = num_categories
        self.deep_supervision = deep_supervision
        self.pretrained = pretrained
        self.filter_sizes = None
        self.main_model_name = 'resnet34_unet'

        # Encoder:
        self.enc1 = nn.ModuleList([resnet34(pretrained=self.pretrained).conv1, resnet34(pretrained=self.pretrained).bn1,
                                   resnet34(pretrained=self.pretrained).relu,
                                   resnet34(pretrained=self.pretrained).maxpool])
        self.enc2 = nn.ModuleList([resnet34(pretrained=self.pretrained).layer1])
        self.enc3 = nn.ModuleList([resnet34(pretrained=self.pretrained).layer2])
        self.enc4 = nn.ModuleList([resnet34(pretrained=self.pretrained).layer3])
        self.enc5 = nn.ModuleList([resnet34(pretrained=self.pretrained).layer4])

        # Decoder:
        self.dec4_UpConcat2d = UpConcat2d(512, 256)
        self.dec4_Conv2dBlock = Conv2dBlock(512, 256, True)
        self.dec3_UpConcat2d = UpConcat2d(256, 128)
        self.dec3_Conv2dBlock = Conv2dBlock(256, 128, True)
        self.dec2_UpConcat2d = UpConcat2d(128, 64)
        self.dec2_Conv2dBlock = Conv2dBlock(128, 64, True)
        self.dec1_Concat2d = Concat2d()
        self.dec1_Conv2dBlock = Conv2dBlock(128, 64, True)
        if self.deep_supervision:
            aggr_channels = 512
            self.dec1_out = nn.Conv2d(in_channels=aggr_channels, out_channels=self.num_categories, kernel_size=1,
                                      stride=1, padding=0)
            self.deep_sup_module = DeepSupervisionModule()
        else:
            self.dec1_out = nn.Conv2d(in_channels=64, out_channels=self.num_categories, kernel_size=1,
                                      stride=1, padding=0)

    def forward(self, x):
        # Encoder:
        for idx, mod in enumerate(self.enc1):
            if idx == 0:
                enc1 = mod(x)
            else:
                enc1 = mod(enc1)

        for idx, mod in enumerate(self.enc2):
            if idx == 0:
                enc2 = mod(enc1)
            else:
                enc2 = mod(enc2)

        for idx, mod in enumerate(self.enc3):
            if idx == 0:
                enc3 = mod(enc2)
            else:
                enc3 = mod(enc3)

        for idx, mod in enumerate(self.enc4):
            if idx == 0:
                enc4 = mod(enc3)
            else:
                enc4 = mod(enc4)

        for idx, mod in enumerate(self.enc5):
            if idx == 0:
                enc5 = mod(enc4)
            else:
                enc5 = mod(enc5)

        # for enc in [enc1, enc2, enc3, enc4, enc5]:
        #     print(enc.shape)
        # print('')
        # return enc5

        # Decoder:
        dec4 = self.dec4_UpConcat2d(enc5, enc4)
        dec4 = self.dec4_Conv2dBlock(dec4)
        dec3 = self.dec3_UpConcat2d(dec4, enc3)
        dec3 = self.dec3_Conv2dBlock(dec3)
        dec2 = self.dec2_UpConcat2d(dec3, enc2)
        dec2 = self.dec2_Conv2dBlock(dec2)
        dec1 = self.dec1_Concat2d(dec2, enc1)
        dec1 = self.dec1_Conv2dBlock(dec1)
        if self.deep_supervision:
            dec1 = self.deep_sup_module(dec4, dec3, dec2, dec1)
        out = F.softmax(self.dec1_out(dec1), dim=1)

        return out

    def initialize_decoder_weights(self):
        for layer in [self.dec4_UpConcat2d, self.dec4_Conv2dBlock, self.dec3_UpConcat2d, self.dec3_Conv2dBlock,
                      self.dec2_UpConcat2d, self.dec2_Conv2dBlock, self.dec1_Conv2dBlock]:
            layer.initialize_weights()
        nn.init.normal_(self.dec1_out.weight.data, mean=0.0, std=.02)
        nn.init.constant_(self.dec1_out.bias.data, 0.0)

    def freeze_encoder(self):
        for enc in [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]:
            enc.requires_grad_(False)

    def unfreeze_encoder(self):
        for enc in [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]:
            enc.requires_grad_(True)


if __name__ == '__main__':
    device = 'cuda:0'
    model = ResNet34_UNet(in_channels=3, num_categories=2, deep_supervision=True)
    model.to(device)
    model.initialize_decoder_weights()

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
