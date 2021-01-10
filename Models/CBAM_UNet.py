import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.utils.Conv2dBlock import Conv2dBlock
from Models.utils.DeepSuperVisionModule import DeepSupervisionModule
from Models.utils.UpConcat2d import UpConcat2d


class CBAM(nn.Module):
    def __init__(self, channels_in):
        super(CBAM, self).__init__()
        self.channels_in = channels_in
        self.cam = CAM(self.channels_in)
        self.sam = SAM(self.channels_in)

    def forward(self, x, save_attention=False):
        ch_att = self.cam(x, save_attention)
        x = x * ch_att
        spat_att = self.sam(x, save_attention)
        out = x * spat_att
        return out

    def initialize_weights(self):
        for layer in [self.cam, self.sam]:
            layer.initialize_weights()


class CAM(nn.Module):
    def __init__(self, channels_in):
        super(CAM, self).__init__()
        self.channels_in = channels_in
        self.hidden = self.channels_in // 8
        self.mlp1 = nn.Linear(in_features=self.channels_in, out_features=self.hidden)
        self.mlp2 = nn.Linear(in_features=self.hidden, out_features=self.channels_in)

    def forward(self, x, save_attention=False):
        bs, ch, h, w = x.shape
        feat_avg = F.avg_pool2d(x, kernel_size=(h, w)).view((bs, ch))
        feat_avg = self.mlp2(F.relu(self.mlp1(feat_avg))).view((bs, ch, 1, 1))
        feat_max = F.max_pool2d(x, kernel_size=(h, w)).view((bs, ch))
        feat_max = self.mlp2(F.relu(self.mlp1(feat_max))).view((bs, ch, 1, 1))
        out = torch.sigmoid(feat_avg + feat_max)
        if save_attention:
            torch.save(out, f'tmp/cbam-attention_channel_{out.shape[1]}.pt')
        return out

    def initialize_weights(self):
        for layer in [self.mlp1, self.mlp2]:
            nn.init.normal_(layer.weight.data, mean=0.0, std=.02)
            nn.init.constant_(layer.bias.data, 0.0)


class SAM(nn.Module):
    def __init__(self, channels_in):
        super(SAM, self).__init__()
        self.channels_in = channels_in
        self.avg_pool = nn.AvgPool3d(kernel_size=(self.channels_in, 1, 1))
        self.max_pool = nn.MaxPool3d(kernel_size=(self.channels_in, 1, 1))
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)

    def forward(self, x, save_attention=False):
        spat_att = torch.cat(tensors=(self.avg_pool(x), self.max_pool(x)), dim=1)
        spat_att = torch.sigmoid(self.conv1(spat_att))
        if save_attention:
            torch.save(spat_att, f'tmp/cbam-attention_spatial_{spat_att.shape[-2]}-{spat_att.shape[-1]}.pt')
        return spat_att

    def initialize_weights(self):
        nn.init.normal_(self.conv1.weight.data, mean=0.0, std=.02)
        nn.init.constant_(self.conv1.bias.data, 0.0)


class CBAM_UNet(nn.Module):
    def __init__(self, in_channels, num_categories=2, filter_sizes=(64, 128, 256, 512, 1024), deep_supervision=False):
        super(CBAM_UNet, self).__init__()
        self.in_channels = in_channels
        self.num_categories = num_categories
        self.filter_sizes = filter_sizes
        self.deep_supervision = deep_supervision
        self.main_model_name = 'cbam_unet'

        # Encoder:
        self.enc1_Conv2dBlock = Conv2dBlock(self.in_channels, self.filter_sizes[0], True)
        self.enc2_Conv2dBlock = Conv2dBlock(self.filter_sizes[0], self.filter_sizes[1], True)
        self.enc3_Conv2dBlock = Conv2dBlock(self.filter_sizes[1], self.filter_sizes[2], True)
        self.enc4_Conv2dBlock = Conv2dBlock(self.filter_sizes[2], self.filter_sizes[3], True)
        self.enc5_Conv2dBlock = Conv2dBlock(self.filter_sizes[3], self.filter_sizes[4], True)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)

        # Decoder:
        self.dec4_cbam = CBAM(self.filter_sizes[3])
        self.dec4_UpConcat2d = UpConcat2d(self.filter_sizes[4], self.filter_sizes[3])
        self.dec4_Conv2dBlock = Conv2dBlock(self.filter_sizes[4], self.filter_sizes[3], True)
        self.dec3_cbam = CBAM(self.filter_sizes[2])
        self.dec3_UpConcat2d = UpConcat2d(self.filter_sizes[3], self.filter_sizes[2])
        self.dec3_Conv2dBlock = Conv2dBlock(self.filter_sizes[3], self.filter_sizes[2], True)
        self.dec2_cbam = CBAM(self.filter_sizes[1])
        self.dec2_UpConcat2d = UpConcat2d(self.filter_sizes[2], self.filter_sizes[1])
        self.dec2_Conv2dBlock = Conv2dBlock(self.filter_sizes[2], self.filter_sizes[1], True)
        self.dec1_cbam = CBAM(self.filter_sizes[0])
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

    def forward(self, x, save_attention=False):
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
        dec4_att = self.dec4_cbam(enc4, save_attention)
        dec4 = self.dec4_UpConcat2d(enc5, dec4_att)
        dec4 = self.dec4_Conv2dBlock(dec4)
        dec3_att = self.dec3_cbam(enc3, save_attention)
        dec3 = self.dec3_UpConcat2d(dec4, dec3_att)
        dec3 = self.dec3_Conv2dBlock(dec3)
        dec2_att = self.dec2_cbam(enc2, save_attention)
        dec2 = self.dec2_UpConcat2d(dec3, dec2_att)
        dec2 = self.dec2_Conv2dBlock(dec2)
        dec1_att = self.dec1_cbam(enc1, save_attention)
        dec1 = self.dec1_UpConcat2d(dec2, dec1_att)
        dec1 = self.dec1_Conv2dBlock(dec1)
        if self.deep_supervision:
            dec1 = self.deep_sup_module(dec4, dec3, dec2, dec1)
        out = F.softmax(self.dec1_out(dec1), dim=1)

        return out

    def initialize_weights(self):
        for layer in [self.enc1_Conv2dBlock, self.enc2_Conv2dBlock, self.enc3_Conv2dBlock, self.enc4_Conv2dBlock,
                      self.enc5_Conv2dBlock, self.dec4_cbam, self.dec4_UpConcat2d, self.dec4_Conv2dBlock,
                      self.dec3_cbam, self.dec3_UpConcat2d, self.dec3_Conv2dBlock, self.dec2_cbam, self.dec2_UpConcat2d,
                      self.dec2_Conv2dBlock, self.dec1_cbam, self.dec1_UpConcat2d, self.dec1_Conv2dBlock]:
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
    model = CBAM_UNet(in_channels=3, num_categories=2, filter_sizes=(32, 64, 128, 256, 512), deep_supervision=True)
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
