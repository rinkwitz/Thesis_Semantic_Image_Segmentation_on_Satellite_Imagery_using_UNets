import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.utils.Conv2dBlock import Conv2dBlock
from Models.utils.DeepSuperVisionModule import DeepSupervisionModule
from Models.utils.UpConcat2d import UpConcat2d


class scAG(nn.Module):
    def __init__(self, num_channels_enc_in, num_channels_dec_in):
        super(scAG, self).__init__()
        self.num_channels_enc_in = num_channels_enc_in
        self.num_channels_dec_in = num_channels_dec_in
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.spat_AttGate = sAG(self.num_channels_enc_in, self.num_channels_dec_in)
        self.ch_AttGate = cAG(self.num_channels_enc_in, self.num_channels_dec_in)

    def forward(self, x_enc, x_dec, save_attention):
        dec_feat = F.relu(self.up(x_dec))
        start = [(x_enc.shape[-2] - 2 * x_dec.shape[-2]) // 2, (x_enc.shape[-1] - 2 * x_dec.shape[-1]) // 2]
        length = [2 * x_dec.shape[-2], 2 * x_dec.shape[-1]]
        enc_feat = torch.narrow(torch.narrow(x_enc, dim=2, start=start[0], length=length[0]), dim=3, start=start[1], length=length[1])
        spat_att = self.spat_AttGate(enc_feat, dec_feat)
        ch_att = self.ch_AttGate(enc_feat, dec_feat)
        if save_attention:
            torch.save(spat_att, f'tmp/scag-attention_spatial_{spat_att.shape[-2]}-{spat_att.shape[-1]}.pt')
            torch.save(ch_att, f'tmp/scag-attention_channel_{ch_att.shape[1]}.pt')
        out = enc_feat * spat_att * ch_att
        return out

    def initialize_weights(self):
        for layer in [self.spat_AttGate, self.ch_AttGate]:
            layer.initialize_weights()


class sAG(nn.Module):
    def __init__(self, num_channels_in_enc, num_channels_in_dec):
        super(sAG, self).__init__()
        self.num_channels_in_enc = num_channels_in_enc
        self.num_channels_in_dec = num_channels_in_dec
        self.ch_max_pool_enc = nn.MaxPool3d(kernel_size=(self.num_channels_in_enc, 1, 1))
        self.ch_avg_pool_enc = nn.AvgPool3d(kernel_size=(self.num_channels_in_enc, 1, 1))
        self.conv1_enc = nn.Conv2d(in_channels=self.num_channels_in_enc, out_channels=1, kernel_size=1, stride=1,
                                   padding=0)
        self.conv2_enc = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.ch_max_pool_dec = nn.MaxPool3d(kernel_size=(self.num_channels_in_dec, 1, 1))
        self.ch_avg_pool_dec = nn.AvgPool3d(kernel_size=(self.num_channels_in_dec, 1, 1))
        self.conv1_dec = nn.Conv2d(in_channels=self.num_channels_in_dec, out_channels=1, kernel_size=1, stride=1,
                                   padding=0)
        self.conv2_dec = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=7, stride=1, padding=3)

    def forward(self, enc, dec):
        enc = torch.cat(tensors=(self.ch_max_pool_enc(enc), self.ch_avg_pool_enc(enc), self.conv1_enc(enc)), dim=1)
        enc = self.conv2_enc(enc)
        dec = torch.cat(tensors=(self.ch_max_pool_dec(dec), self.ch_avg_pool_dec(dec), self.conv1_dec(dec)), dim=1)
        dec = self.conv2_dec(dec)
        out = torch.sigmoid(enc + dec)
        return out

    def initialize_weights(self):
        for layer in [self.conv1_enc, self.conv1_dec, self.conv2_enc, self.conv2_dec]:
            nn.init.normal_(layer.weight.data, mean=0.0, std=.02)
            nn.init.constant_(layer.bias.data, 0.0)


class cAG(nn.Module):
    def __init__(self, num_channels_in_enc, num_channels_in_dec):
        super(cAG, self).__init__()
        self.num_channels_in_enc = num_channels_in_enc
        self.num_channels_in_dec = num_channels_in_dec
        self.N = self.num_channels_in_enc // 8
        self.conv_enc_avg = nn.Conv2d(in_channels=self.num_channels_in_enc, out_channels=self.N, kernel_size=1,
                                      stride=1, padding=0)
        self.conv_enc_max = nn.Conv2d(in_channels=self.num_channels_in_enc, out_channels=self.N, kernel_size=1,
                                      stride=1, padding=0)
        self.conv_dec_avg = nn.Conv2d(in_channels=self.num_channels_in_dec, out_channels=self.N, kernel_size=1,
                                      stride=1, padding=0)
        self.conv_dec_max = nn.Conv2d(in_channels=self.num_channels_in_dec, out_channels=self.N, kernel_size=1,
                                      stride=1, padding=0)
        self.conv_fusion = nn.Conv2d(in_channels=self.N, out_channels=self.num_channels_in_enc, kernel_size=1, stride=1,
                                     padding=0)

    def forward(self, enc, dec):
        bs, ch, h, w = enc.shape
        enc_avg = F.avg_pool2d(enc, kernel_size=(h, w))
        enc_avg = self.conv_enc_avg(enc_avg)
        enc_max = F.max_pool2d(enc, kernel_size=(h, w))
        enc_max = self.conv_enc_max(enc_max)
        enc = enc_avg + enc_max

        bs, ch, h, w = dec.shape
        dec_avg = F.avg_pool2d(dec, kernel_size=(h, w))
        dec_avg = self.conv_dec_avg(dec_avg)
        dec_max = F.max_pool2d(dec, kernel_size=(h, w))
        dec_max = self.conv_dec_max(dec_max)
        dec = dec_avg + dec_max

        out = torch.sigmoid(self.conv_fusion(enc + dec))
        return out

    def initialize_weights(self):
        for layer in [self.conv_enc_avg, self.conv_enc_max, self.conv_dec_avg, self.conv_dec_max, self.conv_fusion]:
            nn.init.normal_(layer.weight.data, mean=0.0, std=.02)
            nn.init.constant_(layer.bias.data, 0.0)


class scAG_UNet(nn.Module):
    def __init__(self, in_channels, num_categories=2, filter_sizes=(64, 128, 256, 512, 1024), deep_supervision=False):
        super(scAG_UNet, self).__init__()
        self.in_channels = in_channels
        self.num_categories = num_categories
        self.filter_sizes = filter_sizes
        self.deep_supervision = deep_supervision
        self.main_model_name = 'scag_unet'

        # Encoder:
        self.enc1_Conv2dBlock = Conv2dBlock(self.in_channels, self.filter_sizes[0], True)
        self.enc2_Conv2dBlock = Conv2dBlock(self.filter_sizes[0], self.filter_sizes[1], True)
        self.enc3_Conv2dBlock = Conv2dBlock(self.filter_sizes[1], self.filter_sizes[2], True)
        self.enc4_Conv2dBlock = Conv2dBlock(self.filter_sizes[2], self.filter_sizes[3], True)
        self.enc5_Conv2dBlock = Conv2dBlock(self.filter_sizes[3], self.filter_sizes[4], True)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)

        # Decoder:
        self.dec4_scAG = scAG(self.filter_sizes[3], self.filter_sizes[4])
        self.dec4_UpConcat2d = UpConcat2d(self.filter_sizes[4], self.filter_sizes[3])
        self.dec4_Conv2dBlock = Conv2dBlock(self.filter_sizes[4], self.filter_sizes[3], True)
        self.dec3_scAG = scAG(self.filter_sizes[2], self.filter_sizes[3])
        self.dec3_UpConcat2d = UpConcat2d(self.filter_sizes[3], self.filter_sizes[2])
        self.dec3_Conv2dBlock = Conv2dBlock(self.filter_sizes[3], self.filter_sizes[2], True)
        self.dec2_scAG = scAG(self.filter_sizes[1], self.filter_sizes[2])
        self.dec2_UpConcat2d = UpConcat2d(self.filter_sizes[2], self.filter_sizes[1])
        self.dec2_Conv2dBlock = Conv2dBlock(self.filter_sizes[2], self.filter_sizes[1], True)
        self.dec1_scAG = scAG(self.filter_sizes[0], self.filter_sizes[1])
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
        dec4_att = self.dec4_scAG(enc4, enc5, save_attention)
        dec4 = self.dec4_UpConcat2d(enc5, dec4_att)
        dec4 = self.dec4_Conv2dBlock(dec4)
        dec3_att = self.dec3_scAG(enc3, dec4, save_attention)
        dec3 = self.dec3_UpConcat2d(dec4, dec3_att)
        dec3 = self.dec3_Conv2dBlock(dec3)
        dec2_att = self.dec2_scAG(enc2, dec3, save_attention)
        dec2 = self.dec2_UpConcat2d(dec3, dec2_att)
        dec2 = self.dec2_Conv2dBlock(dec2)
        dec1_att = self.dec1_scAG(enc1, dec2, save_attention)
        dec1 = self.dec1_UpConcat2d(dec2, dec1_att)
        dec1 = self.dec1_Conv2dBlock(dec1)
        if self.deep_supervision:
            dec1 = self.deep_sup_module(dec4, dec3, dec2, dec1)
        out = F.softmax(self.dec1_out(dec1), dim=1)

        return out

    def initialize_weights(self):
        for layer in [self.enc1_Conv2dBlock, self.enc2_Conv2dBlock, self.enc3_Conv2dBlock, self.enc4_Conv2dBlock,
                      self.enc5_Conv2dBlock, self.dec4_scAG, self.dec4_UpConcat2d, self.dec4_Conv2dBlock,
                      self.dec3_scAG, self.dec3_UpConcat2d, self.dec3_Conv2dBlock, self.dec2_scAG, self.dec2_UpConcat2d,
                      self.dec2_Conv2dBlock, self.dec1_scAG, self.dec1_UpConcat2d, self.dec1_Conv2dBlock]:
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
    model = scAG_UNet(in_channels=3, num_categories=2, filter_sizes=(64, 128, 256, 512, 1024), deep_supervision=True)
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
