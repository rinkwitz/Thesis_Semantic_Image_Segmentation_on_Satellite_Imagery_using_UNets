import torch
import torch.nn as nn
from pathlib import Path
from Models.utils.RandInputImage import RandInputImage


class EncoderActivationModel(nn.Module):
    def __init__(self, model_path):
        super(EncoderActivationModel, self).__init__()
        self.im = RandInputImage()
        try:
            model = torch.load(model_path)
        except RuntimeError as e:
            model = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.enc1_Conv2dBlock = model.enc1_Conv2dBlock
        self.enc2_Conv2dBlock = model.enc2_Conv2dBlock
        self.enc3_Conv2dBlock = model.enc3_Conv2dBlock
        self.enc4_Conv2dBlock = model.enc4_Conv2dBlock
        self.enc5_Conv2dBlock = model.enc5_Conv2dBlock
        self.pool = model.pool
        self.freeze_all_except_randimage()

    def forward(self, num_encoder):
        if num_encoder == 1:
            enc1 = self.enc1_Conv2dBlock(self.im.forward())
            return enc1
        elif num_encoder == 2:
            enc1 = self.enc1_Conv2dBlock(self.im.forward())
            enc2 = self.pool(enc1)
            enc2 = self.enc2_Conv2dBlock(enc2)
            return enc2
        elif num_encoder == 3:
            enc1 = self.enc1_Conv2dBlock(self.im.forward())
            enc2 = self.pool(enc1)
            enc2 = self.enc2_Conv2dBlock(enc2)
            enc3 = self.pool(enc2)
            enc3 = self.enc3_Conv2dBlock(enc3)
            return enc3
        elif num_encoder == 4:
            enc1 = self.enc1_Conv2dBlock(self.im.forward())
            enc2 = self.pool(enc1)
            enc2 = self.enc2_Conv2dBlock(enc2)
            enc3 = self.pool(enc2)
            enc3 = self.enc3_Conv2dBlock(enc3)
            enc4 = self.pool(enc3)
            enc4 = self.enc4_Conv2dBlock(enc4)
            return enc4
        elif num_encoder == 5:
            enc1 = self.enc1_Conv2dBlock(self.im.forward())
            enc2 = self.pool(enc1)
            enc2 = self.enc2_Conv2dBlock(enc2)
            enc3 = self.pool(enc2)
            enc3 = self.enc3_Conv2dBlock(enc3)
            enc4 = self.pool(enc3)
            enc4 = self.enc4_Conv2dBlock(enc4)
            enc5 = self.pool(enc4)
            enc5 = self.enc5_Conv2dBlock(enc5)
            return enc5

    def get_image_tensor(self):
        return self.im.im.data

    def set_image_tensor(self, new):
        self.im.im.data = new

    def freeze_all(self):
        for layer in [self.enc1_Conv2dBlock, self.enc2_Conv2dBlock, self.enc3_Conv2dBlock, self.enc4_Conv2dBlock,
                      self.enc5_Conv2dBlock, self.im]:
            layer.requires_grad_(False)

    def freeze_all_except_randimage(self):
        for layer in [self.enc1_Conv2dBlock, self.enc2_Conv2dBlock, self.enc3_Conv2dBlock, self.enc4_Conv2dBlock,
                      self.enc5_Conv2dBlock]:
            layer.requires_grad_(False)
        self.im.im.data.requires_grad_(True)


if __name__ == '__main__':
    model = EncoderActivationModel(Path('../../models/unet_32-512_wDeepSupervision_120.pt'))
    print(model.forward(3).shape)
