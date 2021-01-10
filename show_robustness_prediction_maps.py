import random
import sys
from math import exp
import numpy as np
from Utils.Metrics import *
from Models.UNet import *
from Utils.Helpers import *
from Utils.Dataloader import Dataloader
from pathlib import Path
import matplotlib.pyplot as plt


num_image = 207

# exp_dir_paths = [(Path('experiments/experiment_0'), 'U-Net'),
#                  (Path('experiments/experiment_1'), 'Attention U-Net'),
#                  (Path('experiments/experiment_2'), 'CBAM U-Net'),
#                  (Path('experiments/experiment_3'), 'Residual Attention U-Net'),
#                  (Path('experiments/experiment_4'), 'scAG U-Net')]

exp_dir_paths = [(Path('experiments/experiment_5'), 'DenseNet121 U-Net'),
                 (Path('experiments/experiment_6'), 'MobileNetV2 U-Net'),
                 (Path('experiments/experiment_7'), 'ResNet34 U-Net'),
                 (Path('experiments/experiment_8'), 'VGG11 U-Net')]

# Prepare:
for idx_exp_dir, t in enumerate(exp_dir_paths):
    exp_dir, label = t
    params, model, dataloader_path, learning_rate_dict, experiment_settings = get_experiment_objs(exp_dir, create_model=True)
    params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = get_model_name(model)
    if '0' in exp_dir.name:
        model_path = exp_dir / 'unet_32-512_wDeepSupervision_116.pt'
        threshold = 0.3684
    elif '1' in exp_dir.name:
        model_path = exp_dir / 'attention_unet_32-512_wDeepSupervision_108.pt'
        threshold = 0.4737
    elif '2' in exp_dir.name:
        model_path = exp_dir / 'cbam_unet_32-512_wDeepSupervision_136.pt'
        threshold = 0.3684
    elif '3' in exp_dir.name:
        model_path = exp_dir / 'residualattention_unet_32-512_wDeepSupervision_188.pt'
        threshold = 0.2632
    elif '4' in exp_dir.name:
        model_path = exp_dir / 'scag_unet_32-512_wDeepSupervision_192.pt'
        threshold = 0.4737
    elif '5' in exp_dir.name:
        model_path = exp_dir / 'densenet121_unet_wDeepSupervision_56.pt'
        threshold = 0.4211
    elif '6' in exp_dir.name:
        model_path = exp_dir / 'mobilenetv2_unet_wDeepSupervision_28.pt'
        threshold = 0.7895
    elif '7' in exp_dir.name:
        model_path = exp_dir / 'resnet34_unet_wDeepSupervision_40.pt'
        threshold = 0.4737
    elif '8' in exp_dir.name:
        model_path = exp_dir / 'vgg11_unet_wDeepSupervision_52.pt'
        threshold = 0.5263
    print(str(model_path.absolute()))
    try:
        model = torch.load(model_path)
    except RuntimeError as e:
        model = torch.load(model_path, map_location=lambda storage, loc: storage)
    model_name = get_model_name(model)
    model.to(params['device'])
    model.eval()
    dataloader = Dataloader()
    dataloader.load(dataloader_path)
    dataloader.on_epoch_start(params)

    for idx_noise_level, noise_level in enumerate(np.linspace(0.0, 1.0, 6)):
        X, Y = dataloader.get_mini_batch(num_image, params, mode='test', weight=False, data_augmentation=False)
        assert X is not None
        assert torch.min(Y) >= 0
        assert torch.max(Y) < params['num_categories']
        out = model(X + noise_level * torch.randn(X.shape).to(params['device']))
        _, _, h, w = out.shape
        pred = out[0, 1, :, :].view((h, w)).to('cpu').detach().numpy()
        span = 3
        plt.subplot2grid((span * len(exp_dir_paths), span * 7), (span * idx_exp_dir, span * (idx_noise_level + 1)), colspan=span, rowspan=span)
        plt.imshow(pred, cmap='plasma', vmin=0.0, vmax=1.0)
        if idx_exp_dir + 1 == len(exp_dir_paths):
            plt.xlabel(f'sigma = {noise_level:.1f}')
        plt.xticks([])
        plt.yticks([])
    plt.subplot2grid((span * len(exp_dir_paths), span * 7), (span * idx_exp_dir, 0), colspan=span, rowspan=span)
    im = np.load(dataloader.test_objs[num_image].im_npy_path)
    im_h, im_w, _ = im.shape
    im = im[(im_h - h) // 2:((im_h - h) // 2) + h, (im_w - w) // 2:((im_w - w) // 2) + w, :]
    if idx_exp_dir + 1 == len(exp_dir_paths):
        plt.xlabel('center crop')
    plt.ylabel(f'{label}')
    plt.imshow(im, cmap='plasma', vmin=0.0, vmax=1.0)
    plt.xticks([])
    plt.yticks([])
# plt.tight_layout()
plt.show()
