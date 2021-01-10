import random
import sys
from math import exp

import numpy as np
from Utils.Metrics import *
from Models.UNet import *
from Utils.Helpers import *
from Utils.Dataloader import Dataloader
from pathlib import Path

# Params:
params = dict(num_categories=2, pretrained_on=None, num_epochs=100,
              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model_path = Path('models/cbam_unet_32-512_wDeepSupervision_100.pt')
dataloader_path = Path('tmp/Dataloader_SN1_Buildings.pkl')
threshold = .5

# Prepare:
random.seed(42)
np.random.seed(42)
running_experiment = False
if len(sys.argv) > 1:
    exp_dir = Path(sys.argv[1])
    running_experiment = True
    params, model, dataloader_path, learning_rate_dict, experiment_settings = get_experiment_objs(exp_dir,
                                                                                                  create_model=True)
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

# Metrics:
num_noise_levels = 21
max_noise = 1.0
num_batches = len(dataloader.test_objs)
dataloader.on_epoch_start(params)
X, Y = dataloader.get_mini_batch(0, params, mode='test', weight=False, data_augmentation=False)
out = model(X)
_, _, h_out, w_out = out.shape
Predictions = torch.empty((num_batches, h_out, w_out), dtype=torch.uint8).to('cpu')
Labels = torch.empty((num_batches, h_out, w_out), dtype=torch.uint8).to('cpu')
noise_levels, accs, ious, f1s, recs, precs, specs = [], [], [], [], [], [], []
for noise_level in np.linspace(0.0, max_noise, num_noise_levels):
    for num_batch in range(num_batches):
        X, Y = dataloader.get_mini_batch(num_batch, params, mode='test', weight=False, data_augmentation=False)
        assert X is not None
        assert torch.min(Y) >= 0
        assert torch.max(Y) < params['num_categories']
        out = model(X + noise_level * torch.randn(X.shape).to(params['device']))
        _, _, h, w = out.shape
        assert h_out == h and w_out == w
        pred = out[0, 1, :, :].view((h_out, w_out))
        pred_threshold = (pred > threshold).type(torch.uint8)
        Predictions[num_batch, :, :] = pred_threshold.to('cpu')
        Y = crop_center_Y(out, Y)
        Y = Y.view((h_out, w_out)).type(torch.uint8)
        Labels[num_batch, :, :] = Y.to('cpu')
        print(f'\rnoise_level {noise_level:.4f}: {num_batch + 1}/{num_batches}' + (10 * ' '), end='')

    noise_levels.append(noise_level)
    accs.append(calc_Accuracy(Predictions, Labels))
    ious.append(calc_IOU(Predictions, Labels, zero_division_safe=True))
    f1s.append(calc_F1(Predictions, Labels, zero_division_safe=True))
    recs.append(calc_Recall(Predictions, Labels, zero_division_safe=True))
    precs.append(calc_Precision(Predictions, Labels, zero_division_safe=True))
    specs.append(calc_Specificity(Predictions, Labels, zero_division_safe=True))

robustness_results = dict(model_name=model_name, noise_levels=noise_levels, accs=accs, ious=ious, f1s=f1s, recs=recs,
                       precs=precs, specs=specs)

num_epochs_suffix = f'_{get_num_epochs_from_model_path(model_path)}' if get_num_epochs_from_model_path(model_path) is not None else ''
if running_experiment:
    results_path = exp_dir / f'{model_name}_robustness_multi_noiselevel{num_epochs_suffix}.pkl'
else:
    results_path = f'tmp/{model_name}_robustness_multi_noiselevel{num_epochs_suffix}.pkl'
save_pickle_obj(robustness_results, results_path)
