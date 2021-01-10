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
model_path = Path('models/scag_unet_32-512_wDeepSupervision_28.pt')
dataloader_path = Path('tmp/Dataloader_SN1_Buildings.pkl')

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
    elif '1' in exp_dir.name:
        model_path = exp_dir / 'attention_unet_32-512_wDeepSupervision_108.pt'
    elif '2' in exp_dir.name:
        model_path = exp_dir / 'cbam_unet_32-512_wDeepSupervision_136.pt'
    elif '3' in exp_dir.name:
        model_path = exp_dir / 'residualattention_unet_32-512_wDeepSupervision_188.pt'
    elif '4' in exp_dir.name:
        model_path = exp_dir / 'scag_unet_32-512_wDeepSupervision_192.pt'
    elif '5' in exp_dir.name:
        model_path = exp_dir / 'densenet121_unet_wDeepSupervision_56.pt'
    elif '6' in exp_dir.name:
        model_path = exp_dir / 'mobilenetv2_unet_wDeepSupervision_28.pt'
    elif '7' in exp_dir.name:
        model_path = exp_dir / 'resnet34_unet_wDeepSupervision_40.pt'
    elif '8' in exp_dir.name:
        model_path = exp_dir / 'vgg11_unet_wDeepSupervision_52.pt'
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
num_thresholds = 20
num_batches = len(dataloader.test_objs)
dataloader.on_epoch_start(params)
X, Y = dataloader.get_mini_batch(0, params, mode='test', weight=False, data_augmentation=False)
out = model(X)
_, _, h_out, w_out = out.shape
Predictions = torch.empty((num_batches * num_thresholds, h_out, w_out), dtype=torch.uint8).to('cpu')
Labels = torch.empty((num_batches, h_out, w_out), dtype=torch.uint8).to('cpu')
for num_batch in range(num_batches):
    X, Y = dataloader.get_mini_batch(num_batch, params, mode='test', weight=False, data_augmentation=False)
    assert X is not None
    assert torch.min(Y) >= 0
    assert torch.max(Y) < params['num_categories']
    out = model(X)
    _, _, h, w = out.shape
    assert h_out == h and w_out == w
    pred = out[0, 1, :, :].view((h_out, w_out))
    for i, threshold in enumerate(np.linspace(0.0, 1.0, num_thresholds)):
        pred_threshold = (pred > threshold).type(torch.uint8)
        Predictions[num_batch + num_batches * i, :, :] = pred_threshold.to('cpu')
    Y = crop_center_Y(out, Y)
    Y = Y.view((h_out, w_out)).type(torch.uint8)
    Labels[num_batch, :, :] = Y.to('cpu')
    print(f'\r{num_batch + 1}/{num_batches}', end='')

thresholds, accs, ious, f1s, recs, precs, specs = [], [], [], [], [], [], []
for i, threshold in enumerate(np.linspace(0.0, 1.0, num_thresholds)):
    thresholds.append(threshold)
    Predictions_thresholded = Predictions[num_batches * i:num_batches * (i + 1), :, :]
    accs.append(calc_Accuracy(Predictions_thresholded, Labels))
    ious.append(calc_IOU(Predictions_thresholded, Labels, zero_division_safe=True))
    f1s.append(calc_F1(Predictions_thresholded, Labels, zero_division_safe=True))
    recs.append(calc_Recall(Predictions_thresholded, Labels, zero_division_safe=True))
    precs.append(calc_Precision(Predictions_thresholded, Labels, zero_division_safe=True))
    specs.append(calc_Specificity(Predictions_thresholded, Labels, zero_division_safe=True))

metrics_results = dict(model_name=model_name, thresholds=thresholds, accs=accs, ious=ious, f1s=f1s, recs=recs,
                       precs=precs, specs=specs)

num_epochs_suffix = f'_{get_num_epochs_from_model_path(model_path)}' if get_num_epochs_from_model_path(model_path) is not None else ''
if running_experiment:
    results_path = exp_dir / f'{model_name}_metrics_multithresholds{num_epochs_suffix}.pkl'
else:
    results_path = f'tmp/{model_name}_metrics_multithresholds{num_epochs_suffix}.pkl'
save_pickle_obj(metrics_results, results_path)
