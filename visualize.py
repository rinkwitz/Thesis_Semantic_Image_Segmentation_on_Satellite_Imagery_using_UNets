import torch
import random
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from Utils.Dataloader import Dataloader
from Utils.Helpers import *
from pathlib import Path
from Utils.Metrics import *


# Params:
# params = dict(num_categories=2, pretrained_on=None, num_epochs=100,
#               device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
# model_path = Path('models/cbam_unet_32-512_wDeepSupervision_100.pt')

params = dict(num_categories=2, pretrained_on='Imagenet', num_epochs=100,
              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model_path = Path('models/vgg11_unet_wDeepSupervision_60.pt')

decicision_threshold = .5
# multimetrics_path = Path('tmp/scag_unet_32-512_wDeepSupervision_metrics_multithresholds_28.pkl')
# decicision_threshold = load_pickle_obj(multimetrics_path)['thresholds'][np.argmax(load_pickle_obj(multimetrics_path)['f1s'])]
dataloader_path = Path('tmp/Dataloader_SN1_Buildings.pkl')
# num_image = 346
# mode = 'train'
# num_image = 76
# mode = 'val'
# num_image = 125
# mode = 'test'
num_image = 207
mode = 'test'
# num_image = 254
# mode = 'test'

# Visualize:
random.seed(42)
np.random.seed(42)
dataloader = Dataloader()
dataloader.load(dataloader_path)
dataloader.on_epoch_start(params)
try:
    model = torch.load(model_path)
except RuntimeError as e:
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
model.eval()
if mode == 'train':
    obj = dataloader.train_objs[num_image]
elif mode == 'val':
    obj = dataloader.val_objs[num_image]
elif mode == 'test':
    obj = dataloader.test_objs[num_image]
X, Y = dataloader.get_mini_batch(num_image, params, mode=mode, data_augmentation=False, weight=False)
out = model(X)
bs_out, ch_out, h_out, w_out = out.shape
_, h_Y, w_Y = Y.shape
Y = crop_center_Y(out, Y)

pred = out[0, 1, :, :].view((h_out, w_out))
pred = (pred > decicision_threshold).type(torch.uint8)
pred_rgb = np.zeros((h_out, w_out, 3), dtype=np.uint8)
pred_rgb[:, :, 0] = pred.to('cpu').detach().numpy() * 255

fig = plt.figure(figsize=(20, 10), dpi=100)
ax_0 = fig.add_subplot(131)
im = np.load(obj.im_npy_path)
h_im, w_im, ch_im = im.shape
h_offset = (h_im - h_out) // 2
w_offset = (w_im - w_out) // 2
ax_0.imshow(im[h_offset:h_offset + h_out, w_offset:w_offset + w_out, :])
ax_0.set_title('original center crop')

ax_1 = fig.add_subplot(132)
ax_1.imshow(pred_rgb)
ax_1.imshow(Y.to('cpu').detach().numpy().reshape((h_out, w_out)), cmap='gray', alpha=.25)
ax_1.set_title('mask')

ax_2 = fig.add_subplot(133)
prediction_scaled = np.array(
    Image.fromarray(out.to('cpu').detach().numpy()[0, 1, :, :].reshape((h_out, w_out))))
pred_im = ax_2.imshow(prediction_scaled, cmap='plasma', vmin=0.0, vmax=1.0)
ax_2.set_title('prediction')
x0, y0, width, height = ax_2.get_position().bounds
cbaxes = fig.add_axes([x0 + width + .01, y0 + .005, .01, height - .01])
cb = plt.colorbar(pred_im, cax=cbaxes)
plt.show()

print(f'\nacc:\t\t{calc_Accuracy(pred, Y):.4f}')
print(f'iou:\t\t{calc_IOU(pred, Y, zero_division_safe=True):.4f}')
print(f'f1:\t\t\t{calc_F1(pred, Y, zero_division_safe=True):.4f}')
print(f'rec:\t\t{calc_Recall(pred, Y, zero_division_safe=True):.4f}')
print(f'prec:\t\t{calc_Precision(pred, Y, zero_division_safe=True):.4f}')
print(f'spec:\t\t{calc_Specificity(pred, Y, zero_division_safe=True):.4f}')
