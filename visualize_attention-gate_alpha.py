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
# model_path = Path('experiments/experiment_1/attention_unet_32-512_wDeepSupervision_108.pt')
model_path = Path('experiments/experiment_3/residualattention_unet_32-512_wDeepSupervision_188.pt')
params = dict(num_categories=2, pretrained_on=None, num_epochs=100,
              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
dataloader_path = Path('tmp/Dataloader_SN1_Buildings.pkl')
num_image = 207
mode = 'test'

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
out = model(X, save_attention=True)

fig = plt.figure(figsize=(20, 10), dpi=100)
tensor_paths = [p for p in Path('tmp').iterdir() if 'attention-gate_alpha_' in p.name]
tensor_paths.sort(key=lambda p:torch.load(p).shape[-2], reverse=True)
for i, tensor_path in enumerate(tensor_paths):
    attention_map = torch.load(tensor_path).to('cpu').detach().numpy()
    _, _, h_att, w_att = attention_map.shape
    attention_map = attention_map.reshape((h_att, w_att))
    ax = fig.add_subplot(int(f'22{i + 1}'))
    ax.set_title(f'attention map {i + 1}')
    assert np.max(attention_map) <= 1.0 and np.min(attention_map) >= 0.0
    # att = ax.imshow(attention_map, cmap='plasma', alpha=1.0, vmin=0.0, vmax=1.0)
    att = ax.imshow(attention_map, cmap='plasma', alpha=1.0, vmin=np.min(attention_map), vmax=np.max(attention_map))
    x0, y0, width, height = ax.get_position().bounds
    cbaxes = fig.add_axes([x0 + width + .01, y0, .01, height])
    cb = plt.colorbar(att, cax=cbaxes)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
print(str(model_path.absolute()))

# clean up:
for tensor_path in tensor_paths:
    os.remove(str(tensor_path.absolute()))
