import os
import random
import torch
import numpy as np
from pathlib import Path
from Utils.Dataloader import Dataloader


params = dict(dataset_name='SN1_Buildings', im_type='3band', path_dataset_folder=Path('data/SN1_Buildings/train'),
              amount_train=.6, amount_val=.2, amount_test=.2, delete_old_npy_data=False, data_fraction=1.0)

### prepare:
random.seed(42)
np.random.seed(42)
if params['delete_old_npy_data']:
    dirs = [Path(f'data/{params["dataset_name"]}/train/{params["im_type"]}'),
            Path(f'data/{params["dataset_name"]}/train/geojson')]
    for dir in dirs:
        for p in dir.iterdir():
            if '.npy' in p.name:
                os.remove(p)

### init dataloader:
dataloader = Dataloader(**params)
dataloader_path = Path(f'tmp/Dataloader_{params["dataset_name"]}.pkl')
dataloader.init()
dataloader.save(dataloader_path)

### calculate standardization parameters:
mean_path = Path(f'tmp/mean_{params["dataset_name"]}.npy')
std_path = Path(f'tmp/std_{params["dataset_name"]}.npy')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Mean:
n = torch.tensor([len(dataloader.train_objs)]).to(device)
mean_0 = torch.zeros(1).to(device)
mean_1 = torch.zeros(1).to(device)
mean_2 = torch.zeros(1).to(device)
for i, obj in enumerate(dataloader.train_objs):
    im = torch.from_numpy(np.load(obj.im_npy_path) / (2 ** obj.bits - 1)).to(device)
    mean_0 += torch.sum(im[:, :, 0]) / (n * im.shape[0] * im.shape[1])
    mean_1 += torch.sum(im[:, :, 1]) / (n * im.shape[0] * im.shape[1])
    mean_2 += torch.sum(im[:, :, 2]) / (n * im.shape[0] * im.shape[1])
    print(f'\rmean: {i + 1}/{len(dataloader.train_objs)}', end='')
mean = np.array([mean_0.item(), mean_1.item(), mean_2.item()])
np.save(mean_path, mean)
print('')

# Std:
m = torch.zeros(1).to(device)
std_0 = torch.zeros(1).to(device)
std_1 = torch.zeros(1).to(device)
std_2 = torch.zeros(1).to(device)
for i, obj in enumerate(dataloader.train_objs):
    im = torch.from_numpy(np.load(obj.im_npy_path) / (2 ** obj.bits - 1)).to(device)
    std_0 += torch.sum((im[:, :, 0] - mean_0) ** 2)
    std_1 += torch.sum((im[:, :, 1] - mean_1) ** 2)
    std_2 += torch.sum((im[:, :, 2] - mean_2) ** 2)
    m += im.shape[0] * im.shape[1]
    print(f'\rstd: {i + 1}/{len(dataloader.train_objs)}', end='')
std_0 = torch.sqrt(std_0 / m)
std_1 = torch.sqrt(std_1 / m)
std_2 = torch.sqrt(std_2 / m)
std = np.array([std_0.item(), std_1.item(), std_2.item()])
np.save(std_path, std)

print(f'\nmean: {mean}')
print(f'std: {std}')
