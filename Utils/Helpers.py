import os
import pickle
import time
import imageio
import pathlib
import tifffile
import torch
import numpy as np
from PIL import Image
from Models.AttentionUNet import AttentionUNet
from Models.CBAM_UNet import CBAM_UNet
from Models.DenseNet121_UNet import DenseNet121_UNet
from Models.DualAttentionUNet import DualAttentionUNet
from Models.MobileNetV2_UNet import MobileNetV2_UNet
from Models.ResNet34_UNet import ResNet34_UNet
from Models.ResidualAttentionUNet import ResidualAttentionUNet
from Models.UNet import UNet
from Models.scAG_UNet import scAG_UNet
from Models.vgg11_UNet import vgg11_UNet
from Models.vgg16_UNet import vgg16_UNet
from Utils.SolarisHelpers import footprint_mask


def save_pickle_obj(obj, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


def load_pickle_obj(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def get_segmentation_mask(path_geojson, path_im, num_categories_assert=2):
    mask = np.array(footprint_mask(df=str(path_geojson), reference_im=str(path_im), burn_value=1, do_transform=True),
                    dtype=np.uint8)
    assert np.min(mask) >= 0
    assert np.max(mask) < num_categories_assert
    return mask


def np_channels_last_to_first(arr):
    # 3d array resort: channels last to first; e.g. (444, 444, 3) -> (3, 444, 444)
    out = np.empty((arr.shape[2], arr.shape[0], arr.shape[1]), dtype=np.float32)
    for i in range(arr.shape[2]):
        out[i, :, :] = arr[:, :, i]
    return out


def display_progress_bar(mode, num_epoch, num_batch, num_batches, avg_running_loss, start_epoch):
    if num_batch < num_batches:
        num_stars = int((num_batch / num_batches * 100) // 2)
        bar = '[' + (num_stars * '#') + ((50 - num_stars) * '-') + ']'
        current_time = time.time()
        est_time_left = int((current_time - start_epoch) / (num_batch / num_batches) - (current_time - start_epoch))
        eol = ''
    else:
        bar = '[' + (50 * '#') + ']'
        est_time_left = 0
        eol = '\n'
        if mode == 'val':
            eol += '\n'
    out = f'\repoch: {num_epoch}\tmode: {mode}\t{num_batch}/{num_batches}      \t{bar}\tloss: {avg_running_loss:.4f}\testimtated time left: {format_to_time_string(est_time_left)}{eol}'
    print(out + (10 * ' '), end='')


def format_to_time_string(time_sec):
    days = time_sec // 86400
    hours = time_sec // 3600 - 24 * days
    mins = time_sec // 60 - 60 * (24 * days + hours)
    secs = time_sec % 60
    s = f'{f"{days}d " if days != 0 else ""}{f"{hours}h " if hours != 0 else ""}{f"{mins}m " if mins != 0 else ""}'
    return f'{s}{f"{secs}s" if (secs != 0 or (secs == 0 and s == "")) else ""}'


def remove_old_model_data(model_name):
    training_hist_path = pathlib.Path(f'training_history/{model_name}_hist.pkl')
    models_path = pathlib.Path('models')
    if training_hist_path.exists():
        os.remove(training_hist_path)
    if models_path.exists():
        for p in models_path.iterdir():
            if model_name in str(p.name):
                os.remove(p)


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def crop_center_Y(out, Y):
    # out: torch.Size([1, 2, 36, 44]), Y:torch.Size([1, 406, 439])
    start = [(Y.shape[-2] - out.shape[-2]) // 2, (Y.shape[-1] - out.shape[-1]) // 2]
    length = [out.shape[-2], out.shape[-1]]
    Y = torch.narrow(torch.narrow(Y, dim=1, start=start[0], length=length[0]), dim=2, start=start[1],
                     length=length[1])
    return Y


def get_experiment_objs(exp_dir, create_model=True):
    settings = load_pickle_obj(exp_dir / 'settings.pkl')
    params = dict(num_categories=settings['num_categories'], pretrained_on=settings['pretrained_on'],
                  num_epochs=settings['num_epochs'], device=settings['device'])
    model_name = settings['model_name']
    model = None
    if create_model:
        if settings['continue_training']:
            assert settings['initialize_weights'] is False and settings['continue_model_path'] is not None
            model = torch.load(settings['continue_model_path'])
        else:
            # base model:
            if model_name == 'unet':
                model = UNet(in_channels=settings['in_channels'], num_categories=settings['num_categories'],
                             filter_sizes=settings['filter_sizes'], deep_supervision=settings['deep_supervision'])
            # attention models:
            elif model_name == 'attention_unet':
                model = AttentionUNet(in_channels=settings['in_channels'], num_categories=settings['num_categories'],
                                      filter_sizes=settings['filter_sizes'],
                                      deep_supervision=settings['deep_supervision'])
            elif model_name == 'cbam_unet':
                model = CBAM_UNet(in_channels=settings['in_channels'], num_categories=settings['num_categories'],
                                  filter_sizes=settings['filter_sizes'], deep_supervision=settings['deep_supervision'])
            elif model_name == 'dualattention_unet':
                model = DualAttentionUNet(in_channels=settings['in_channels'],
                                          num_categories=settings['num_categories'],
                                          filter_sizes=settings['filter_sizes'],
                                          deep_supervision=settings['deep_supervision'])
            elif model_name == 'residualattention_unet':
                model = ResidualAttentionUNet(in_channels=settings['in_channels'],
                                              num_categories=settings['num_categories'],
                                              filter_sizes=settings['filter_sizes'],
                                              deep_supervision=settings['deep_supervision'])
            elif model_name == 'scag_unet':
                model = scAG_UNet(in_channels=settings['in_channels'], num_categories=settings['num_categories'],
                                  filter_sizes=settings['filter_sizes'], deep_supervision=settings['deep_supervision'])
            # transfer learning models:
            elif model_name == 'densenet121_unet':
                model = DenseNet121_UNet(in_channels=settings['in_channels'], num_categories=settings['num_categories'],
                                         deep_supervision=settings['deep_supervision'], pretrained=True)
            elif model_name == 'mobilenetv2_unet':
                model = MobileNetV2_UNet(in_channels=settings['in_channels'], num_categories=settings['num_categories'],
                                         deep_supervision=settings['deep_supervision'], pretrained=True)
            elif model_name == 'resnet34_unet':
                model = ResNet34_UNet(in_channels=settings['in_channels'], num_categories=settings['num_categories'],
                                      deep_supervision=settings['deep_supervision'], pretrained=True)
            elif model_name == 'vgg11_unet':
                model = vgg11_UNet(in_channels=settings['in_channels'], num_categories=settings['num_categories'],
                                   deep_supervision=settings['deep_supervision'], pretrained=True)
            elif model_name == 'vgg16_unet':
                model = vgg16_UNet(in_channels=settings['in_channels'], num_categories=settings['num_categories'],
                                   deep_supervision=settings['deep_supervision'], pretrained=True)
            assert model is not None
    dataloader_path = settings['dataloader_path']
    learning_rate_dict = dict(static_lr=settings['learning_rate_dict']['static_lr'],
                              use_cyclic_learning_rate=settings['learning_rate_dict']['use_cyclic_learning_rate'],
                              base_lr=settings['learning_rate_dict']['base_lr'],
                              max_lr=settings['learning_rate_dict']['max_lr'])
    return params, model, dataloader_path, learning_rate_dict, settings


def get_model_name(model):
    filter_sizes_desc = f'_{model.filter_sizes[0]}-{model.filter_sizes[-1]}' if (model.filter_sizes is not None) else ''
    supervision_desc = '_wDeepSupervision' if model.deep_supervision else ''
    return f'{model.main_model_name}{filter_sizes_desc}{supervision_desc}'


def get_num_epochs_from_model_path(path):
    filename = path.name
    last = filename.split('_')[-1].replace('.pt', '')
    if last.isdigit():
        return last
    return None


def normalize_channelwise(arr):
    # expected input of shape (h, w, ch)
    for ch in range(arr.shape[-1]):
        minimum = np.min(arr[:, :, ch])
        maximum = np.max(arr[:, :, ch])
        arr[:, :, ch] = (arr[:, :, ch] - minimum) / (maximum - minimum)
    return arr

if __name__ == '__main__':
    from pathlib import Path
    model_path = Path('../models/unet_32-512_wDeepSupervision_120.pt')
    print(get_num_epochs_from_model_path(model_path))
