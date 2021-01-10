import sys
import random
import torch

from Models.MobileNetV2_UNet import MobileNetV2_UNet
from Utils.CyclicLearningRate import get_cyclic_lr
from Models.UNet import UNet
from Models.AttentionUNet import AttentionUNet
from Models.DualAttentionUNet import DualAttentionUNet
from Models.scAG_UNet import scAG_UNet
from Utils.Helpers import *
from Utils.Dataloader import Dataloader
from pathlib import Path
from Utils.Losses import DICE_coefficient
import torch.optim as optim


# Training Params:
params = dict(num_categories=2, pretrained_on='Imagenet', num_epochs=100,
              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model = MobileNetV2_UNet(in_channels=3, num_categories=params['num_categories'])
dataloader_path = Path('tmp/Dataloader_SN1_Buildings.pkl')
learning_rate_dict = dict(static_lr=1e-4, use_cyclic_learning_rate=True, base_lr=3e-3, max_lr=1e-2)
momentum = .95
running_experiment = False
initialize_weights = True
freeze_encoder = False
continue_training = False
continue_from_epoch = 0
training_history = {
    'avg_loss_train': [],
    'avg_loss_val': [],
}
if len(sys.argv) > 1:
    exp_dir = Path(sys.argv[1])
    running_experiment = True
    params, model, dataloader_path, learning_rate_dict, experiment_settings = get_experiment_objs(exp_dir)
    initialize_weights = experiment_settings['initialize_weights']
    freeze_encoder = experiment_settings['freeze_encoder']
    continue_training = experiment_settings['continue_training']
    continue_from_epoch = experiment_settings['continue_from_epoch']
model.to(params['device'])
if initialize_weights:
    assert not continue_training
    if params['pretrained_on'] is None:
        model.initialize_weights()
    else:
        model.initialize_decoder_weights()
if freeze_encoder:
    model.freeze_encoder()
else:
    model.unfreeze_encoder()
if continue_training and learning_rate_dict['use_cyclic_learning_rate']:
    assert continue_from_epoch % 4 == 0
if continue_training:
    training_history = load_pickle_obj(experiment_settings['continue_model_path'].parent / 'training_history.pkl')
model_name = get_model_name(model)
optimizer = optim.SGD(model.parameters(), lr=learning_rate_dict['static_lr'], momentum=momentum)


# Prepare:
random.seed(42)
np.random.seed(42)
if learning_rate_dict['use_cyclic_learning_rate']:
    assert params['num_epochs'] % 4 == 0
if running_experiment:
    training_history_path = exp_dir / 'training_history.pkl'
else:
    training_history_path = Path(f'training_history/{model_name}_training_history.pkl')
if not Path('models').exists():
    Path('models').mkdir()
dataloader = Dataloader()
dataloader.load(dataloader_path)


# Training:
for num_epoch in range(continue_from_epoch, continue_from_epoch + params['num_epochs']):
    dataloader.on_epoch_start(params)

    # train:
    start_epoch = time.time()
    model.train()
    running_loss_train = 0.0
    num_batches = len(dataloader.train_objs)
    for num_batch in range(num_batches):
        X, Y = dataloader.get_mini_batch(num_batch, params, mode='train', weight=False, data_augmentation=True)
        if learning_rate_dict['use_cyclic_learning_rate']:
            lr = get_cyclic_lr(learning_rate_dict, num_epoch, num_batches, num_batch)
            set_lr(optimizer, lr)
        optimizer.zero_grad()
        if X is not None:
            assert torch.min(Y) >= 0
            assert torch.max(Y) < params['num_categories']
            out = model(X)
            Y = crop_center_Y(out, Y)
            loss = 1 - DICE_coefficient(out, Y, zero_division_safe=True)
            running_loss_train += loss.item()
            loss.backward()
            optimizer.step()
        display_progress_bar('train', num_epoch + 1, num_batch + 1, num_batches, running_loss_train / (num_batch + 1),
                             start_epoch)
    avg_loss_train = running_loss_train / num_batches

    # val:
    start_epoch = time.time()
    model.eval()
    running_loss_val = 0.0
    num_batches = len(dataloader.val_objs)
    for num_batch in range(num_batches):
        X, Y = dataloader.get_mini_batch(num_batch, params, mode='val', weight=False, data_augmentation=False)
        if X is not None:
            assert torch.min(Y) >= 0
            assert torch.max(Y) < params['num_categories']
            out = model(X)
            Y = crop_center_Y(out, Y)
            loss = 1 - DICE_coefficient(out, Y, zero_division_safe=True)
            running_loss_val += loss.item()
        display_progress_bar('val', num_epoch + 1, num_batch + 1, num_batches, running_loss_val / (num_batch + 1),
                             start_epoch)
    avg_loss_val = running_loss_val / num_batches

    # save model:
    if running_experiment:
        if experiment_settings['save_model_every_epoch']:
            model_save_path = exp_dir / f'{model_name}_{num_epoch + 1}.pt'
        else:
            model_save_path = exp_dir / f'{model_name}.pt'
    else:
        model_save_path = Path(f'models/{model_name}_{num_epoch + 1}.pt')
    torch.save(model, model_save_path)

    # training history:
    training_history['avg_loss_train'].append(avg_loss_train)
    training_history['avg_loss_val'].append(avg_loss_val)
    save_pickle_obj(training_history, training_history_path)
