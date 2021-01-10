import random
import sys
import torch.optim as optim
from Models.MobileNetV2_UNet import MobileNetV2_UNet
from Utils.Dataloader import Dataloader
from pathlib import Path
from Utils.Losses import *
from Utils.Helpers import *
from Models.UNet import UNet
from Models.AttentionUNet import AttentionUNet
import matplotlib.pyplot as plt


# Params:
min_lr = 1e-9
max_lr = 1.0
window_size_smoothing = 100
num_epochs = 3
momentum = .95
params = dict(num_categories=2, pretrained_on='Imagenet', num_epochs=100,
              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
dataloader_path = Path('tmp/Dataloader_SN1_Buildings.pkl')
model = UNet(in_channels=3, num_categories=params['num_categories'], filter_sizes=(32, 64, 128, 256, 512), deep_supervision=True)
running_experiment = False
initialize_weights = True
freeze_encoder = False
continue_training = False
if len(sys.argv) > 1:
    exp_dir = Path(sys.argv[1])
    running_experiment = True
    params, model, dataloader_path, learning_rate_dict, experiment_settings = get_experiment_objs(exp_dir)
    initialize_weights = experiment_settings['initialize_weights']
    freeze_encoder = experiment_settings['freeze_encoder']
    continue_training = experiment_settings['continue_training']
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
model_name = get_model_name(model)

# Exploring lr range:
lr_history = {
    'lr': [],
    'loss': [],
    'val_loss': [],
}
random.seed(42)
np.random.seed(42)
dataloader = Dataloader()
dataloader.load(dataloader_path)
num_batches = len(dataloader.train_objs)
num_batches_val = len(dataloader.val_objs)
num_lr_increases = num_batches * num_epochs
num_lr_increase = 0

if torch.cuda.is_available():
    model.cuda(params['device'])

optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=momentum)

start = time.time()
for num_epoch in range(num_epochs):
    dataloader.on_epoch_start(params)

    for num_batch in range(num_batches):
        model.train()
        X, Y = dataloader.get_mini_batch(num_batch, params, mode='train', weight=False, data_augmentation=True)
        lr = min_lr * 3. ** ((float(num_lr_increase) / num_lr_increases) * (np.log(max_lr / min_lr) / np.log(3.)))
        set_lr(optimizer, lr)
        optimizer.zero_grad()
        out = model(X)
        Y = crop_center_Y(out, Y)
        loss = 1 - DICE_coefficient(out, Y, zero_division_safe=True)

        loss.backward()
        optimizer.step()
        lr_history['lr'].append(lr)
        lr_history['loss'].append(loss.item())

        model.eval()
        num_batch_val = random.randint(0, num_batches_val - 1)
        X, Y = dataloader.get_mini_batch(num_batch_val, params, mode='val', weight=False, data_augmentation=False)
        out = model(X)
        Y = crop_center_Y(out, Y)
        val_loss = 1 - DICE_coefficient(out, Y, zero_division_safe=True)

        lr_history['val_loss'].append(val_loss.item())

        num_lr_increase += 1
        current_time = time.time()
        est_time_left = int((current_time - start) / (num_lr_increase / num_lr_increases) - (current_time - start))
        time_string = format_to_time_string(est_time_left)
        print(
            f'\r{num_lr_increase}/{num_lr_increases}:\tlr: {lr}\tloss: {loss.item():.4f}\tval_loss: {val_loss.item():.4f}\testimated time left: {time_string}',
            end='')

# save results:
if running_experiment:
    lr_history_path = exp_dir / f'{model_name}_lr_finder_history.pkl'
else:
    lr_history_path = Path(f'tmp/{model_name}_lr_finder_history.pkl')
save_pickle_obj(lr_history, lr_history_path)

# Visualization smoothed:
plt.figure(figsize=(20, 10), dpi=100)
for i, metric_name in enumerate(['loss', 'val_loss']):
    plt.subplot(int(f'12{i + 1}'))
    mvg_avg = [np.mean(np.array(lr_history[metric_name][i - window_size_smoothing:i])) for i in
               range(window_size_smoothing, len(lr_history['lr']))]
    plt.plot(lr_history['lr'][window_size_smoothing:], mvg_avg)
    plt.xscale('log')
    plt.xlabel('lr'.upper())
    plt.ylabel(metric_name.upper())
plt.savefig(Path(f'figs/{model_name}_lr_finder_visualization.png'))
if not running_experiment:
    plt.show()
