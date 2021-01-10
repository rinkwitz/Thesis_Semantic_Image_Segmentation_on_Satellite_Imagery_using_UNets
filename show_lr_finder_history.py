import numpy as np
import matplotlib.pyplot as plt
from Utils.Helpers import load_pickle_obj
from pathlib import Path

lr_history = load_pickle_obj(Path('experiments/experiment_2/cbam_unet_32-512_wDeepSupervision_lr_finder_history.pkl'))
model_name = 'cbam-unet'
window_size_smoothing = 400
max_lr = 1e-2
show_max_lr_in = [1]

plt.rcParams.update({'font.size': 16})
for i, metric_name in enumerate(['loss', 'val_loss']):
    plt.subplot(int(f'12{i + 1}'))
    mvg_avg = [np.mean(np.array(lr_history[metric_name][i - window_size_smoothing:i])) for i in range(window_size_smoothing, len(lr_history['lr']))]
    plt.plot(lr_history['lr'][window_size_smoothing:], mvg_avg)
    if i in show_max_lr_in:
        plt.plot([max_lr, max_lr], [np.min(mvg_avg), np.max(mvg_avg)], '--', c='r')
    plt.xscale('log')
    plt.xlabel('learning rate')
    plt.ylabel('training loss' if metric_name == 'loss' else 'validation loss')
plt.tight_layout()
plt.show()

latex_cmd = '\\begin{figure}[H]\n\t\\centering\n\t\\includegraphics[width=1.0\columnwidth]{../figs/'
latex_cmd += f'{model_name}_lr_range_test__window{window_size_smoothing}.png' + '}\n\t\\caption{TODO ... }'
latex_cmd += '\n\t\\label{fig_lr_range_' + f'{model_name}' +'}\n\\end{figure}'
print(latex_cmd)

"""
DenseNet121 U-Net           & 1e-2                      & 3e-2                      \\ \hline
MobileNetV2 U-Net           & 2e-2                      & 6e-2                      \\ \hline
ResNet34 U-Net              & 1e-2                      & 3e-2                      \\ \hline
VGG11 U-Net                 & 3e-3                      & 1e-2                      \\ \hline
"""
