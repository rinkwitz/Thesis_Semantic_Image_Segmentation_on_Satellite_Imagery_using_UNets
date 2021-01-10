import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from Utils.Helpers import *

# metrics_result_paths = [(Path('experiments/experiment_0/unet_32-512_wDeepSupervision_robustness_multi_noiselevel_116.pkl'), 'U-Net'),
#                         (Path('experiments/experiment_1/attention_unet_32-512_wDeepSupervision_robustness_multi_noiselevel_108.pkl'), 'Attention U-Net'),
#                         (Path('experiments/experiment_2/cbam_unet_32-512_wDeepSupervision_robustness_multi_noiselevel_136.pkl'), 'CBAM U-Net'),
#                         (Path('experiments/experiment_3/residualattention_unet_32-512_wDeepSupervision_robustness_multi_noiselevel_188.pkl'), 'Residual Attention U-Net'),
#                         (Path('experiments/experiment_4/scag_unet_32-512_wDeepSupervision_robustness_multi_noiselevel_192.pkl'), 'scAG U-Net')]

metrics_result_paths = [(Path('experiments/experiment_5/densenet121_unet_wDeepSupervision_robustness_multi_noiselevel_56.pkl'), 'DenseNet121 U-Net'),
                        (Path('experiments/experiment_6/mobilenetv2_unet_wDeepSupervision_robustness_multi_noiselevel_28.pkl'), 'MobileNetV2 U-Net'),
                        (Path('experiments/experiment_7/resnet34_unet_wDeepSupervision_robustness_multi_noiselevel_40.pkl'), 'ResNet34 U-Net'),
                        (Path('experiments/experiment_8/vgg11_unet_wDeepSupervision_robustness_multi_noiselevel_52.pkl'), 'VGG11 U-Net')]

def label_from_metricname(name):
    if name == 'accs':
        return 'Accuracy'
    elif name == 'ious':
        return 'IOU'
    elif name == 'f1s':
        return 'F1-score'
    elif name == 'recs':
        return 'Recall'
    elif name == 'precs':
        return 'Precision'
    elif name == 'specs':
        return 'Specificity'

cmap = 'hsv'
cmap = plt.cm.get_cmap(cmap, len(metrics_result_paths) + 2)
# plt.rcParams.update({'font.size': 16})
for idx_plot, metric_name in enumerate(['accs', 'ious', 'f1s', 'recs', 'precs', 'specs']):
    plt.subplot(f'24{idx_plot + 1 if idx_plot < 3 else idx_plot + 2}')
    ymax = -1.0
    for i, t in enumerate(metrics_result_paths):
        path, label = t
        metrics = load_pickle_obj(path)
        plt.plot(metrics['noise_levels'], metrics[f'{metric_name}'], label=f'{label}', c=cmap(i))
        ymax = np.maximum(np.max(metrics[f'{metric_name}']), ymax)
    plt.xlim([0.0, np.max(metrics['noise_levels'])])
    plt.ylim([0.0, np.minimum(ymax * 1.03, 1.0)])
    plt.xlabel('sigma')
    plt.ylabel(label_from_metricname(metric_name))
plt.legend(bbox_to_anchor=(1.9, .9), loc='lower right')
plt.show()
