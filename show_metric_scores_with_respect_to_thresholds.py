import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from Utils.Helpers import *


# metrics_result_paths = [(Path('experiments/experiment_0/unet_32-512_wDeepSupervision_metrics_multithresholds_116.pkl'), 'U-Net'),
#                         (Path('experiments/experiment_1/attention_unet_32-512_wDeepSupervision_metrics_multithresholds_108.pkl'), 'Attention U-Net'),
#                         (Path('experiments/experiment_2/cbam_unet_32-512_wDeepSupervision_metrics_multithresholds_136.pkl'), 'CBAM U-Net'),
#                         (Path('experiments/experiment_3/residualattention_unet_32-512_wDeepSupervision_metrics_multithresholds_188.pkl'), 'Residual Attention U-Net'),
#                         (Path('experiments/experiment_4/scag_unet_32-512_wDeepSupervision_metrics_multithresholds_192.pkl'), 'scAG U-Net')]

metrics_result_paths = [(Path('experiments/experiment_5/densenet121_unet_wDeepSupervision_metrics_multithresholds_56.pkl'), 'DenseNet121 U-Net'),
                        (Path('experiments/experiment_6/mobilenetv2_unet_wDeepSupervision_metrics_multithresholds_28.pkl'), 'MobileNetV2 U-Net'),
                        (Path('experiments/experiment_7/resnet34_unet_wDeepSupervision_metrics_multithresholds_40.pkl'), 'ResNet34 U-Net'),
                        (Path('experiments/experiment_8/vgg11_unet_wDeepSupervision_metrics_multithresholds_52.pkl'), 'VGG11 U-Net')]

cmap = 'hsv'
cmap = plt.cm.get_cmap(cmap, len(metrics_result_paths) + 2)
metric_name = 'f1s'
# metric_name = 'ious'
assert metric_name in ['accs', 'ious', 'f1s', 'recs', 'precs', 'specs']

for i, t in enumerate(metrics_result_paths):
    path, label = t
    metrics = load_pickle_obj(path)
    metric_max = np.max(metrics[f'{metric_name}'])
    idx_max = np.argmax(metrics[f'{metric_name}'])
    threshold_max = metrics['thresholds'][idx_max]
    plt.plot(metrics['thresholds'][1:-1], metrics[f'{metric_name}'][1:-1], label=f'{label}', c=cmap(i))
    plt.plot([threshold_max, threshold_max], [metric_max, 0.0], '--', c=cmap(i))
    print(f'{label}: maximum {metric_name[:-1]}-score of {metric_max:.4f} at threshold: {threshold_max:.4f}')
plt.xlabel('Threshold')
plt.ylabel(f'{metric_name[:-1].upper()}-score')
plt.legend(loc='upper right')
plt.show()
