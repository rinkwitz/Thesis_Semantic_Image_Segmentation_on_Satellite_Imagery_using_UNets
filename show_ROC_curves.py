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
for i, t in enumerate(metrics_result_paths):
    path, label = t
    metrics = load_pickle_obj(path)
    fpr1 = 1.0
    rec1 = 1.0
    auc = 0.0
    for spec2, rec2 in zip(metrics['specs'], metrics['recs']):
        fpr2 = 1 - spec2
        plt.plot([fpr1, fpr2], [rec1, rec1], label=None, c=cmap(i))
        plt.plot([fpr2, fpr2], [rec1, rec2], label=None, c=cmap(i))
        auc += np.abs(fpr2 - fpr1) * np.minimum(rec1, rec2)
        fpr1, rec1 = fpr2, rec2
    plt.plot([], [], label=f'{label} (AUC={auc:.4f})', c=cmap(i))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.03])
plt.xlabel('False positive rate')
plt.ylabel('Recall')
plt.legend(loc='lower right')
plt.show()
