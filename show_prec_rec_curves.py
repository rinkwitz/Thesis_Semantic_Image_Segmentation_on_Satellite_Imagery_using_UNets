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
    plt.plot(metrics['recs'][1:-1], metrics['precs'][1:-1], label=f'{label}', c=cmap(i))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.show()
