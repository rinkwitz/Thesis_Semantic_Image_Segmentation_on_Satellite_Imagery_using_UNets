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

latex = '\n\nmodel\\_name & acc & iou & f1 & rec & prec & spec \\\\ \\hline'
for i, t in enumerate(metrics_result_paths):
    path, label = t
    metrics = load_pickle_obj(path)
    idx_max = np.argmax(metrics['f1s'])
    threshold = metrics['thresholds'][idx_max]
    acc = metrics['accs'][idx_max]
    iou = metrics['ious'][idx_max]
    f1 = metrics['f1s'][idx_max]
    rec = metrics['recs'][idx_max]
    prec = metrics['precs'][idx_max]
    spec = metrics['specs'][idx_max]
    print(f'{label} thresholded at {threshold:.4f}:\nacc:\t{acc:.4f}\niou:\t{iou:.4f}\nf1:\t\t{f1:.4f}'
          f'\nrec:\t{rec:.4f}\nprec:\t{prec:.4f}\nspec:\t{acc:.4f}\n\n')
    latex += f'\n{label} & {acc:.4f} & {iou:.4f} & {f1:.4f} & {rec:.4f} & {prec:.4f} & {spec:.4f} \\\\ \\hline'
print(latex)
