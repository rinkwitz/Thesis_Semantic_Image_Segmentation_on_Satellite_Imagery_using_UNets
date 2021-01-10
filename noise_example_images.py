import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from Utils.Helpers import *

mean = np.load('tmp/mean_SN1_Buildings.npy')
std = np.load('tmp/std_SN1_Buildings.npy')

for i, noise in enumerate(np.linspace(0.0, 1.0, 6)):
    plt.subplot(f'23{i + 1}')
    im = np.load(Path('data/SN1_Buildings/train/3band/3band_AOI_1_RIO_img1520.npy'))
    # im = ((im / 255.) - mean) / std
    im = im / 255.
    im = im + noise * np.random.randn(im.shape[0], im.shape[1], im.shape[2])
    plt.xlabel(f'sigma = {noise:.1f}')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(im)
plt.show()
