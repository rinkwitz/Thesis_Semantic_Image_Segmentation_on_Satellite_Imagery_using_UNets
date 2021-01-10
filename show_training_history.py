import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from Utils.Helpers import *

idx_experiment = 8
training_hist_path = Path(f'experiments/experiment_{idx_experiment}/training_history.pkl')
training_hist = load_pickle_obj(training_hist_path)
settings_path = Path(f'experiments/experiment_{idx_experiment}/settings.pkl')
settings = load_pickle_obj(settings_path)

print(settings['model_name'])
x = np.arange(1, len(training_hist['avg_loss_train']) + 1)
y_train = np.array(training_hist['avg_loss_train'])
y_val = np.array(training_hist['avg_loss_val'])
train, = plt.plot(x, y_train, label='train')
val, = plt.plot(x, y_val, label='validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(handles=[train, val], loc='upper right')
plt.show()
