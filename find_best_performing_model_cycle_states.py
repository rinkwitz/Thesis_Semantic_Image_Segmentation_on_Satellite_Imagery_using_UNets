import numpy as np
from pathlib import Path
from Utils.Helpers import *


for exp_path in [Path('experiments') / f'experiment_{i}' for i in range(9)]:
    if not ((exp_path / 'training_history.pkl').exists() and (exp_path / 'settings.pkl').exists()):
        continue
    settings = load_pickle_obj(exp_path / 'settings.pkl')
    training_history = load_pickle_obj(exp_path / 'training_history.pkl')
    avg_cycle_losses = []
    for num_cycle in range(len(training_history['avg_loss_val']) // 4):
        avg_cycle_losses.append(np.mean(training_history['avg_loss_val'][num_cycle * 4: (num_cycle + 1) * 4]))
    idx_min = np.argmin(avg_cycle_losses)
    print(f'{exp_path.name}\n{settings["model_name"]}\nbest_cycle_idx: {idx_min}\ncorresponds to modelstate at epoch: {4 * (idx_min + 1)} with avg_cycle_loss: {avg_cycle_losses[idx_min]:.4f}\n\n')
