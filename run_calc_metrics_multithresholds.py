import os
import torch
from Utils.Helpers import *
from experiments import *

# Experiments params:
python_env_path = '$HOME/anaconda3/envs/thesis_env/bin/python3'
exps = [exp_0, exp_1, exp_2, exp_3, exp_4, exp_5, exp_6, exp_7, exp_8]

# Run experiments:
working_dir = Path('')
if not Path('experiments').exists():
    Path('experiments').mkdir()
for idx, exp in enumerate(exps):
    exp_dir = Path(f'experiments/experiment_{idx}')
    if (not exp_dir.exists()) or exp['write_access']:
        if not exp_dir.exists():
            exp_dir.mkdir()
        settings_path = exp_dir / 'settings.pkl'
        save_pickle_obj(exp, settings_path)

        # metrics:
        cmd = f'{python_env_path} {str((working_dir / "calc_metrics_multithresholds.py").absolute())} {str(exp_dir.absolute())}'
        print(f'{cmd}\n{exp}')
        os.system(cmd)
        print('\n\n')
