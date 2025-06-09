import os
import numpy as np
from configs import LOG_FOLDER, get_parse_command

datasets = ['MultiNLI','CelebA','CUB']
# datasets = ['PACS']
gpu_idx = 0  # could be None if you want to use cpu

# Suppose we already trained the models for seeds 0, 1, 2, 3, 4,
# then we can parse these traind models by choosing log_seeds = np.arange(5)
train_log_seeds = np.arange(5)

# The training algorithms we want to parse
# algos = ['ERM', 'reweight', 'groupDRO']
algos = ['ERM', ]
# algos = ['reweight', 'groupDRO']

# load checkpoint with a model selection rule
# best: take the model at the epoch of largest worst-group validation accuracy
# best_avg_acc: take the model at the epoch of largest average-group validation accuracy
# last: take the trained model at the last epoch
model_selects = ['best', ]

log_dir = LOG_FOLDER

for dataset in datasets:
    if dataset == 'MultiNLI':
        command = get_parse_command(dataset=dataset, algos=algos, model_selects=model_selects,
                                train_log_seeds=train_log_seeds, log_dir=log_dir, gpu_idx=gpu_idx,
                                parse_script='parse_features.py')
    else:
        command = get_parse_command(dataset=dataset, algos=algos, model_selects=model_selects,
                                train_log_seeds=train_log_seeds, log_dir=log_dir, gpu_idx=gpu_idx,
                                parse_script='parse_features_clip.py')


    print('Command:', command)
    os.system(command)
