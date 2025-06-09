import csv
import os
import random
import sys

import numpy as np
import torch


def check_args(args):
    if args.shift_type == 'confounder':
        assert args.confounder_names
        assert args.target_name
    elif args.shift_type.startswith('label_shift'):
        assert args.minority_fraction
        assert args.imbalance_ratio


class Logger(object):
    def __init__(self, fpath=None, mode='w'):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)


    def write_txt(self, msg):
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()




class CSVBatchLogger_ISR:
    def __init__(self, csv_path, n_groups, n_envs, mode='w'):
        columns = ['epoch','step']
        for idx in range(n_groups):
            columns.append(f'group_count:{idx}')
            columns.append(f'group_frac:{idx}')
            columns.append(f'acc-{idx}')
        for idx in range(n_envs):
            columns.append(f'env_count:{idx}')
            columns.append(f'env_frac:{idx}')
            columns.append(f'erm_loss_env:{idx}')
            columns.append(f'grad_penalty_env:{idx}')
            columns.append(f'hessian_penalty_env:{idx}')
        columns.append('grad_alpha')
        columns.append('hess_beta')
        columns.append('anneal_iters')
        columns.append('total_loss')
        columns.append('erm_loss')
        columns.append('grad_loss')
        columns.append('hessian_loss')
        columns.append('avg_acc')
        columns.append('worst_group')
        columns.append('worst_acc')


        self.path = csv_path
        self.file = open(csv_path, mode)
        self.columns = columns
        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        if mode == 'w':
            self.writer.writeheader()

    def log(self, epoch, batch, stats_dict):
        stats_dict['epoch'] = epoch
        stats_dict['batch'] = batch
        self.writer.writerow(stats_dict)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy_fn(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    temp = target.view(1, -1).expand_as(pred)
    temp = temp.cuda()
    correct = pred.eq(temp)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def set_seed(args):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def log_args(args, logger):
    for argname, argval in vars(args).items():
        logger.write(f'{argname.replace("_", " ").capitalize()}: {argval}\n')
    logger.write('\n')
