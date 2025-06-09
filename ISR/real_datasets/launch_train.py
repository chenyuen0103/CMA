import os
from itertools import product
from tqdm import tqdm
from configs import get_train_command
import pdb
import argparse
import numpy as np
# gpu_idx = 0,1,2,3  # could be None if you want to use cpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='MultiNLI')
    # parser.add_argument('--algos', type=list, default=['ERM'])
    # parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_best', type=bool, default=True)
    parser.add_argument('--save_last', type=bool, default=True)
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--seed_list', nargs = '+',type=int, default=[1,2,3,4])
    parser.add_argument('--model', type=str, default='bert')
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--hessian_align', action='store_true', default=False)
    parser.add_argument('--algo_suffix', type=str, default='', help='The suffix of log folder name')
    parser.add_argument('--scheduler', action='store_true', default=False)
    # parser.add_argument('--grad_alpha', type=float, default=10e-5)
    # parser.add_argument('--hess_beta', type=float, default=10e-5)
    # parser.add_argument('--grad_alpha_values', type=float, nargs='+', default=np.logspace(-6, -1, num=6))
    # parser.add_argument('--hess_beta_values', type=float, nargs='+', default=np.logspace(-6, -1, num=6))
    parser.add_argument('--grad_alpha_values', type=float, nargs='+', default=[1e-4])
    parser.add_argument('--hess_beta_values', type=float, nargs='+', default=[1e-4])

    algos = ['ERM']
    args = parser.parse_args()
    # Iterate over combinations of seeds, algorithms, grad_alpha, and hess_beta
    print(f"Loop over combinations of seeds {args.seed_list}, algorithms {algos}, grad_alpha {args.grad_alpha_values}, and hess_beta {args.hess_beta_values} ")

    if not args.hessian_align:
        # If hessian_align is False, then grad_alpha and hess_beta are not used, set them to default values
        args.grad_alpha_values = [1e-4]
        args.hess_beta_values = [1e-4]
        # if args.dataset == 'CUB':
        #     args.grad_alpha_values = [1e-5]
        #     args.hess_beta_values = [1e-5]

    for seed, algo, grad_alpha, hess_beta in tqdm(
            list(product(args.seed_list, algos, args.grad_alpha_values, args.hess_beta_values)), desc='Experiments'):
        # Generate the command with the specific grad_alpha and hess_beta
        command = get_train_command(dataset=args.dataset, algo=algo, gpu_idx=args.gpu_idx, model=args.model, seed=seed,
                                    save_best=args.save_best, save_last=args.save_last, resume=args.resume,
                                    hessian_align=args.hessian_align,
                                    algo_suffix=args.algo_suffix, scheduler=args.scheduler, grad_alpha=grad_alpha,
                                    hess_beta=hess_beta, learning_rate=args.learning_rate, batch_size=args.batch_size)

        print('Command:', command)
        os.system(command)
        resume_command = get_train_command(dataset=args.dataset, algo=algo, gpu_idx=args.gpu_idx, model=args.model, seed=seed,
                                    save_best=args.save_best, save_last=args.save_last, resume=True,
                                    hessian_align=args.hessian_align,
                                    algo_suffix=args.algo_suffix, scheduler=args.scheduler, grad_alpha=grad_alpha,
                                    hess_beta=hess_beta, learning_rate=args.learning_rate, batch_size=args.batch_size)
        # run the resume command for a few times with different gpu_idx
        if args.dataset == 'CelebA':
            resume_run = 20
        else:
            resume_run = 3
        for i in range(resume_run):
            os.system(resume_command)
            args.gpu_idx = (args.gpu_idx + 1) % 4

if __name__ == '__main__':
    main()