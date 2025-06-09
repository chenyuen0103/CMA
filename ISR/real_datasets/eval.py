import argparse
import os
import pickle
import warnings
from itertools import product
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import shutil
# from memory_profiler import profile

from configs import DATA_FOLDER
from configs import LOG_FOLDER
from hisr import HISRClassifier, check_clf
from isr import ISRClassifier
from utils.eval_utils import extract_data, save_df, measure_group_accs, load_data, group2env, env2group
from utils.train_utils import  CSVBatchLogger_ISR


warnings.filterwarnings('ignore')  # filter out Pandas append warnings


def eval_ISR(args, train_data=None, val_data=None, test_data=None, log_dir=None):
    if (train_data is None) or (val_data is None) or (test_data is None) or (log_dir is None):
        train_data, val_data, test_data, log_dir = load_data(args)
    train_gs = train_data['group']
    n_train = len(train_gs)
    groups, counts = np.unique(train_data['group'], return_counts=True, axis=0)
    n_groups = len(groups)
    n_classes = len(np.unique(train_data['label']))
    # we do this because the original group is defined by (class * attribute)
    n_spu_attr = n_groups // n_classes
    assert n_spu_attr >= 2
    assert n_groups % n_classes == 0

    zs, ys, gs, preds = extract_data(train_data)


    test_zs, test_ys, test_gs, test_preds = extract_data(
        test_data, )
    val_zs, val_ys, val_gs, val_preds = extract_data(
        val_data, )

    if args.algo == 'ERM' or args.no_reweight:
        # no_reweight: do not use reweightning in the ISR classifier even if the args.algo is 'reweight' or 'groupDRO'
        sample_weight = None
    else:
        sample_weight = np.ones(n_train)
        for group, count in zip(groups, counts):
            sample_weight[train_gs == group] = n_train / n_groups / count
        if args.verbose:
            print('Computed non-uniform sample weight')

    grad_alpha_formatted = "{:.1e}".format(args.alpha).replace('.0e', 'e')
    hess_beta_formatted = "{:.1e}".format(args.beta).replace('.0e', 'e')


    df = pd.DataFrame(
        columns=['dataset', 'algo', 'seed', 'ckpt', 'split', 'method', 'clf_type', 'C', 'pca_dim', 'd_spu', 'ISR_class',
                 'ISR_scale', 'env_label_ratio','num_iter','gradient_alpha','hessian_beta'] +
                ['ema', 'lambda', 'penalty_anneal_iters'] if args.hessian_approx_method == 'fishr' else [] +
                ['mmd_gamma'] if args.hessian_approx_method == 'coral' else [] +
                [f'acc-{g}' for g in groups] + ['worst_group', 'avg_acc', 'worst_acc', ])

    base_row = {'dataset': args.dataset, 'algo': args.algo,
                'seed': args.seed, 'ckpt': args.model_select,
                'num_iter': args.max_iter,
                'gradient_alpha': args.alpha,
                'hessian_beta': args.beta,
                'ema' : args.ema,
                'lambda': args.lam,
                'penalty_anneal_iters': args.penalty_anneal_iters,
                'mmd_gamma': args.mmd_gamma,
        }
    if args.hessian_approx_method == 'fishr':
        save_path = os.path.join(args.save_dir,
                                 f"{args.dataset}_results{args.file_suffix}_s{args.seed}_fishr.csv")
    elif args.hessian_approx_method == 'coral':
        save_path = os.path.join(args.save_dir,
                                 f"{args.dataset}_results{args.file_suffix}_s{args.seed}_coral.csv")
    else:
        save_path = os.path.join(args.save_dir,
                                 f"{args.dataset}_results{args.file_suffix}_s{args.seed}{f'_hessian_{args.hessian_approx_method}' if args.hessian_approx_method else '_ISR'}.csv")


    df_check = pd.read_csv(save_path) if os.path.exists(save_path) else None


    # if df_check is not None and 'ema' or 'lambda' or 'penalty_anneal_iters' not in df_check.columns:
    #     df_check['ema'] = args.ema
    #     df_check['lambda'] = args.lam
    #     df_check['penalty_anneal_iters'] = args.penalty_anneal_iters

    df_check_base = df_check[['dataset', 'algo', 'seed', 'ckpt','num_iter','gradient_alpha','hessian_beta']].drop_duplicates() if df_check is not None else None


    if df_check is not None and base_row in df_check_base.to_dict('records'):
        print(f"Already evaluated {base_row}")
        return df_check

    # Need to convert group labels to env labels (i.e., spurious-attribute labels)
    es, val_es, test_es = group2env(gs, n_spu_attr), group2env(val_gs, n_spu_attr), group2env(test_gs, n_spu_attr)

    # eval_groups = np.array([0] + list(range(n_groups)))
    if args.hessian_approx_method in ['Fishr','fishr']:
        args.hessian_approx_method = 'fishr'
        method = f'Fishr-{args.ISR_version.capitalize()}'
    elif args.hessian_approx_method is not None:
        method = f'HISR-{args.hessian_approx_method}-{args.ISR_version.capitalize()}'
    else:
        method = f'ISR-{args.ISR_version.capitalize()}'
    if args.no_reweight and (not args.use_orig_clf) and args.algo != 'ERM':
        method += '_noRW'
    if args.use_orig_clf:
        ckpt = pickle.load(open(log_dir + f'/{args.model_select}_clf.p', 'rb'))
        orig_clf = check_clf(ckpt, n_classes=n_classes)
        # Record original val accuracy:
        for (split, eval_zs, eval_ys, eval_gs) in [('val', val_zs, val_ys, val_gs),
                                                   ('test', test_zs, test_ys, test_gs)]:
            eval_group_accs, eval_worst_acc, eval_worst_group = measure_group_accs(orig_clf, eval_zs, eval_ys, eval_gs,
                                                                                   include_avg_acc=True)
            row = {**base_row, 'split': split, 'method': 'orig', **eval_group_accs, 'clf_type': 'orig',
                   'worst_acc': eval_worst_acc, 'worst_group': eval_worst_group}
            # df = df.append(row, ignore_index=True)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        args.n_components = -1
        given_clf = orig_clf
        clf_type = 'orig'
    else:
        given_clf = None
        clf_type = 'logistic'

    if args.env_label_ratio < 1:
        rng = np.random.default_rng()
        # take a subset of training data
        idxes = rng.choice(len(zs), size=int(
            len(zs) * args.env_label_ratio), replace=False)
        zs, ys, gs, es = zs[idxes], ys[idxes], gs[idxes], es[idxes]

    np.random.seed(args.seed)
    # Start ISR
    ISR_classes = np.arange(
        n_classes) if args.ISR_class is None else [args.ISR_class]

    clf_kwargs = dict(C=args.C, max_iter=args.max_iter,
                      random_state=args.seed,
                      gradient_hyperparam = args.alpha,
                      hessian_hyperparam = args.beta,
                      ema = args.ema,
                      lam = args.lam,
                      penalty_anneal_iters = args.penalty_anneal_iters,
                      )
    if args.ISR_version == 'mean': args.d_spu = n_spu_attr - 1
    isr_clf = HISRClassifier(version=args.ISR_version, hessian_approx_method = args.hessian_approx_method, pca_dim=args.n_components, d_spu=args.d_spu,
                        clf_type='LogisticRegression',
                             clf_kwargs=clf_kwargs, )

    isr_clf.fit_data(zs, ys, es, n_classes=n_classes, n_envs=n_spu_attr)
    if type(args.alpha) != list:
        alphas = [args.alpha]
    else:
        alphas = args.alpha
    if type(args.beta) != list:
        betas = [args.beta]
    else:
        betas = args.beta

    for ISR_class, ISR_scale, alpha, beta in tqdm(list(product(ISR_classes, args.ISR_scales, alphas, betas)), desc='ISR iter', leave=True):
        train_csv_logger, val_csv_logger, test_csv_logger = None, None, None
        if args.hessian_approx_method in ['exact']:
            progress_dir = os.path.join(args.progress_save_dir, args.dataset, f's{args.seed}',
                                        f"ISRclass_{ISR_class}_grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}_anneal_{args.penalty_anneal_iters}")
            if args.hessian_approx_method in ['hgp', 'hutchinson']:
                progress_dir = os.path.join(args.progress_save_dir, args.dataset, f's{args.seed}', args.hessian_approx_method,
                                            f"ISRclass_{ISR_class}_grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}")
        elif args.hessian_approx_method == 'fishr':
            progress_dir = os.path.join(args.progress_save_dir, args.dataset, f's{args.seed}',
                                        f"ISRclass_{ISR_class}_ema_{args.ema}_lam_{args.lam}_anneal_{args.penalty_anneal_iters}")
        elif args.hessian_approx_method == 'coral':
            progress_dir = os.path.join(args.progress_save_dir, args.dataset, f's{args.seed}',
                                        f"ISRclass_{ISR_class}_mmd_{args.mmd_gamma}")
        elif args.hessian_approx_method == 'hgp':
            progress_dir = os.path.join(args.progress_save_dir, args.dataset, f's{args.seed}',
                                        f"ISRclass_{ISR_class}_grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}")

        # Check if the directory exists
        if os.path.exists(progress_dir):
            # Remove the directory and its contents
            shutil.rmtree(progress_dir)

        # Create the directory, including any necessary parent directories
        os.makedirs(progress_dir)

        train_csv_logger = CSVBatchLogger_ISR(os.path.join(progress_dir, 'train.csv'), n_groups, n_spu_attr)
        val_csv_logger = CSVBatchLogger_ISR(os.path.join(progress_dir, 'val.csv'), n_groups, n_spu_attr)
        test_csv_logger = CSVBatchLogger_ISR(os.path.join(progress_dir, 'test.csv'), n_groups, n_spu_attr)

        isr_clf.set_params(chosen_class=ISR_class, spu_scale=ISR_scale)
        if args.ISR_version == 'mean':
            isr_clf.fit_isr_mean(chosen_class=ISR_class, )
        elif args.ISR_version == 'cov':
            isr_clf.fit_isr_cov(chosen_class=ISR_class, )
        else:
            raise ValueError('Unknown ISR version')
        val_es = group2env(val_gs, n_spu_attr)
        test_es = group2env(test_gs, n_spu_attr)
        isr_clf.fit_clf(zs, ys, es, args, given_clf=given_clf, sample_weight=sample_weight, hessian_approx= args.hessian_approx_method,
                        alpha=args.alpha, beta=args.beta, train_csv_logger = train_csv_logger, val_csv_logger = val_csv_logger, test_csv_logger = test_csv_logger,
                        val_zs = val_zs, val_ys = val_ys, val_es = val_es, val_gs = val_gs, test_zs = test_zs, test_ys = test_ys, test_es = test_es, test_gs = test_gs)
        for (split, eval_zs, eval_ys, eval_gs) in [('val', val_zs, val_ys, val_gs),
                                                   ('test', test_zs, test_ys, test_gs)]:
            group_accs, worst_acc, worst_group = measure_group_accs(
                isr_clf, eval_zs, eval_ys, eval_gs, include_avg_acc=True)
            row = {**base_row, 'split': split, 'method': method, 'clf_type': clf_type, 'ISR_class': ISR_class,
                   'ISR_scale': ISR_scale, 'd_spu': args.d_spu, **group_accs, 'worst_group': worst_group,
                   'worst_acc': worst_acc, 'env_label_ratio': args.env_label_ratio}
            if not args.use_orig_clf:
                row.update({'C': args.C, 'pca_dim': args.n_components, })
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            # if not args.no_save:
            #     Path(args.save_dir).mkdir(parents=True,
            #                               exist_ok=True)
            #     save_path = os.path.join(args.save_dir,
            #                              f"{args.dataset}_results{args.file_suffix}_s{args.seed}{f'_hessian_{args.hessian_approx_method}' if args.hessian_approx_method else '_ISR'}.csv")
            #     save_df(df, save_path, subset=None, verbose=args.verbose)
            #     print(f"Saved to {args.save_dir} as {save_path}")
            if train_csv_logger is not None:
                train_csv_logger.close()
                val_csv_logger.close()
                test_csv_logger.close()

    if args.verbose:
        print('Evaluation result')
        print(df)
    if not args.no_save:
        Path(args.save_dir).mkdir(parents=True,
                                  exist_ok=True)  # make dir if not exists
        save_df(df, save_path, subset=None, verbose=args.verbose)
        print(f"Saved to {args.save_dir} as {save_path}")


    return df

def parse_args(args: list = None, specs: dict = None):
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--root_dir', type=str,
                           default='inv-feature-bert-trained/logs')
    argparser.add_argument('--algo', type=str, default='ERM',
                           choices=['ERM', 'groupDRO', 'reweight'])
    argparser.add_argument(
        '--dataset', type=str, default='CelebA', choices=['CelebA', 'MultiNLI', 'CUB'])
    argparser.add_argument('--model_select', type=str,
                           default='best', choices=['best', 'best_avg_acc', 'last','CLIP_init', 'init'])

    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--n_components', type=int, default=100)
    argparser.add_argument('--C', type=float, default=1)
    argparser.add_argument('--ISR_version', type=str, default='mean', choices=['mean', 'cov'])
    argparser.add_argument('--ISR_class', type=int, default=None,
                           help='None means enumerating over all classes.')
    argparser.add_argument('--ISR_scales', type=float,
                           nargs='+', default=[0])
    argparser.add_argument('--d_spu', type=int, default=-1)
    argparser.add_argument('--save_dir', type=str, default='./logs/ISR_Hessian_results_new')
    argparser.add_argument('--progress_save_dir', type=str, default='./logs/ISR_training_progress')
    argparser.add_argument('--no_save', default=False, action='store_true')
    argparser.add_argument('--verbose', default=False, action='store_true')

    argparser.add_argument('--use_orig_clf', default=False,
                           action='store_true', help='Original Classifier only')
    argparser.add_argument('--env_label_ratio', default=1,
                           type=float, help='ratio of env label')
    argparser.add_argument('--feature_file_prefix', default='',
                           type=str, help='Prefix of the feature files to load')
    argparser.add_argument('--max_iter', default=1000, type=int,
                           help='Max iterations for the logistic solver')
    argparser.add_argument('--file_suffix', default='', type=str, )
    argparser.add_argument('--no_reweight', default=False, action='store_true',
                           help='No reweighting for ISR classifier on reweight/groupDRO features')
    argparser.add_argument('--hessian_approx_method', default = 'fishr', type=str, )
    argparser.add_argument('--alpha', default=10000, type=float, help='gradient hyperparameter')
    argparser.add_argument('--beta', default=100, type=float, help='hessian hyperparameter')
    argparser.add_argument('--cuda', default=1, type=int, help='cuda device')
    argparser.add_argument('--ema', default=0.99, type=float, help='fishr ema')
    argparser.add_argument('--lam', default=10000, type =int, help='fishr penalty weight')
    argparser.add_argument('--penalty_anneal_iters', default = 1200, type=int,  help='fishr penalty anneal iters')
    argparser.add_argument('--mmd_gamma', default = 1, type=float, help='mmd_gamma for CORAL')
    config = argparser.parse_args(args=args)

    print("Specs:", specs)
    print("Config:", config.__dict__)
    if specs is not None:
        config.__dict__.update(specs)
    return config


def run_fishr(args, penalty_anneal_iters_list, fishr_top5 = None):
    seed_list = [0]
    # args.ISR_class = None
    # randomly choose 50 triples of lambda, penalty_anneal_iters, ema from the following ranges
    lambda_list = 10 ** np.linspace(1, 4, 4)
    # penalty_anneal_iters_list = np.linspace(0,5000,6)
    ema_list = np.linspace(0.9, 0.99, 5)

    def sample_hyperparams(seed):
        np.random.seed(seed)
        lam = np.random.choice(lambda_list)
        penalty_anneal_iters = np.random.choice(penalty_anneal_iters_list)
        ema = np.random.choice(ema_list)
        return lam, penalty_anneal_iters, ema

    # for i in range(50):
        # lam, penalty_anneal_iters, ema = sample_hyperparams(i)
    if fishr_top5:
        params_list = fishr_top5
        seed_list = [0,1,2,3,4]
    else:
        params_list = product(ema_list, lambda_list, penalty_anneal_iters_list)
    for seed in seed_list:
        for ema, lam, penalty_anneal_iters in params_list:
            lam = round(lam, 4)
            penalty_anneal_iters = round(penalty_anneal_iters, 1)
            ema = round(ema, 4)
            args.seed = seed
            args.lam = lam
            args.penalty_anneal_iters = penalty_anneal_iters
            args.ema = ema
            result_file = os.path.join(args.save_dir, f"{args.dataset}_results{args.file_suffix}_s{args.seed}_fishr.csv")
            if os.path.exists(result_file):
                existing_df = pd.read_csv(result_file)
                df_current = existing_df[(existing_df['lambda'] == lam) & (existing_df['penalty_anneal_iters'] == penalty_anneal_iters) & (existing_df['ema'] == ema)]
                if len(df_current) > 0:
                    print(f"Already evaluated seed: {seed}, lambda: {lam}, anneal iters: {penalty_anneal_iters}, ema: {ema}")
                    args.ISR_class = 1
                    if len(df_current) > 4 and args.dataset == 'CUB' or 'CelebA':
                        continue
                    elif len(df_current) > 6 and args.dataset == 'MultiNLI':
                        continue
            print(f"Running seed: {seed}, lambda: {lam}, anneal iters: {penalty_anneal_iters}, ema: {ema}")
            eval_ISR(args)


def run_coral(args, coral_top5 = None):
    if coral_top5:
        params_list = np.round(coral_top5, decimals=8)
        seed_list = [0,1,2,3,4]
    else:
        params_list = np.round(10 ** np.linspace(-1, 1, 50),decimals=8)
        seed_list = [0]

    for seed in seed_list:
        for mmd_gamma in params_list:
            args.seed = seed
            args.mmd_gamma = mmd_gamma
            result_file = os.path.join(args.save_dir, f"{args.dataset}_results{args.file_suffix}_s{args.seed}_coral.csv")
            if os.path.exists(result_file):
                existing_df = pd.read_csv(result_file)
                df_current = existing_df[(existing_df['mmd_gamma'] == mmd_gamma)]
                if len(df_current) > 0:
                    print(f"Already evaluated seed: {seed}, mmd_gamma: {mmd_gamma}")
                    args.ISR_class = 1
            print(f"Running seed: {seed}, mmd_gamma: {mmd_gamma}")
            eval_ISR(args)

def run_hgp(args, hgp_top5 = None):
    if hgp_top5:
        params_list = np.round(hgp_top5, decimals=8)
        seed_list = [0,1,2,3,4]
    else:
        alpha_list = np.round(10 ** np.linspace(-1, 3, 5),decimals=8)
        beta_list = np.round(10 ** np.linspace(-3, 5, 9),decimals=8)
        if args.dataset == 'CUB':
            penalty_anneal_iters_list = np.linspace(0, 2800, 5)
        elif args.dataset == 'CelebA':
            penalty_anneal_iters_list = np.linspace(0, 16000, 5)
        elif args.dataset == 'MultiNLI':
            penalty_anneal_iters_list = np.linspace(0, 1200, 5)

        params_list = product(alpha_list, beta_list, penalty_anneal_iters_list)
        seed_list = [0]

    for seed in seed_list:
        for grad_alpha, hess_beta, anneal_iters in params_list:
            args.seed = seed
            args.alpha = grad_alpha
            args.beta = hess_beta
            args.penalty_anneal_iters = anneal_iters
            result_file = os.path.join(args.save_dir, f"{args.dataset}_results{args.file_suffix}_s{args.seed}_hessian_hgp.csv")
            if os.path.exists(result_file):
                existing_df = pd.read_csv(result_file)
                df_current = existing_df[(existing_df['gradient_alpha'] == grad_alpha) &
                                         (existing_df['hessian_beta'] == hess_beta) &
                                            (existing_df['penalty_anneal_iters'] == anneal_iters)
                                         ]
                if len(df_current) > 0:
                    print(f"Already evaluated seed: {seed}, grad_alpha: {grad_alpha}, hess_beta: {hess_beta}, anneal_iters: {anneal_iters}")
                    continue
            print(f"Running seed: {seed}, grad_alpha: {grad_alpha}, hess_beta: {hess_beta}, anneal_iters: {anneal_iters} for HGP")
            eval_ISR(args)


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)



    # alpha_beta_list = list(product([0],10 ** np.linspace(-1, 3, 5))) + list(product(10 ** np.linspace(-1, 3, 5), [0])) + [(0,0)]
    seed_list = [0]
    fishr_top5 = None
    coral_top5 = None
    hgp_top5 = None
    # Define specific pairs of alpha and beta values
    if args.dataset == 'CUB':
        alpha_list = np.round([2000, 5000, 10000] + list(10 ** np.linspace(-1, 3, 5)[::-1]), decimals=8)
        beta_list = np.round([2000, 5000, 10000] + list(10 ** np.linspace(-1, 3, 5)[::-1]), decimals=8)
        args.max_iter = 300
        args.root_dir = './inv-feature-ViT-B/logs'
        args.model_select = 'init'
        penalty_anneal_iters_list = np.linspace(0, 2800, 5)

        alpha_beta_anneal = product(alpha_list, beta_list, penalty_anneal_iters_list)
        # Best ISR_class
        args.ISR_class = 0

        alpha_beta_anneal = [
            (0,0,0),
            (10, 1000, 2100),
            (100.0, 100.0, 0.0),
            (10000,100.0, 1200),
            (10.0, 100.0, 0.0),
            (1.0, 100.0, 0.0)
        ]
        # alpha_beta_anneal = [(10,1000,2100)]
        fishr_top5 = [
            (0.945, 100.0, 700.0),
            (0.9675, 10.0, 0.0),
            (0.9225, 100.0, 700.0),
            (0.9225, 10000.0, 1400.0),
            (0.945, 100.0, 2800.0)
        ]
        coral_top5 = [0.23299518,
                    10.0,
                    0.25595479,
                    0.95409548,
                    0.21209509,
                    ]
        hgp_top5 = [ (0.1,0.01,1400.0),
                    (0.1,0.01,0.0),
                    (1.0,0.001,700.0),
                    (1.0,0.001,2100.0),
                    (0.1,0.01,700.0)
                                    ]

        num_rows = 2
    if args.dataset == 'CelebA':
        alpha_list = np.round([2000, 5000, 10000] + list(10 ** np.linspace(-1, 3, 5)[::-1]), decimals=8)
        beta_list = np.round([2000, 5000, 10000] + list(10 ** np.linspace(-1, 3, 5)[::-1]), decimals=8)
        args.max_iter = 50

        # args.save_dir = './logs/ISR_Hessian_results_new'
        args.root_dir = './inv-feature-ViT-B/logs'
        args.model_select = 'init'
        penalty_anneal_iters_list = np.linspace(0, 16000, 5)
        alpha_beta_anneal = product(alpha_list, beta_list, penalty_anneal_iters_list)
        alpha_beta_anneal = [
            (0.0, 0.0, 0.0),
            (5000.0, 100.0, 4000.0),
            (2000.0, 100.0, 6000.0),
            (2000.0, 100.0, 6000.0),
            (5000.0, 100.0, 6000.0),
            (1000.0, 100.0, 8000.0),\
        ]

        num_rows = 2
        # Best ISR_class
        args.ISR_class = 0
        fishr_top5 = [
            (0.945, 10.0, 8000.0),
            (0.9225, 10.0, 12000.0),
            (0.9, 10.0, 4000.0),
            (0.945, 10000.0, 8000.0),
            (0.9, 100.0, 2000.0)
        ]
        coral_top5 = [0.13257114,
                        0.14563485,
                        0.15998587,
                        0.33932218,
                        0.44984327]

        hgp_top5 = [(0.1,0.001,0.0),
                    (0.1,0.001,4000.0),
                    (0.1,0.001,8000.0),
                    (0.1,0.001,12000.0),
                    (0.1,0.001,16000.0)
                    ]

    if args.dataset == 'MultiNLI':
        alpha_list = np.round([2000, 5000, 10000] + list(10 ** np.linspace(-1, 3, 5) + [0]), decimals=8)
        beta_list = np.round([0] + [2000, 5000, 10000] + list(10 ** np.linspace(-1, 3, 5)), decimals=8)[::-1]

        args.max_iter = 3
        penalty_anneal_iters_list = np.linspace(0, 1200, 5)[::-1]
        alpha_beta_anneal = product(alpha_list, beta_list, penalty_anneal_iters_list)


        # Best ISR_class
        args.ISR_class = 2
        alpha_beta_anneal = [
            (0.0, 0.0, 0.0),
            (5000,1,0),
            (2000.0, 100.0, 0.0),
            (1.0, 5000.0, 900.0),
            (100.0, 0.1, 0.0),
            (100, 5000, 300)
        ]
        fishr_top5 = [
            (0.9675, 10000.0, 300.0),
            (0.9675, 10000.0, 600.0),
            (0.945, 1000.0, 900.0),
            (0.945, 10000.0, 900.0),
            (0.99, 10000.0, 600.0)
        ]
        coral_top5 =[5.68986603,
                    0.44984327,
                    1.38949549,
                    0.10985411,
                    0.33932218,
                    ]

        hgp_top5 = [(1.0,0.001,0.0),
                    (0.1,0.01,600.0),
                    (0.1,0.001,600.0),
                    (0.1,0.001,300.0),
                    (0.1,0.001,900.0)]
    if args.hessian_approx_method == 'fishr':
        run_fishr(args, penalty_anneal_iters_list, fishr_top5 = fishr_top5)
    elif args.hessian_approx_method == 'coral':
        run_coral(args, coral_top5 = coral_top5)
    elif args.hessian_approx_method.lower() == 'hgp':
        run_hgp(args, hgp_top5 = hgp_top5)
    else:
        for seed in seed_list:
            for alpha, beta, anneal_iters in alpha_beta_anneal:
                args.alpha = round(alpha, 4)
                args.beta = round(beta, 4)
                args.seed = seed
                args.penalty_anneal_iters = round(anneal_iters, 1)
                result_file = os.path.join(args.save_dir,
                                           f"{args.dataset}_results{args.file_suffix}_s{args.seed}_hessian_exact.csv")
                if os.path.exists(result_file):
                    existing_df = pd.read_csv(result_file)
                    # Replace non-numeric values with NaN or a specific number
                    existing_df['gradient_alpha'] = pd.to_numeric(existing_df['gradient_alpha'], errors='coerce')
                    existing_df['hessian_beta'] = pd.to_numeric(existing_df['hessian_beta'], errors='coerce')
                    existing_df['penalty_anneal_iters'] = pd.to_numeric(existing_df['penalty_anneal_iters'], errors='coerce')
                    df_current = existing_df[
                        np.isclose(existing_df['gradient_alpha'], alpha, atol=1e-8) &
                        (existing_df['penalty_anneal_iters'] == anneal_iters) &
                        np.isclose(existing_df['hessian_beta'], beta, atol=1e-8) &
                        (existing_df['algo'] == args.algo)
                        ]
                    if len(df_current) >= num_rows:
                        print(
                            f"Already evaluated seed: {seed}, alpha: {alpha}, anneal iters: {anneal_iters}, beta: {beta}")
                        continue

                print(f"Running alpha = {alpha}, beta = {beta}, anneal_iters = {anneal_iters}, seed = {seed}")
                eval_ISR(args)




