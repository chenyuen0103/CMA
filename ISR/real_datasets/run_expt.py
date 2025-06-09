import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
import torchvision
import open_clip
import timm
import configs
from configs.model_config import model_attributes
from data import dataset_attributes, shift_types, prepare_data, log_data
# from train_hessian import train
from train import train
from utils.train_utils import set_seed, Logger, CSVBatchLogger, log_args
import pdb

def main():
    parser = argparse.ArgumentParser()

    # Settings
    parser.add_argument('-d', '--dataset', choices=dataset_attributes.keys(), required=True)
    parser.add_argument('-s', '--shift_type', choices=shift_types, required=True)
    # Confounders
    parser.add_argument('-t', '--target_name')
    parser.add_argument('-c', '--confounder_names', nargs='+')
    # Resume?
    parser.add_argument('--resume', default=False, action='store_true')
    # Label shifts
    parser.add_argument('--minority_fraction', type=float)
    parser.add_argument('--imbalance_ratio', type=float)
    # Data
    parser.add_argument('--fraction', type=float, default=1.0)
    parser.add_argument('--root_dir', default=None)
    parser.add_argument('--reweight_groups', action='store_true', default=False)
    parser.add_argument('--augment_data', action='store_true', default=False)
    parser.add_argument('--val_fraction', type=float, default=0.1)
    # Objective
    parser.add_argument('--robust', default=False, action='store_true')
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--generalization_adjustment', default="0.0")
    parser.add_argument('--automatic_adjustment', default=False, action='store_true')
    parser.add_argument('--robust_step_size', default=0.01, type=float)
    parser.add_argument('--use_normalized_loss', default=False, action='store_true')
    parser.add_argument('--btl', default=False, action='store_true')
    parser.add_argument('--hinge', default=False, action='store_true')

    # Model
    parser.add_argument(
        '--model',
        choices=model_attributes.keys(),
        default='clip')
    parser.add_argument('--train_from_scratch', action='store_true', default=False)

    # Optimization
    parser.add_argument('--n_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--scheduler', action='store_true', default=False)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--minimum_variational_weight', type=float, default=0)
    # Misc
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--show_progress', default=False, action='store_true')
    parser.add_argument('--log_dir', default=configs.LOG_FOLDER)
    parser.add_argument('--log_every', default=1e8, type=int)
    parser.add_argument('--save_step', type=int, default=1e8)
    parser.add_argument('--save_best', action='store_true', default=False)
    parser.add_argument('--save_last', action='store_true', default=False)
    parser.add_argument('--algo_suffix', type=str, default='', help='The suffix of log folder name')
    parser.add_argument('--hessian_align', action='store_true', default=False)
    parser.add_argument('--grad_alpha', type=float, default=1e-4)
    parser.add_argument('--hess_beta', type=float, default=1e-4)
    args = parser.parse_args()
    check_args(args)

    # BERT-specific configs copied over from run_glue.py
    if args.model == 'bert':
        args.max_grad_norm = 1.0
        args.adam_epsilon = 1e-8
        args.warmup_steps = 0
    # breakpoint()
    if args.robust:
        algo = 'groupDRO'
    elif args.reweight_groups:
        algo = 'reweight'
    elif args.hessian_align:
        algo = 'HessianERM'
    else:
        algo = 'ERM'

    grad_alpha_formatted = "{:.1e}".format(args.grad_alpha).replace('.0e', 'e')
    hess_beta_formatted = "{:.1e}".format(args.hess_beta).replace('.0e', 'e')
    lr_formatted = "{:.1e}".format(args.lr).replace('.0e', 'e')

    # args.log_dir = os.path.join(args.log_dir, args.dataset, args.model, algo + args.algo_suffix, f's{args.seed}', f'grad_alpha_{args.grad_alpha}_hess_beta_{args.hess_beta}')
    # breakpoint()
    if args.dataset == 'MultiNLI':
        args.log_dir = os.path.join(args.log_dir, args.dataset, args.model, algo + args.algo_suffix, f's{args.seed}',
                                    f"grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}")
    elif args.seed >= 200:
        args.seed = args.seed - 200
        args.log_dir = os.path.join(args.log_dir, args.dataset, args.model,algo + args.algo_suffix,f'lr{lr_formatted}_batchsize_{args.batch_size}', f's{args.seed}',
                                    f"grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}")
    elif args.seed >= 100:
        args.seed = args.seed - 100
        args.log_dir = os.path.join(args.log_dir, args.dataset, args.model,algo + args.algo_suffix,f'lr{lr_formatted}_batchsize_{args.batch_size}', f's{args.seed}',
                                    f"grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}")
    else:
        args.log_dir = os.path.join(args.log_dir, args.dataset, args.model, algo + args.algo_suffix, f's{args.seed}',
                                    f"grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}{'_no_scheduler' if not args.scheduler else ''}")

    print(f'Logging to {args.log_dir}')
    # breakpoint()
    # if os.path.exists(args.log_dir) and args.resume:
    if os.path.exists(args.log_dir) and args.resume:
        resume = True
        mode = 'a'
    else:
        resume = False
        mode = 'w'

    ## Initialize logs
    if not os.path.exists(args.log_dir):
        # print(args.log_dir)
        # breakpoint()
        os.makedirs(args.log_dir)

    logger = Logger(os.path.join(args.log_dir, 'log.txt'), mode)
    # Record args
    log_args(args, logger)

    set_seed(args.seed)

    # Data
    # Test data for label_shift_step is not implemented yet
    test_data = None
    test_loader = None
    if args.shift_type == 'confounder':
        train_data, val_data, test_data = prepare_data(args, train=True)
    elif args.shift_type == 'label_shift_step':
        train_data, val_data = prepare_data(args, train=True)

    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 4, 'pin_memory': True}
    train_loader = train_data.get_loader(train=True, reweight_groups=args.reweight_groups, **loader_kwargs)
    val_loader = val_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)
    if test_data is not None:
        test_loader = test_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)

    data = {}
    data['train_loader'] = train_loader
    data['val_loader'] = val_loader
    data['test_loader'] = test_loader
    data['train_data'] = train_data
    data['val_data'] = val_data
    data['test_data'] = test_data
    n_classes = train_data.n_classes

    log_data(data, logger)

    ## Initialize model
    pretrained = not args.train_from_scratch
    optimizer, scheduler,clf = None, None, None
    # breakpoint()
    if resume:
        # breakpoint()
        model = torch.load(os.path.join(args.log_dir, 'last_model.pth'))
        # clf = torch.load(os.path.join(args.log_dir, 'last_clf.pth'))
        if args.scheduler:
            # breakpoint()
            optimizer = torch.load(os.path.join(args.log_dir, 'last_optimizer.pth'))
            scheduler = torch.load(os.path.join(args.log_dir, 'last_scheduler.pth'))
        d = train_data.input_size()[0]
    elif model_attributes[args.model]['feature_type'] in ('precomputed', 'raw_flattened'):
        assert pretrained
        # Load precomputed features
        d = train_data.input_size()[0]
        model = nn.Linear(d, n_classes)
        model.has_aux_logits = False
    elif args.model == 'vits':
        model = timm.create_model(
            'vit_small_patch16_224.dino',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        )
    elif args.model == 'clip512':
        model, preprocess = open_clip.create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP-512')
        tokenizer = open_clip.get_tokenizer('hf-hub:timm/ViT-B-16-SigLIP-512')
    elif args.model == 'clip':
        # model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        #     'hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')
        # tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')

        model, preprocess = open_clip.create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP')
        tokenizer = open_clip.get_tokenizer('hf-hub:timm/ViT-B-16-SigLIP')
    elif args.model == 'resnet50':
        model = torchvision.models.resnet50(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'resnet34':
        model = torchvision.models.resnet34(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'wideresnet50':
        model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'bert':
        assert args.dataset == 'MultiNLI'

        from transformers import BertConfig, BertForSequenceClassification
        config_class = BertConfig
        model_class = BertForSequenceClassification

        config = config_class.from_pretrained(
            'bert-base-uncased',
            num_labels=3,
            finetuning_task='mnli')
        model = model_class.from_pretrained(
            'bert-base-uncased',
            from_tf=False,
            config=config)
    else:
        raise ValueError('Model not recognized.')

    logger.flush()

    ## Define the objective
    if args.hinge:
        assert args.dataset in ['CelebA', 'CUB']  # Only supports binary

        def hinge_loss(yhat, y):
            # The torch loss takes in three arguments so we need to split yhat
            # It also expects classes in {+1.0, -1.0} whereas by default we give them in {0, 1}
            # Furthermore, if y = 1 it expects the first input to be higher instead of the second,
            # so we need to swap yhat[:, 0] and yhat[:, 1]...
            torch_loss = torch.nn.MarginRankingLoss(margin=1.0, reduction='none')
            y = (y.float() * 2.0) - 1.0
            return torch_loss(yhat[:, 1], yhat[:, 0], y)

        criterion = hinge_loss
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

    if resume:
        df = pd.read_csv(os.path.join(args.log_dir, 'test.csv'))
        epoch_offset = df.loc[len(df) - 1, 'epoch'] + 1
        logger.write(f'starting from epoch {epoch_offset}')
    else:
        epoch_offset = 0
    train_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, 'train.csv'), train_data.n_groups, mode=mode)
    val_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, 'val.csv'), train_data.n_groups, mode=mode)
    test_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, 'test.csv'), train_data.n_groups, mode=mode)
    # train(model, criterion, data, logger, train_csv_logger, val_csv_logger, test_csv_logger, args,
    #       epoch_offset=epoch_offset)
    train(model, criterion, data,
          logger, train_csv_logger, val_csv_logger, test_csv_logger,
          args, epoch_offset=epoch_offset)

    train_csv_logger.close()
    val_csv_logger.close()
    test_csv_logger.close()


def check_args(args):
    if args.shift_type == 'confounder':
        assert args.confounder_names
        assert args.target_name
    elif args.shift_type.startswith('label_shift'):
        assert args.minority_fraction
        assert args.imbalance_ratio


if __name__ == '__main__':
    main()
