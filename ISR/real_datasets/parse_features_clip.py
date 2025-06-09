import argparse
import os
import pickle
from itertools import product
import timm
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm.auto import tqdm

from configs.model_config import model_attributes
from data import dataset_attributes, shift_types, prepare_data
from utils.train_utils import check_args
from utils.train_utils import set_seed
import open_clip
import pdb


import pickle
from itertools import product
import gc

parser = argparse.ArgumentParser()

# Settings
parser.add_argument('-d', '--dataset',
                    choices=dataset_attributes.keys(), default='MultiNLI')
parser.add_argument('-s', '--shift_type',
                    choices=shift_types, default='confounder')


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
parser.add_argument('--automatic_adjustment',
                    default=False, action='store_true')
parser.add_argument('--robust_step_size', default=0.01, type=float)
parser.add_argument('--use_normalized_loss',
                    default=False, action='store_true')
parser.add_argument('--btl', default=False, action='store_true')
parser.add_argument('--hinge', default=False, action='store_true')

# Model
parser.add_argument(
    '--model',
    choices=model_attributes.keys(),
    default='vits')
parser.add_argument('--train_from_scratch', action='store_true', default=False)

# Optimization
parser.add_argument('--n_epochs', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--scheduler', action='store_true', default=False)
parser.add_argument('--weight_decay', type=float, default=5e-5)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--minimum_variational_weight', type=float, default=0)
# Misc
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--show_progress', default=False, action='store_true')
parser.add_argument('--log_dir', default='/data/common/inv-feature/logs/')
parser.add_argument('--log_every', default=1e8, type=int)
parser.add_argument('--save_step', type=int, default=1e8)
parser.add_argument('--save_best', action='store_true', default=False)
parser.add_argument('--save_last', action='store_true', default=False)

parser.add_argument('--parse_algos', nargs='+',
                    default=['ERM', 'groupDRO', 'reweight'])
parser.add_argument('--parse_model_selects', nargs='+',
                    default=['best', 'best_avg_acc', 'last'],
                    help='best is based on worst-group validation accuracy.')
parser.add_argument('--parse_seeds', nargs='+',
                    default=[0])
parser.add_argument(
    '--parse_dir', default='/data/common/inv-feature/logs/', type=str)

args = parser.parse_args()
check_args(args)

# model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')
# tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')


# model = model.eval()
if args.dataset == "MultiNLI":
    model,  _, preprocess  = open_clip.create_model_and_transforms('ViT-B-32',
                                                                   pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model = model.eval()
    model = model.to('cuda')
    tokenizer = tokenizer
    # model = model.encode_text


else:
    model = timm.create_model(
    'vit_base_patch16_clip_384.laion2b_ft_in1k',
    pretrained=True,
    num_classes=0,  # remove classifier nn.Linear
)
# data_config = timm.data.resolve_model_data_config(model)
# transforms = timm.data.create_transform(**data_config, is_training=False)
#
if args.model == 'bert':
    args.max_grad_norm = 1.0
    args.adam_epsilon = 1e-8
    args.warmup_steps = 0



if args.robust:
    algo = 'groupDRO'
elif args.reweight_groups:
    algo = 'reweight'
else:
    algo = 'ERM'

args.log_dir = os.path.join(args.log_dir, args.dataset, algo, f's{args.seed}')

if os.path.exists(args.log_dir) and args.resume:
    resume=True
    mode='a'
else:
    resume=False
    mode='w'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(args.seed)
# Data
# Test data for label_shift_step is not implemented yet
test_data = None
test_loader = None
if args.shift_type == 'confounder':
    train_data, val_data, test_data = prepare_data(args, train=True)
elif args.shift_type == 'label_shift_step':
    train_data, val_data = prepare_data(args, train=True)

loader_kwargs = {'batch_size': args.batch_size,
                 'num_workers': 4, 'pin_memory': True}
train_loader = train_data.get_loader(
    train=True, reweight_groups=args.reweight_groups, **loader_kwargs)
val_loader = val_data.get_loader(
    train=False, reweight_groups=None, **loader_kwargs)
if test_data is not None:
    test_loader = test_data.get_loader(
        train=False, reweight_groups=None, **loader_kwargs)

data = {}
data['train_loader'] = train_loader
data['val_loader'] = val_loader
data['test_loader'] = test_loader
data['train_data'] = train_data
data['val_data'] = val_data
data['test_data'] = test_data
n_classes = train_data.n_classes


use_clip = type(model).__name__ == 'CLIP'
save_prefix = 'CLIP_' if use_clip else ''
load_ckpt = not use_clip
#%%
load_ckpt = False

if args.dataset == "MultiNLI":
    encoder = model.encode_text
else:
    # breakpoint()
    encoder = model
output_layer = None


def process_batch(model,  x, y=None, g=None, bert=False):
    if args.dataset in ['MultiNLI']:
        breakpoint()
        input_ids = x[:, :, 0]
        input_masks = x[:, :, 1]
        segment_ids = x[:, :, 2]
        outputs = model.encode_text(
            input_ids=input_ids,
            attention_mask=input_masks,
            token_type_ids=segment_ids,
        )
        pooled_output = outputs[1]
        logits = model.classifier(pooled_output)
        result = {'feature': pooled_output.detach().cpu().numpy(),
                  'pred': np.argmax(logits.detach().cpu().numpy(), axis=1),}
    else:
        # breakpoint()
        features = encoder(x)
        result = {'feature': features.detach().cpu().numpy(), }


    if output_layer is not None:
        logits = output_layer(features)
        result['pred'] = np.argmax(logits.detach().cpu().numpy(), axis=1)



    if y is not None: result['label'] = y.detach().cpu().numpy()
    if g is not None: result['group'] = g.detach().cpu().numpy()
    return result




# algos = ['ERM', 'reweight', 'groupDRO']
algos = ['ERM']

model_selects = ['init']
seeds = np.arange(5)
for algo, model_select, seed in tqdm(list(product(algos, model_selects, seeds)), desc='Iter'):
    print('Current iter:', algo, model_select, seed)
    save_dir = f'./inv-feature-ViT-B/logs/{args.dataset}/{algo}/s{seed}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if load_ckpt:
        model.load_state_dict(torch.load(save_dir + f'/{model_select}_model.pth',
                                         map_location='cpu').state_dict())

    model.eval()
    model = model.to(device)
    for split, loader in zip(['train', 'val', 'test'], [train_loader, val_loader, test_loader]):
        results = []
        fname = f'{split}_data.p'
        fname = save_prefix + model_select + '_' + fname
        if os.path.exists(save_dir + '/' + fname):
            continue
        with torch.set_grad_enabled(False):
            for batch_idx, batch in enumerate(tqdm(loader)):
                batch = tuple(t.to(device) for t in batch)
                x = batch[0]
                y = batch[1]
                g = batch[2]
                if args.model.startswith("bert"):
                    result = process_batch(model, x, y, g, bert=True)
                else:
                    result = process_batch(model, x, y, g, bert=False)
                results.append(result)
        parsed_data = {}
        breakpoint()
        for key in results[0].keys():
            parsed_data[key] = np.concatenate([result[key] for result in results])


        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        pickle.dump(parsed_data, open(save_dir + '/' + fname, 'wb'))

        print("Parsed data save to ", save_dir + '/' + fname, 'wb')

        del results
        del parsed_data

gc.collect()