


TRAIN_COMMANDS = dict(
    CelebA=
    {
        "ERM": "-s confounder -d CelebA -t Blond_Hair -c Male --model resnet50 --weight_decay 0.01 --lr 0.0001 "
               "--batch_size 128 --n_epochs 50",  # ERM

        "groupDRO": "-s confounder -d CelebA -t Blond_Hair -c Male --model resnet50 --weight_decay 0.1 --lr 1e-05 "
                    "--batch_size 128 --n_epochs 50 --reweight_groups --robust --alpha 0.01 --gamma 0.1 --generalization_adjustment 3",
        # groupDRO
        "reweight": "-s confounder -d CelebA -t Blond_Hair -c Male --model resnet50 --weight_decay 0.1 --lr 1e-05 "
                    "--batch_size 128 --n_epochs 50 --reweight_groups"  # reweight,
    }
    ,
    CUB={
        "ERM": '-s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model resnet50 --weight_decay 0.1 --lr 1e-5 '
               '--batch_size 128 --n_epochs 300',  # ERM
        "reweight": '-s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model resnet50 --weight_decay 1 --lr 1e-05 '
                    '--batch_size 128 --n_epochs 300 --reweight_groups',  # reweight
        "groupDRO": ' -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model resnet50 --weight_decay 1 --lr 1e-05 '
                    '--batch_size 128 --n_epochs 300 --reweight_groups --robust --alpha 0.01 --gamma 0.1 --generalization_adjustment 2',
        # groupDRO
    },
    MultiNLI={
        "ERM": '-s confounder -d MultiNLI -t gold_label_random -c sentence2_has_negation --model bert --weight_decay 0 --lr '
               f'2e-05 --batch_size 32 --n_epochs 3',  # ERM
        "groupDRO": '-s confounder -d MultiNLI -t gold_label_random -c sentence2_has_negation --model bert --weight_decay 0 --lr '
                    f'2e-05 --batch_size 32 --n_epochs 3 --reweight_groups --robust --alpha 0.01 --gamma 0.1 '
                    '--generalization_adjustment 0',  # groupDRO
        "reweight": '-s confounder -d MultiNLI -t gold_label_random -c sentence2_has_negation --model bert --weight_decay 0 --lr '
                    f'2e-05 --batch_size 32 --n_epochs 3 --reweight_groups',  # reweight
    }
)

TRAIN_COMMANDS_CLIP = dict(
    CelebA=
    {
        "ERM": "-s confounder -d CelebA -t Blond_Hair -c Male --model clip --weight_decay 0.01 --lr 0.0001 "
               "--batch_size 128 --n_epochs 50",  # ERM
        "groupDRO": "-s confounder -d CelebA -t Blond_Hair -c Male --model clip --weight_decay 0.1 --lr 1e-05 "
                    "--batch_size 128 --n_epochs 50 --reweight_groups --robust --alpha 0.01 --gamma 0.1 --generalization_adjustment 3",
        # groupDRO
        "reweight": "-s confounder -d CelebA -t Blond_Hair -c Male --model clip --weight_decay 0.1 --lr 1e-05 "
                    "--batch_size 128 --n_epochs 50 --reweight_groups"  # reweight,
    }
    ,
    CUB={
        "ERM": '-s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model clip --weight_decay 0.1 --lr 0.0001 '
               '--batch_size 128 --n_epochs 100',  # ERM
        # "ERM": '-s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model resnet50 --weight_decay 0.1 --lr 0.0001 '
        #         '--batch_size 32 --n_epochs 300',  # ERM
        "reweight": '-s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model clip --weight_decay 1 --lr 1e-05 '
                    '--batch_size 32 --n_epochs 300 --reweight_groups',  # reweight
        "groupDRO": ' -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model clip --weight_decay 1 --lr 1e-05 '
                    '--batch_size 32 --n_epochs 300 --reweight_groups --robust --alpha 0.01 --gamma 0.1 --generalization_adjustment 2',
        # groupDRO
    },
    MultiNLI={
        "ERM": '-s confounder -d MultiNLI -t gold_label_random -c sentence2_has_negation --model clip --weight_decay 0 --lr '
               f'2e-05 --batch_size 32 --n_epochs 3',  # ERM
        "groupDRO": '-s confounder -d MultiNLI -t gold_label_random -c sentence2_has_negation --model clip --weight_decay 0 --lr '
                    f'2e-05 --batch_size 32 --n_epochs 3 --reweight_groups --robust --alpha 0.01 --gamma 0.1 '
                    '--generalization_adjustment 0',  # groupDRO
        "reweight": '-s confounder -d MultiNLI -t gold_label_random -c sentence2_has_negation --model clip --weight_decay 0 --lr '
                    f'2e-05 --batch_size 32 --n_epochs 3 --reweight_groups',  # reweight
    }
)


TRAIN_COMMANDS_CLIP_512 = dict(
    CelebA=
    {
        "ERM": "-s confounder -d CelebA -t Blond_Hair -c Male --model clip512 --weight_decay 0.01 --lr 0.0001 "
               "--batch_size 128 --n_epochs 50",  # ERM
        "groupDRO": "-s confounder -d CelebA -t Blond_Hair -c Male --model clip512 --weight_decay 0.1 --lr 1e-05 "
                    "--batch_size 128 --n_epochs 50 --reweight_groups --robust --alpha 0.01 --gamma 0.1 --generalization_adjustment 3",
        # groupDRO
        "reweight": "-s confounder -d CelebA -t Blond_Hair -c Male --model clip512 --weight_decay 0.1 --lr 1e-05 "
                    "--batch_size 128 --n_epochs 50 --reweight_groups"  # reweight,
    }
    ,
    CUB={
        "ERM": '-s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model clip512 --weight_decay 0.1 --lr 0.0001 '
               '--batch_size 128 --n_epochs 300',  # ERM
        # "ERM": '-s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model resnet50 --weight_decay 0.1 --lr 0.0001 '
        #         '--batch_size 32 --n_epochs 300',  # ERM
        "reweight": '-s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model clip512 --weight_decay 1 --lr 1e-05 '
                    '--batch_size 32 --n_epochs 300 --reweight_groups',  # reweight
        "groupDRO": ' -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model clip512 --weight_decay 1 --lr 1e-05 '
                    '--batch_size 32 --n_epochs 300 --reweight_groups --robust --alpha 0.01 --gamma 0.1 --generalization_adjustment 2',
        # groupDRO
    },
    MultiNLI={
        "ERM": '-s confounder -d MultiNLI -t gold_label_random -c sentence2_has_negation --model clip512 --weight_decay 0 --lr '
               f'2e-05 --batch_size 32 --n_epochs 3',  # ERM
        "groupDRO": '-s confounder -d MultiNLI -t gold_label_random -c sentence2_has_negation --model clip512 --weight_decay 0 --lr '
                    f'2e-05 --batch_size 32 --n_epochs 3 --reweight_groups --robust --alpha 0.01 --gamma 0.1 '
                    '--generalization_adjustment 0',  # groupDRO
        "reweight": '-s confounder -d MultiNLI -t gold_label_random -c sentence2_has_negation --model clip512 --weight_decay 0 --lr '
                    f'2e-05 --batch_size 32 --n_epochs 3 --reweight_groups',  # reweight
    }
)


TRAIN_COMMANDS_ViT_S = dict(
    CelebA=
    {
        "ERM": "-s confounder -d CelebA -t Blond_Hair -c Male --model vits --weight_decay 5e-4 --lr 1e-5 "
               "--batch_size 128 --n_epochs 50",  # ERM
        "groupDRO": "-s confounder -d CelebA -t Blond_Hair -c Male --model vits --weight_decay 5e-4 --lr 1e-05 "
                    "--batch_size 128 --n_epochs 50 --reweight_groups --robust --alpha 0.01 --gamma 0.1 --generalization_adjustment 3",
        # groupDRO
        "reweight": "-s confounder -d CelebA -t Blond_Hair -c Male --model vits --weight_decay 5e-4 --lr 1e-05 "
                    "--batch_size 128 --n_epochs 50 --reweight_groups"  # reweight,
    }
    ,
    CUB={
        "ERM": '-s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model vits --weight_decay 5e-4 --lr 1e-05 '
               '--batch_size 128 --n_epochs 100',  # ERM
        # "ERM": '-s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model resnet50 --weight_decay 0.1 --lr 0.0001 '
        #         '--batch_size 32 --n_epochs 300',  # ERM
        "reweight": '-s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model vits --weight_decay 5e-4 --lr 1e-05 '
                    '--batch_size 32 --n_epochs 100 --reweight_groups',  # reweight
        "groupDRO": ' -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model vits --weight_decay 5e-4 --lr 1e-05 '
                    '--batch_size 32 --n_epochs 100 --reweight_groups --robust --alpha 0.01 --gamma 0.1 --generalization_adjustment 2',
        # groupDRO
    },
    MultiNLI={
        "ERM": '-s confounder -d MultiNLI -t gold_label_random -c sentence2_has_negation --model vits --weight_decay 0 --lr '
               f'2e-05 --batch_size 32 --n_epochs 3',  # ERM
        "groupDRO": '-s confounder -d MultiNLI -t gold_label_random -c sentence2_has_negation --model vits --weight_decay 0 --lr '
                    f'2e-05 --batch_size 32 --n_epochs 3 --reweight_groups --robust --alpha 0.01 --gamma 0.1 '
                    '--generalization_adjustment 0',  # groupDRO
        "reweight": '-s confounder -d MultiNLI -t gold_label_random -c sentence2_has_negation --model vits --weight_decay 0 --lr '
                    f'2e-05 --batch_size 32 --n_epochs 3 --reweight_groups',  # reweight
    }
)
def get_train_command(dataset: str, algo: str , model: str = 'clip',gpu_idx: int = None, train_script: str = 'run_expt.py', hessian_align: bool = False,
                      algo_suffix: str = '',seed:int=None,save_best:bool=True,save_last:bool=True, resume:bool = False, scheduler:bool = False,
                      grad_alpha:float = 10e-5, hess_beta:float = 10e-5, learning_rate:float = None, batch_size:int = None):
    prefix = f'CUDA_VISIBLE_DEVICES={gpu_idx}' if gpu_idx is not None else ''
    # prefix = ''
    suffix = f' --algo_suffix {algo_suffix}' if algo_suffix else ''
    if save_best:
        suffix += ' --save_best'
    if save_last:
        suffix += ' --save_last'
    if resume:
        suffix += ' --resume'
    seed = 0 if seed is None else seed
    if model == 'vits':
        args_command = TRAIN_COMMANDS_ViT_S[dataset][algo]
    elif model == 'clip512':
        args_command = TRAIN_COMMANDS_CLIP_512[dataset][algo]
    elif model == 'clip':
        args_command = TRAIN_COMMANDS_CLIP[dataset][algo]
    else:
        args_command = TRAIN_COMMANDS[dataset][algo]
    command = f"{prefix} python {train_script} {args_command} --seed {seed} {suffix} {'--hessian_align' if hessian_align else ''} {'--scheduler' if scheduler else ''} --grad_alpha {grad_alpha} --hess_beta {hess_beta}"
    if learning_rate:
        # replace current learning rate with the new learning rate
        # first extract current lr string
        lr_str = command.split('--lr')[1].split(' ')[1]
        command = command.replace(f'--lr {lr_str}', f'--lr {learning_rate}')
        command = command.replace(f'seed {seed}', f'seed {seed+100}')
        seed += 100
    if batch_size:
        # replace current batch size with the new batch size
        # first extract current batch size string
        batch_size_str = command.split('--batch_size')[1].split(' ')[1]
        command = command.replace(f'--batch_size {batch_size_str}', f'--batch_size {batch_size}')
        command = command.replace(f'seed {seed}', f'seed {seed+100}')

    # breakpoint()
    return command