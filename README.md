# Closed-Form Momnet Alignment (CMA)



Our code is devided into two parts, `ISR/` for linear probing experiments and `DomainBed/` for full fine-tuning experiments.

+ `ISR/`: This part is built on the [code](https://github.com/Haoxiang-Wang/ISR) of [Provable Domain Generalization via Invariant-Feature Subspace Recovery](https://arxiv.org/abs/2201.12919). Please install necessary packages following their instructions.
+ `DomainBed/`: This part is built on the [code](https://github.com/facebookresearch/DomainBed) of [In Search of Lost Domain Generalization](https://arxiv.org/abs/2007.01434). Please install necessary packages following their instructions.

## Linear Probing (Sec. 7.1)
In `ISR/`

**Step 1:** Download data following `ISR/README.md`

**Step 2:** Train BERTs for MultiNLI:
```
python launch_train.py
```

**Step 3:** Parse features:
```
python launch_parse.py
```

**Step 4:** Evaluate different algorithms 
e.g., 

Evaluate Fishr on CUB
```
python eval.py --dataset CUB --hessian_approx_method fishr
```
or Evaluate CMA on CUB

```
python eval.py --dataset CUB --hessian_approx_method exact
```



## Full Fine-Tuning (Sec. 7.2)

In `DomainBed/`

**Step 1:** Download data following 'DomainBed/README.md`

**Step 2:** Launch a sweep

```
python -m domainbed.scripts.sweep launch\
       --data_dir=./domainbed/data/\
       --output_dir=./domainbed/results\
       --command_launcher multi_gpu\
       --algorithms ERM Fishr CORAL CMA\
       --datasets ColoredMNIST ROTATEDMNIST VLCS PACS TerraIncognita\
       --single_test_envs\
       --n_hparams 5\
       --n_trials 3
```

**Step 3:** View results
```       
python -m domainbed.scripts.collect_results\
       --input_dir=./domainbed/results
```

