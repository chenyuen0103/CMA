# Closed-Form Momnet Alignment (CMA)

This repo contains the code and data for the paper:

[**Moment Alignment: Unifying Gradient and Hessian Matching for Domain Generalization**]()(2025)

*Yuen Chen, Haozhe Si, Guojun Zhang, and Han Zhao*


Our code is devided into two parts, `ISR/` for linear probing experiments and `DomainBed/` for full fine-tuning experiments.

+ `ISR/`: This part is built on the [code](https://github.com/Haoxiang-Wang/ISR) of [Provable Domain Generalization via Invariant-Feature Subspace Recovery](https://arxiv.org/abs/2201.12919). Please install necessary packages following their instructions.
+ `DomainBed/`: This part is built on the [code](https://github.com/facebookresearch/DomainBed) of [In Search of Lost Domain Generalization](https://arxiv.org/abs/2007.01434). Please install necessary packages following their instructions.

## Linear Probing
In `ISR/`

**Step 1:** Download data following `ISR/README.md`

In `ISR/real_datasets`

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

**Step 5:** Print result table
```
python collect_results.py
```



## Full Fine-Tuning

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

