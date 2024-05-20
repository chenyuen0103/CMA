# Closed-Form Momnet Alignment (CMA)



Our code is devided into two parts, `ISR/` for linear probing experiments and `DomainBed/` for full fine-tuning experiments.

+ `ISR/`: This part is built on the [code](https://github.com/Haoxiang-Wang/ISR) of [Provable Domain Generalization via Invariant-Feature Subspace Recovery](https://arxiv.org/abs/2201.12919). Please install necessary packages following their instructions.
+ `DomainBed/`: This part is built on the [code](https://github.com/facebookresearch/DomainBed) of [In Search of Lost Domain Generalization](https://arxiv.org/abs/2007.01434). Please install necessary packages following their instructions.

## Linear Probing (Sec. 7.1)
In `ISR/`

**Step 1:** Download Data following `ISR/README.md'

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
