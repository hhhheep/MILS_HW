# Reproducing The CIFAR-10-C AI-Claim Audit

## Dataset paths

- Clean CIFAR-10: `/ssd1/datasets/cifar10`
- CIFAR-10-C: `/ssd3/fan/hw/data/CIFAR-10-C`

## Model checkpoints

- `resnet20`: `/ssd1/hsiao/statpruning/pytorch_resnet_cifar10-master/pretrained_models/resnet20.th`
- `resnet56`: `/ssd1/hsiao/statpruning/pytorch_resnet_cifar10-master/pretrained_models/resnet56.th`

## Scripts

- Evaluation: `code/evaluate_cifar10c.py`

## Environment

Python path used:

```text
/home/wang256hep/miniconda3/envs/aigi_cu129/bin/python
```

The full run was executed on CPU because CUDA was not visible in the sandbox. This affects runtime only, not the experimental definition.

## Smoke test

```bash
env MPLCONFIGDIR=/tmp/matplotlib \
  /home/wang256hep/miniconda3/envs/aigi_cu129/bin/python \
  code/evaluate_cifar10c.py \
  --smoke --batch_size 256
```

## Full evaluation

```bash
env MPLCONFIGDIR=/tmp/matplotlib \
  /home/wang256hep/miniconda3/envs/aigi_cu129/bin/python \
  code/evaluate_cifar10c.py \
  --batch_size 1024
```

## Output descriptions

- `evidence/results.csv`: full model-condition metrics.
- `evidence/claim_audit.csv`: original claim audit output.
- `evidence/failure_cases.csv`: selected failure case metadata.
- `evidence/audit_recheck.json`: recomputed validation of the five claims.
- `tables/`: report-ready CSV and Markdown tables.
- `figures/`: report-ready figures and failure case panel.
