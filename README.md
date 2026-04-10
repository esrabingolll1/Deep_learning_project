# Deep Learning Project — CIFAR-10 Regularization and FGSM Robustness

This repository contains a team project analyzing how regularization and optimization choices affect:
- **generalization** (overfitting control), and
- **adversarial robustness** under FGSM attacks.

The project is implemented in Python/PyTorch and organized as reproducible experiment scripts.

## Team Responsibilities
| Student ID | Role |
|---|---|
| 210911026 | System Architect, training core, model-capacity ablation |
| 210911051 | L1/L2 regularization and optimizer comparison |
| 190722054 | Dropout + BatchNorm + initialization study (Default/He/Xavier) |
| 210911028 | Data augmentation, label smoothing, scheduler analysis |
| 210911030 | FGSM robustness and optimizer robustness comparison |

## Technical Stack
- Python
- PyTorch / Torchvision
- NumPy
- Matplotlib

## Dataset
- **CIFAR-10**: 50,000 training + 10,000 test images, 10 classes, 32×32 RGB
- Train preprocessing: random crop + horizontal flip + normalization
- Test preprocessing: normalization only

## Core Training Features
Implemented in `train.py` and exposed via `main.py`:
- Cross-Entropy loss (+ optional label smoothing)
- L1 regularization (manual penalty)
- L2 regularization (`weight_decay`)
- Adam and SGD(+momentum)
- CosineAnnealingLR / ReduceLROnPlateau scheduler
- Early stopping (patience=5)
- Gradient clipping
- Optional adversarial training

## Main Entry Point
Train a single configuration:
```bash
python main.py --epochs 10 --optimizer adam --scheduler cosine --l2 0.001
```

## Experiment Scripts
- `generate_student1_capacity_report.py` — Small/Base/Large capacity ablation
- `student2_optimizer_comparison.py` — L1/L2 with Adam vs SGD+Momentum
- `generate_student3_init_report.py` — Default vs He vs Xavier initialization
- `generate_student4_report.py` — augmentation/smoothing with scheduler comparison
- `fgsm_eval.py` — FGSM robustness across methods and optimizers

## FGSM Note (Methodological Correctness)
Inputs are normalized, so adversarial clipping is applied in **normalized channel bounds** (not raw `[0,1]` clipping). This is critical for valid robustness evaluation.

## Artifacts
Outputs are saved under:
- `ogrenci1/`
- `ogrenci2/`
- `ogrenci3/`
- `ogrenci4/`
- `ogrenci5/`
- and `results/` for shared plots

## Installation
```bash
pip install -r requirements.txt
```

## Full Technical Report
See `REPORT.md` for methodology, experiment details, and interpretation.
