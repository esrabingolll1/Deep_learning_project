# REPORT — Regularization and Adversarial Robustness on CIFAR-10

## 1. Project Objective
This project investigates how regularization methods affect both:
- **generalization performance** (overfitting control), and
- **adversarial robustness** under **FGSM** attacks.

We train CNN models on CIFAR-10 and compare methods from Weeks 2–5 in a controlled and reproducible pipeline.

## 2. Research Questions
1. Which regularization methods reduce overfitting most effectively?
2. Which methods preserve performance better under adversarial perturbations?
3. How do optimization choices (Adam vs SGD, clipping, schedulers) influence robustness and convergence?

## 3. Dataset and Preprocessing
- Dataset: **CIFAR-10** (50,000 train / 10,000 test, RGB 32×32, 10 classes)
- Training transforms: `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`, `Normalize`
- Test transforms: `Normalize`
- Implementation: `data.py`

### Why CIFAR-10?
CIFAR-10 is a balanced benchmark with sufficient complexity for regularization and robustness studies while remaining computationally tractable for repeated ablation experiments.

## 4. Model and Training Core
Implementation files: `models.py`, `train.py`, `main.py`

### Model
Parameterized CNN with optional:
- `dropout_rate`
- `use_batchnorm`
- `init_scheme` (`default`, `he`, `xavier`)
- capacity controls: `base_channels`, `fc_hidden_dim`

### Optimization and Loss
- Base loss: `CrossEntropyLoss`
- Label smoothing: `CrossEntropyLoss(label_smoothing=...)`
- L1: manual penalty term added to loss
- L2: optimizer `weight_decay`
- Optimizers: Adam, SGD(+momentum)
- Schedulers: CosineAnnealingLR, ReduceLROnPlateau
- Early stopping: patience=5 with best-weight restore
- Gradient clipping: optional max-norm clipping

### Why Cross-Entropy?
For multi-class classification, minimizing cross-entropy is equivalent to maximizing conditional likelihood (MLE perspective), making it a theoretically grounded objective.

### Why ReLU?
ReLU preserves stronger gradients in active regions and mitigates vanishing-gradient behavior in deep stacks, improving optimization stability.

## 5. Methodology
We use controlled comparisons: one factor changes while core settings remain fixed (same dataset family, architecture family, seed policy, optimizer family where applicable).

### Methods included
- L1 regularization
- L2 weight decay
- Dropout
- Batch Normalization
- Data augmentation
- Label smoothing
- Capacity ablation
- He/Xavier initialization comparison
- FGSM evaluation
- Adversarial training
- Scheduler/optimizer/clipping analyses

## 6. FGSM Setup (Critical Correctness)
FGSM perturbation:
\[
x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(x,y))
\]

Because inputs are normalized, clipping is applied in **normalized channel bounds**:
\[
x_{adv} \in \left[\frac{0-\mu_c}{\sigma_c}, \frac{1-\mu_c}{\sigma_c}\right]
\]

This detail is essential for methodological correctness and prevents invalid perturbation scaling.

## 7. Experiments and Results

### 7.1 Student 1 — Capacity Ablation
Artifacts:
- `ogrenci1/capacity_ablation_summary.json`
- `ogrenci1/student1_capacity_ablation_curves.png`
- `ogrenci1/generalization_gap_bar.png`

| Config | base_channels | fc_hidden_dim | Params | Best Test Acc (%) |
|---|---:|---:|---:|---:|
| Small | 8 | 128 | 278,842 | 47.30 |
| Base | 16 | 256 | 1,111,914 | 56.48 |
| Large | 32 | 512 | 4,440,778 | **60.05** |

Interpretation: higher capacity improved accuracy in the reported run; train-test gap tracking is included to monitor generalization behavior.

### 7.2 Student 2 — L1/L2 and Optimizer Analysis
Artifacts (selected):
- `student2_optimizer_comparison.py`
- `ogrenci2/seyreklik_karsilastirma_grafigi.png`
- `ogrenci2/agirlik_histogrami_dagilimi.png`
- `ogrenci2/final_accuracy_bar.png`

Interpretation: L1 shows stronger sparsity effects, L2 provides smoother shrinkage; optimizer choice affects convergence speed and final stability.

### 7.3 Student 3 — Initialization with Dropout+BatchNorm
Artifacts:
- `ogrenci3/student3_init_summary.json`
- `ogrenci3/student3_init_comparison_curves.png`

| Init Scheme | Best Test Acc (%) |
|---|---:|
| Default | **63.76** |
| Xavier | 49.04 |
| He | 10.28 |

Interpretation: in this setup, default initialization outperformed Xavier and He; initialization significantly affected convergence quality.

### 7.4 Student 4 — Augmentation + Label Smoothing + Scheduler
Artifacts:
- `ogrenci4/week5_scheduler_analysis.json`
- `ogrenci4/student4_scheduler_aug_smooth_comparison.png`

| Experiment | Best Test Acc (%) | Epoch to 70% |
|---|---:|---:|
| Cosine \| Base (No Aug, No Smooth) | 74.42 | 3 |
| Cosine \| Aug+Smooth | **79.43** | 3 |
| Plateau \| Base (No Aug, No Smooth) | 75.72 | 3 |
| Plateau \| Aug+Smooth | 78.92 | 4 |

Interpretation: augmentation + smoothing consistently improved outcomes; cosine schedule gave the strongest result in this set.

### 7.5 Student 5 — FGSM Robustness
Artifacts:
- `ogrenci5/fgsm_results.json`
- `ogrenci5/robustness_curve_regularization.png`
- `ogrenci5/robustness_curve_optimizer.png`
- `ogrenci5/accuracy_drop_bar_regularization.png`
- `ogrenci5/accuracy_drop_bar_optimizer.png`
- `ogrenci5/adversarial_samples.png`

Optimizer snapshot (Clean and attack-time degradation):

| Model | Clean (%) | epsilon=0.1 (%) |
|---|---:|---:|
| SGD (No Clip) | 44.30 | 14.72 |
| SGD (Clip=1.0) | 36.11 | 22.24 |
| Adam (No Clip) | 77.18 | 14.89 |
| Adam (Clip=1.0) | **77.75** | 14.51 |

Interpretation: clean and robust performance can diverge; clipping may help specific epsilon regimes, especially in SGD settings.

## 8. Reproducibility
Install:
```bash
pip install -r requirements.txt
```

Main training entry:
```bash
python main.py --epochs 10 --optimizer adam --scheduler cosine --l2 0.001
```

Experiment scripts:
```bash
python generate_student1_capacity_report.py
python student2_optimizer_comparison.py
python generate_student3_init_report.py --quick
python generate_student4_report.py
python fgsm_eval.py
```

## 9. Deliverables
- Source code and scripts in repository root
- Technical report (`REPORT.md`)
- Per-student contribution files in `responsibilities/`
- Experiment artifacts in `ogrenci1/`, `ogrenci2/`, `ogrenci3/`, `ogrenci4/`, `ogrenci5/`

## 10. Final Evidence Table (Method → Accuracy → Robustness)
This table summarizes the minimum evidence requested for method comparison: each method is tied to clean performance and, where applicable, FGSM behavior.

| Method / Setting | Primary Metric (Clean) | FGSM Evidence (epsilon=0.1 or curve) | Main Interpretation |
|---|---:|---:|---|
| Baseline (Adam) | 77.96% clean | 14.89% (@ epsilon=0.1) | Strong clean baseline, large adversarial drop |
| L1 (lambda=1e-4) | 67.77% clean | Included in regularization FGSM curves | Promotes sparsity; can reduce overfitting but may lower clean accuracy |
| L2 (lambda=1e-3) | 77.17% clean | Included in regularization FGSM curves | Smooth weight shrinkage with stable clean behavior |
| Dropout 0.2 / 0.5 | 75.73% / 74.60% clean | Included in regularization FGSM curves | Regularization via stochastic masking; cleaner generalization trend |
| BatchNorm | **79.87% clean** | Included in regularization FGSM curves | Best clean score in regularization set, improved optimization stability |
| Label Smoothing 0.1 | 79.09% clean | Included in regularization FGSM curves | Reduces over-confidence; improves calibration-oriented behavior |
| Adv Training (FGSM) | Tracked in FGSM run artifacts | Directly evaluated via FGSM curves | Trains on perturbed samples; robustness-oriented objective |
| Base (SGD, no clip) | 44.30% clean | 14.72% (@ epsilon=0.1) | Lower clean score, different robustness tradeoff |
| Base (SGD, clip=1.0) | 36.11% clean | **22.24%** (@ epsilon=0.1) | Clipping can improve robustness at some epsilon levels despite clean drop |
| Base (Adam, clip=1.0) | **77.75% clean** | 14.51% (@ epsilon=0.1) | Highest clean in optimizer comparison; robustness not always proportional to clean score |

Evidence sources:
- `ogrenci5/fgsm_results.json`
- `ogrenci5/robustness_curve_regularization.png`
- `ogrenci5/robustness_curve_optimizer.png`
- `ogrenci5/accuracy_drop_bar_regularization.png`
- `ogrenci5/accuracy_drop_bar_optimizer.png`

