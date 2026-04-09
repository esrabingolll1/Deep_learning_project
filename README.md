# Deep Learning Project — CIFAR-10 CNN Analysis

A collaborative deep learning project where five students each investigate a different aspect of training and evaluating a CNN on the CIFAR-10 image classification benchmark.

---

## Team

| Student ID | Role |
|---|---|
| 210911026 | System Architect & Model Capacity Specialist |
| 210911051 | Weight Regularization Specialist (L1/L2) |
| Student 3 | Weight Initialization Specialist (He/Xavier) |
| 210911028 | Data & Label Manipulation Specialist |
| 210911030 | Security Analyst (FGSM Adversarial Attack) |

---

## Dataset

**CIFAR-10** — 60,000 RGB images (32×32 pixels), 10 classes, 6,000 images per class.

| Split | Size |
|---|---|
| Training | 50,000 images |
| Test | 10,000 images |

Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Preprocessing:
- **Training:** RandomCrop(32, padding=4) + RandomHorizontalFlip + Normalize
- **Test:** Normalize only (no augmentation)

---

## Model Architecture

A 4-layer CNN defined in `models.py`, parameterized for capacity control:

```
Input (3×32×32)
  → Conv2d(3 → base_channels, 3×3, pad=1) → BN → ReLU
  → Conv2d(base_channels → base_channels×2, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)
  → Conv2d(base_channels×2 → base_channels×4, 3×3, pad=1) → BN → ReLU
  → Conv2d(base_channels×4 → base_channels×4, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)
  → Flatten
  → Linear(base_channels×4 × 8×8 → fc_hidden_dim) → ReLU → Dropout
  → Linear(fc_hidden_dim → 10)
```

| Parameter | Default | Description |
|---|---|---|
| `base_channels` | 16 | Controls filter count in conv layers |
| `fc_hidden_dim` | 256 | Hidden size of fully connected layer |
| `dropout_rate` | 0.0 | Dropout probability (0 = disabled) |
| `use_batchnorm` | False | Enables BatchNorm after each conv layer |
| `init_scheme` | default | Weight initialization: `default`, `he`, `xavier` |

**Capacity configurations used in experiments:**

| Config | base_channels | fc_hidden_dim | Parameters |
|---|---|---|---|
| Small | 8 | 128 | 278,842 |
| Base | 16 | 256 | 1,111,914 |
| Large | 32 | 512 | 4,440,778 |

---

## Training

All experiments use the same centralized training loop in `train.py`.

### Default Training Parameters

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 128 |
| Epochs | 10 |
| LR Scheduler | CosineAnnealingLR |
| Early stopping | patience = 5 |
| L1 lambda | 0.0 (disabled) |
| L2 weight decay | 0.0 (disabled) |
| Label smoothing | 0.0 (disabled) |
| Gradient clipping | disabled |

### How Training Works

Each epoch:
1. **Forward pass** — inputs are fed through the CNN, logits are produced
2. **Loss computation** — Cross-Entropy Loss (+ optional L1/L2 penalty)
3. **Backward pass** — gradients are computed via backpropagation
4. **Optimizer step** — weights are updated (Adam or SGD+Momentum)
5. **Scheduler step** — learning rate is adjusted based on the scheduler
6. **Evaluation** — model is evaluated on the test set after every epoch

Early stopping monitors validation loss and restores the best weights if no improvement is seen for 5 consecutive epochs.

### Regularization

**L1** — manually added to the loss at each step:
```
Loss = CrossEntropy + λ × Σ|w|
```

**L2** — applied through the optimizer's `weight_decay` parameter:
```
Loss = CrossEntropy + λ × Σw²
```

### Supported Options (via `main.py`)

```
--epochs          int     Number of training epochs (default: 10)
--batch_size      int     Batch size (default: 128)
--lr              float   Learning rate (default: 0.001)
--optimizer       str     adam | sgd (default: adam)
--scheduler       str     cosine | plateau | none (default: cosine)
--l1              float   L1 regularization lambda (default: 0.0)
--l2              float   L2 weight decay (default: 0.0)
--label_smoothing float   Label smoothing factor (default: 0.0)
--dropout_rate    float   Dropout rate (default: 0.0)
--use_batchnorm           Enable BatchNorm
--init_scheme     str     default | he | xavier
--base_channels   int     Conv filter count (default: 16)
--fc_hidden_dim   int     FC hidden size (default: 256)
--adv_train               Enable adversarial training (FGSM)
--grad_clip_norm  float   Gradient clipping max norm (default: 0.0)
--no_augmentation         Disable data augmentation
```

---

## Experiments

### Student 1 — Model Capacity (`generate_student1_capacity_report.py`)
Trains Small / Base / Large models with all other settings fixed. Measures how capacity affects test accuracy and the generalization gap (train acc − test acc).

### Student 2 — Regularization & Optimizers (`student2_optimizer_comparison.py`)
Trains 6 combinations: {None, L1, L2} × {Adam, SGD+Momentum}. Produces weight histograms, sparsity analysis (~80% of weights zeroed by L1), accuracy curves, and convergence speed plots.

### Student 3 — Weight Initialization (`generate_student3_init_report.py`)
Compares PyTorch default, He (Kaiming), and Xavier initialization on the same model. Measures the effect on convergence speed and final accuracy.

### Student 4 — Augmentation & Label Smoothing (`generate_student4_report.py`)
Tests {No Aug, Aug} × {No Smooth, Smooth} × {Cosine, Plateau scheduler}. Best result: **79.43% test accuracy** with Cosine + Augmentation + Label Smoothing.

### Student 5 — Adversarial Robustness (`fgsm_eval.py`)
Applies FGSM attack at 8 epsilon levels (0.0 → 0.3) to models trained with different optimizers and gradient clipping. FGSM perturbs each pixel in the direction that maximizes loss:
```
x_adv = clip( x + ε × sign(∇_x L) )
```
Best clean accuracy: **77.75%** (Adam + Clip=1.0). All models drop significantly under attack.

---

## Setup & Running

```bash
pip install -r requirements.txt

# Train a single model with custom settings
python main.py --epochs 10 --optimizer adam --l2 0.001

# Run individual experiments
python generate_student1_capacity_report.py
python student2_optimizer_comparison.py --epochs 10
python generate_student3_init_report.py
python generate_student4_report.py
python fgsm_eval.py
```

Output plots and JSON results are saved to `ogrenci1/`, `ogrenci2/`, `ogrenci4/`, `ogrenci5/`.

---

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- numpy
