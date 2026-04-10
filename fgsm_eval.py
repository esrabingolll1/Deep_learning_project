"""FGSM değerlendirmesi: tüm düzenleme modelleri + SGD vs Adam (temel model)."""

import json
import os
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from data import get_dataloaders, denormalize, CIFAR10_MEAN, CIFAR10_STD
from models import CNN
from train import train_model

matplotlib.rcParams.update({
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor': '#FAFAFA',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'legend.fontsize': 9,
    'lines.linewidth': 2.0,
    'lines.markersize': 5,
})

OUTPUT_DIR = 'ogrenci5'
EPOCHS = 10
EPSILONS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]

REGULARIZATION_CONFIGS = [
    {
        'label': 'Base (Adam)',
        'dropout_rate': 0.0, 'use_batchnorm': False,
        'l1_lambda': 0.0, 'l2_weight_decay': 0.0,
        'label_smoothing': 0.0, 'adv_train': False,
        'use_augmentation': True, 'optimizer_name': 'adam',
    },
    {
        'label': 'L1 (λ=1e-4)',
        'dropout_rate': 0.0, 'use_batchnorm': False,
        'l1_lambda': 1e-4, 'l2_weight_decay': 0.0,
        'label_smoothing': 0.0, 'adv_train': False,
        'use_augmentation': True, 'optimizer_name': 'adam',
    },
    {
        'label': 'L2 (λ=1e-3)',
        'dropout_rate': 0.0, 'use_batchnorm': False,
        'l1_lambda': 0.0, 'l2_weight_decay': 1e-3,
        'label_smoothing': 0.0, 'adv_train': False,
        'use_augmentation': True, 'optimizer_name': 'adam',
    },
    {
        'label': 'Dropout 0.2',
        'dropout_rate': 0.2, 'use_batchnorm': False,
        'l1_lambda': 0.0, 'l2_weight_decay': 0.0,
        'label_smoothing': 0.0, 'adv_train': False,
        'use_augmentation': True, 'optimizer_name': 'adam',
    },
    {
        'label': 'Dropout 0.5',
        'dropout_rate': 0.5, 'use_batchnorm': False,
        'l1_lambda': 0.0, 'l2_weight_decay': 0.0,
        'label_smoothing': 0.0, 'adv_train': False,
        'use_augmentation': True, 'optimizer_name': 'adam',
    },
    {
        'label': 'BatchNorm',
        'dropout_rate': 0.0, 'use_batchnorm': True,
        'l1_lambda': 0.0, 'l2_weight_decay': 0.0,
        'label_smoothing': 0.0, 'adv_train': False,
        'use_augmentation': True, 'optimizer_name': 'adam',
    },
    {
        'label': 'Label Smooth 0.1',
        'dropout_rate': 0.0, 'use_batchnorm': False,
        'l1_lambda': 0.0, 'l2_weight_decay': 0.0,
        'label_smoothing': 0.1, 'adv_train': False,
        'use_augmentation': True, 'optimizer_name': 'adam',
    },
    {
        'label': 'Adv Train (FGSM)',
        'dropout_rate': 0.0, 'use_batchnorm': False,
        'l1_lambda': 0.0, 'l2_weight_decay': 0.0,
        'label_smoothing': 0.0, 'adv_train': True,
        'use_augmentation': True, 'optimizer_name': 'adam',
    },
]

OPTIMIZER_BASE_SGD = {
    'label': 'Base (SGD)',
    'dropout_rate': 0.0, 'use_batchnorm': False,
    'l1_lambda': 0.0, 'l2_weight_decay': 0.0,
    'label_smoothing': 0.0, 'adv_train': False,
    'use_augmentation': True, 'optimizer_name': 'sgd',
}


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fgsm_attack(model, images, labels, epsilon, device):
    criterion = nn.CrossEntropyLoss()
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    images.requires_grad_(True)
    outputs = model(images)
    loss = criterion(outputs, labels)
    model.zero_grad()
    loss.backward()
    sign_grad = images.grad.data.sign()
    adv_images = images + epsilon * sign_grad
    mean = torch.tensor(CIFAR10_MEAN, device=device, dtype=adv_images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=device, dtype=adv_images.dtype).view(1, 3, 1, 1)
    lower = (0.0 - mean) / std
    upper = (1.0 - mean) / std
    adv_images = torch.max(torch.min(adv_images, upper), lower)
    return adv_images.detach()


def evaluate_under_fgsm(model, testloader, epsilon, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        if epsilon > 0.0:
            adv_images = fgsm_attack(model, images, labels, epsilon, device)
        else:
            adv_images = images
        with torch.no_grad():
            outputs = model(adv_images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def train_and_store(cfg, trainloader, testloader, device):
    model = CNN(
        num_classes=10,
        dropout_rate=cfg['dropout_rate'],
        use_batchnorm=cfg['use_batchnorm'],
    )
    train_model(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        epochs=EPOCHS,
        device=device,
        lr=0.001,
        l1_lambda=cfg['l1_lambda'],
        l2_weight_decay=cfg['l2_weight_decay'],
        label_smoothing=cfg['label_smoothing'],
        adv_train=cfg['adv_train'],
        optimizer_name=cfg['optimizer_name'],
        scheduler_name='cosine',
        grad_clip_norm=0.0,
        verbose=True,
    )
    return model


def plot_curves(results, epsilons, filename, title):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("tab10")
    for i, (label, accs) in enumerate(results.items()):
        ax.plot(epsilons, accs, label=label, color=cmap(i % 10), marker='o', linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel('Epsilon ε')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_xticks(epsilons)
    ax.legend(framealpha=0.92, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


def plot_bar_compare(results, epsilons, epsilon_value, filename, title):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        idx = epsilons.index(epsilon_value)
    except ValueError:
        idx = 4
    labels = list(results.keys())
    clean_accs = [results[l][0] for l in labels]
    adv_accs = [results[l][idx] for l in labels]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.9), 6))
    ax.bar(x - width / 2, clean_accs, width, label='Temiz (ε=0)', alpha=0.85)
    ax.bar(x + width / 2, adv_accs, width, label=f'FGSM (ε={epsilon_value})', alpha=0.55, hatch='//')
    ax.set_title(title)
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


def plot_adversarial_samples(model, testloader, epsilons_to_show, device, filename='adversarial_samples.png'):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.eval()
    images, labels = next(iter(testloader))
    images, labels = images[:8].to(device), labels[:8].to(device)
    n_eps = len(epsilons_to_show)
    fig, axes = plt.subplots(n_eps, 8, figsize=(16, n_eps * 2.2))
    if n_eps == 1:
        axes = np.array([axes])
    for row_idx, eps in enumerate(epsilons_to_show):
        if eps > 0:
            adv = fgsm_attack(model, images, labels, eps, device)
        else:
            adv = images.clone()
        for col_idx in range(8):
            ax = axes[row_idx, col_idx]
            img = denormalize(adv[col_idx].cpu()).permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.axis('off')
            if col_idx == 0:
                ax.set_title(f'ε={eps}', loc='left', fontsize=11, fontweight='bold', pad=6)
    fig.suptitle('FGSM — örnek görüntüler', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


def run_fgsm_all_models(trained_models, testloader, device, epsilons):
    results = {}
    for label, model in trained_models.items():
        print(f'\n  FGSM: {label}')
        accs = []
        for eps in epsilons:
            acc = evaluate_under_fgsm(model, testloader, eps, device)
            accs.append(acc)
            tag = 'CLEAN' if eps == 0 else f'ε={eps}'
            print(f'    {tag:<12} → {acc:.2f}%')
        results[label] = accs
    return results


def main():
    set_seed(42)
    device = get_device()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print('=' * 60)
    print('  FGSM — düzenleme modelleri + Base (SGD)')
    print(f'  Device: {device}')
    print('=' * 60)

    trained_models = {}

    for cfg in REGULARIZATION_CONFIGS:
        print('\n' + '-' * 60)
        print(f'  Eğitim: {cfg["label"]}')
        print('-' * 60)
        trainloader, testloader = get_dataloaders(
            batch_size=128, use_augmentation=cfg['use_augmentation'])
        model = train_and_store(cfg, trainloader, testloader, device)
        trained_models[cfg['label']] = model

    print('\n' + '-' * 60)
    print(f'  Eğitim: {OPTIMIZER_BASE_SGD["label"]}')
    print('-' * 60)
    trainloader, testloader = get_dataloaders(
        batch_size=128, use_augmentation=OPTIMIZER_BASE_SGD['use_augmentation'])
    model_sgd = train_and_store(OPTIMIZER_BASE_SGD, trainloader, testloader, device)
    trained_models[OPTIMIZER_BASE_SGD['label']] = model_sgd

    _, testloader_eval = get_dataloaders(batch_size=128, use_augmentation=False)
    print('\n' + '=' * 60)
    print('  FGSM değerlendirme (tüm modeller)')
    print('=' * 60)
    all_results = run_fgsm_all_models(trained_models, testloader_eval, device, EPSILONS)

    reg_labels = [c['label'] for c in REGULARIZATION_CONFIGS]
    reg_results = {k: all_results[k] for k in reg_labels if k in all_results}
    opt_results = {
        'Base (Adam)': all_results['Base (Adam)'],
        'Base (SGD)': all_results['Base (SGD)'],
    }

    print('\nGrafikler...')
    plot_curves(reg_results, EPSILONS, 'robustness_curve_regularization.png',
                'FGSM — düzenleme yöntemleri (Adam)')
    plot_curves(opt_results, EPSILONS, 'robustness_curve_optimizer.png',
                'FGSM — SGD vs Adam (temel model)')
    plot_bar_compare(reg_results, EPSILONS, 0.1, 'accuracy_drop_bar_regularization.png',
                     'Temiz vs FGSM (ε=0.1) — düzenleme')
    plot_bar_compare(opt_results, EPSILONS, 0.1, 'accuracy_drop_bar_optimizer.png',
                     'Temiz vs FGSM (ε=0.1) — optimizer')
    plot_adversarial_samples(
        trained_models['Base (Adam)'], testloader_eval, [0.0, 0.05, 0.2], device)

    json_path = os.path.join(OUTPUT_DIR, 'fgsm_results.json')
    serializable = {k: [float(x) for x in v] for k, v in all_results.items()}
    serializable['epsilons'] = EPSILONS
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f'  → {json_path}')

    shutil.copyfile(
        os.path.join(OUTPUT_DIR, 'robustness_curve_regularization.png'),
        os.path.join(OUTPUT_DIR, 'robustness_curve.png'),
    )
    shutil.copyfile(
        os.path.join(OUTPUT_DIR, 'accuracy_drop_bar_regularization.png'),
        os.path.join(OUTPUT_DIR, 'accuracy_drop_bar.png'),
    )

    print('\n' + '=' * 60)
    print('  Özet (Clean / ε=0.1 / ε=0.2)')
    print('=' * 60)
    for label, accs in all_results.items():
        print(f'  {label:<22} {accs[0]:6.2f}%  {accs[4]:6.2f}%  {accs[6]:6.2f}%')
    print(f'\n  Çıktılar: {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
