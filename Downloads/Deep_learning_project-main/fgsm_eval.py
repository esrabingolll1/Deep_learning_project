"""
fgsm_eval.py - Öğrenci 5: Week 5 Optimization & Güvenlik Analizi

FGSM (Fast Gradient Sign Method) saldırısını farklı optimizer
senaryolarıyla (SGD vs Adam) eğitilmiş modellere uygular.
Ayrıca gradient clipping (max-norm) etkisini robustness açısından kıyaslar.

Çalıştırma:
    python fgsm_eval.py

Çıktılar (ogrenci5/ klasörüne):
    - robustness_curve.png        → ε vs test doğruluğu eğrisi
    - adversarial_samples.png     → temiz / düşük-ε / yüksek-ε görüntü karşılaştırması
    - accuracy_drop_bar.png       → ε=0.1'de doğruluk düşüş bar grafiği
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data import get_dataloaders, denormalize, CLASSES, CIFAR10_MEAN, CIFAR10_STD
from models import CNN
from train import train_model

# ─── Stil Ayarları ─────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor':   '#FAFAFA',
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'grid.linestyle':   '--',
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.titleweight': 'bold',
    'legend.fontsize':  10,
    'lines.linewidth':  2.2,
    'lines.markersize': 6,
})

COLORS     = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']
OUTPUT_DIR = 'ogrenci5'
EPOCHS     = 10


# ─── Yardımcı Fonksiyonlar ──────────────────────────────────────────────────────

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


# ─── FGSM Saldırı Fonksiyonu ───────────────────────────────────────────────────

def fgsm_attack(model, images, labels, epsilon, device):
    """
    Fast Gradient Sign Method (FGSM) saldırısı uygular.

    Formül: x_adv = clip(x + ε × sign(∇_x L(θ, x, y)), 0, 1)

    Args:
        model:   Saldırının uygulanacağı eğitilmiş model.
        images:  Temiz giriş görüntüleri tensörü.
        labels:  Gerçek etiketler.
        epsilon: Pertürbasyon büyüklüğü (saldırı şiddeti).
        device:  Hesaplama cihazı.

    Returns:
        adv_images: Saldırı uygulanmış görüntüler.
    """
    criterion = nn.CrossEntropyLoss()

    images  = images.clone().detach().to(device)
    labels  = labels.clone().detach().to(device)
    images.requires_grad_(True)

    outputs = model(images)
    loss    = criterion(outputs, labels)
    model.zero_grad()
    loss.backward()

    # Gradyanın işaretine göre pertürbasyon uygula
    sign_grad   = images.grad.data.sign()
    adv_images  = images + epsilon * sign_grad

    # Girişler normalize uzayda olduğu için clamp aralığını da
    # kanal bazında normalize uzaya dönüştürerek uygula.
    mean = torch.tensor(CIFAR10_MEAN, device=device, dtype=adv_images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=device, dtype=adv_images.dtype).view(1, 3, 1, 1)
    lower = (0.0 - mean) / std
    upper = (1.0 - mean) / std
    adv_images = torch.max(torch.min(adv_images, upper), lower)

    return adv_images.detach()


# ─── Model Değerlendirme ───────────────────────────────────────────────────────

def evaluate_under_fgsm(model, testloader, epsilon, device):
    """
    Tüm test seti üzerinde FGSM saldırısı uygulayıp doğruluğu döndürür.

    Args:
        epsilon = 0.0 → temiz (clean) doğruluk.
    """
    model.eval()
    correct = 0
    total   = 0

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        if epsilon > 0.0:
            adv_images = fgsm_attack(model, images, labels, epsilon, device)
        else:
            adv_images = images

        with torch.no_grad():
            outputs   = model(adv_images)
            _, preds  = outputs.max(1)
            correct  += preds.eq(labels).sum().item()
            total    += labels.size(0)

    return 100.0 * correct / total


# ─── Grafik Fonksiyonları ──────────────────────────────────────────────────────

def plot_robustness_curve(results, epsilons, filename='robustness_curve.png'):
    """
    Her model için ε değişimine göre test doğruluğunu çizer.
    X = epsilon, Y = Accuracy (%)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (label, accs) in enumerate(results.items()):
        ax.plot(epsilons, accs, label=label, color=COLORS[i % len(COLORS)],
                marker='o', linewidth=2.2)

    ax.set_title('FGSM Robustness Curve — Accuracy vs Epsilon')
    ax.set_xlabel('Epsilon (ε) — Saldırı Şiddeti')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_xticks(epsilons)
    ax.legend(framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  → {save_path}')


def plot_accuracy_drop_bar(results, epsilon_idx, epsilons, filename='accuracy_drop_bar.png'):
    """
    Belirli bir epsilon değerinde clean vs saldırı altı doğruluk düşüşünü gösterir.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    eps_val = epsilons[epsilon_idx]

    labels     = list(results.keys())
    clean_accs = [results[l][0]          for l in labels]   # ε=0
    adv_accs   = [results[l][epsilon_idx] for l in labels]  # seçilen epsilon

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, clean_accs, width, label='Temiz (ε=0)',
                   color=[COLORS[i % len(COLORS)] for i in range(len(labels))],
                   alpha=0.85)
    bars2 = ax.bar(x + width/2, adv_accs,   width, label=f'FGSM (ε={eps_val})',
                   color=[COLORS[i % len(COLORS)] for i in range(len(labels))],
                   alpha=0.45, hatch='//')

    # Değerleri barların üstüne yaz
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title(f'Clean vs FGSM Accuracy (ε = {eps_val})')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha='right')
    ax.legend(framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  → {save_path}')


def plot_adversarial_samples(model, testloader, epsilons_to_show, device,
                              filename='adversarial_samples.png'):
    """
    3 epsilon seviyesinde (0, düşük, yüksek) aynı görüntülerin
    nasıl değiştiğini yan yana gösterir.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.eval()

    images, labels = next(iter(testloader))
    images, labels = images[:8].to(device), labels[:8].to(device)

    n_eps   = len(epsilons_to_show)
    fig, axes = plt.subplots(n_eps, 8, figsize=(16, n_eps * 2.2))

    for row_idx, eps in enumerate(epsilons_to_show):
        if eps > 0:
            adv = fgsm_attack(model, images, labels, eps, device)
        else:
            adv = images.clone()

        for col_idx in range(8):
            ax  = axes[row_idx, col_idx]
            img = denormalize(adv[col_idx].cpu()).permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.axis('off')
            if col_idx == 0:
                ax.set_title(f'ε={eps}', loc='left', fontsize=11,
                             fontweight='bold', pad=6)

    fig.suptitle('FGSM Adversarial Examples — Base Model\n'
                 '(Her satır farklı epsilon, her sütun farklı görüntü)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  → {save_path}')


# ─── Ana Akış ─────────────────────────────────────────────────────────────────

def main():
    set_seed(42)
    device = get_device()
    print('=' * 55)
    print(f'  Öğrenci 5 — FGSM Saldırı & Güvenlik Analizi')
    print(f'  Device: {device}')
    print('=' * 55 + '\n')

    trainloader, testloader = get_dataloaders(batch_size=128)

    # ── Model Konfigürasyonları (Week 5: Optimizer + Grad Clipping) ────────
    configs = [
        {'label': 'SGD (No Clip)',    'optimizer': 'sgd',  'clip': 0.0},
        {'label': 'SGD (Clip=1.0)',   'optimizer': 'sgd',  'clip': 1.0},
        {'label': 'Adam (No Clip)',   'optimizer': 'adam', 'clip': 0.0},
        {'label': 'Adam (Clip=1.0)',  'optimizer': 'adam', 'clip': 1.0},
    ]

    trained_models = {}

    for cfg in configs:
        sep = '-' * 55
        print(sep)
        print(f'  Egitim: {cfg["label"]}')
        print(sep)
        model = CNN(num_classes=10)
        train_model(
            model=model,
            trainloader=trainloader,
            testloader=testloader,
            epochs=EPOCHS,
            device=device,
            optimizer_name=cfg['optimizer'],
            scheduler_name='cosine',
            grad_clip_norm=cfg['clip'],
        )
        trained_models[cfg['label']] = model
        print()

    # ── FGSM Değerlendirme ───────────────────────────────────────────────────
    epsilons = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
    results  = {}

    print('='*55)
    print('  FGSM Saldırı Değerlendirmesi Başlıyor...')
    print('='*55)

    for label, model in trained_models.items():
        print(f'\n  Model: {label}')
        accs = []
        for eps in epsilons:
            acc = evaluate_under_fgsm(model, testloader, eps, device)
            accs.append(acc)
            tag = 'CLEAN' if eps == 0 else f'ε={eps}'
            print(f'    {tag:<12} → {acc:.2f}%')
        results[label] = accs

    # ── Grafik Üretimi ───────────────────────────────────────────────────────
    sep2 = '-' * 55
    print('\n' + sep2)
    print('  Grafikler olusturuluyor...')
    print(sep2)

    plot_robustness_curve(results, epsilons)

    # ε=0.1 index = 4
    plot_accuracy_drop_bar(results, epsilon_idx=4, epsilons=epsilons)

    # Adversarial görüntüler referans senaryo ile gösterilir
    plot_adversarial_samples(
        model=trained_models['Adam (No Clip)'],
        testloader=testloader,
        epsilons_to_show=[0.0, 0.05, 0.2],
        device=device,
    )

    print('\n' + '=' * 55)
    print('  Tüm analizler tamamlandı!')
    print(f'  Çıktılar: {OUTPUT_DIR}/ klasöründe')
    print('=' * 55)

    # ── Özet Tablo ───────────────────────────────────────────────────────────
    print(f'\n  {"Model":<25} {"Clean":>8} {"ε=0.05":>8} {"ε=0.1":>8} {"ε=0.2":>8}')
    print(f'  {"─"*57}')
    for label, accs in results.items():
        print(f'  {label:<25} {accs[0]:>7.2f}% {accs[3]:>7.2f}% {accs[4]:>7.2f}% {accs[6]:>7.2f}%')


if __name__ == '__main__':
    main()
