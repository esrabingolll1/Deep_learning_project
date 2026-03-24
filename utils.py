"""
utils.py - Görselleştirme Araçları

Eğitim sonuçlarını ve model analizlerini profesyonel kalitede
görselleştiren yardımcı fonksiyonları içerir.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

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
    'lines.linewidth':  2.0,
    'lines.markersize': 5,
})

RESULTS_DIR = 'results'
COLORS = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4']


def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_training_curves(experiments, filename="training_curves.png"):
    """
    Birden fazla deneyin eğitim kaybı, eğitim doğruluğu ve test doğruluğunu
    yan yana 3 panelde çizer.

    Args:
        experiments: {label: history_dict} sözlüğü.
        filename: Kaydedilecek dosya adı.
    """
    _ensure_results_dir()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (label, history) in enumerate(experiments.items()):
        color = COLORS[i % len(COLORS)]
        epochs = range(1, len(history['train_loss']) + 1)

        axes[0].plot(epochs, history['train_loss'], label=label, color=color, marker='o')
        axes[1].plot(epochs, history['train_acc'],  label=label, color=color, marker='s')
        axes[2].plot(epochs, history['test_acc'],   label=label, color=color, marker='^')

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")

    axes[1].set_title("Training Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")

    axes[2].set_title("Validation Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy (%)")

    for ax in axes:
        ax.legend(framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle("Regularization Comparison", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


def plot_weight_distribution(weight_dict, filename="weight_distribution.png"):
    """
    Farklı modellerin ağırlık dağılımlarını üst üste binen histogramlar
    olarak çizer.

    Args:
        weight_dict: {label: numpy_array} sözlüğü.
        filename: Kaydedilecek dosya adı.
    """
    _ensure_results_dir()

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (label, weights) in enumerate(weight_dict.items()):
        color = COLORS[i % len(COLORS)]
        ax.hist(weights, bins=120, alpha=0.55, label=label,
                color=color, range=(-0.15, 0.15), edgecolor='white', linewidth=0.3)

    ax.set_title("FC1 Layer — Weight Distribution")
    ax.set_xlabel("Weight Value")
    ax.set_ylabel("Frequency")
    ax.legend(framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


def plot_cifar10_samples(images, labels, classes, filename="cifar10_samples.png"):
    """
    CIFAR-10 veri setinden örnek görüntüleri etiketleriyle birlikte gösterir.

    Args:
        images: Denormalize edilmiş görüntü tensörü.
        labels: Etiket tensörü.
        classes: Sınıf isimleri tuple'ı.
        filename: Kaydedilecek dosya adı.
    """
    _ensure_results_dir()

    fig, axes = plt.subplots(2, 8, figsize=(14, 4))

    for idx, ax in enumerate(axes.flat):
        img = images[idx].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(classes[labels[idx]], fontsize=9)
        ax.axis('off')

    fig.suptitle("CIFAR-10 Dataset — Sample Images", fontsize=13, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")
