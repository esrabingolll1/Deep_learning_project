"""
generate_student2_bonus.py - Öğrenci 2 İçin Bonus Görseller

Bu script, L1 ve L2 düzenlileştirmesinin etkilerini kanıtlamak için
"Ağırlık Seyreklik (Sparsity) Bar Grafiği" ve "Conv1 Filtre Isı Haritası"
görsellerini doğrudan ogrenci2/ klasörüne üretir.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from data import get_dataloaders
from models import CNN
from train import train_model
from utils import COLORS

matplotlib.rcParams.update({
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor':   '#FAFAFA',
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'grid.linestyle':   '--',
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.titleweight': 'bold',
})

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def plot_sparsity_bar(sparsity_dict, save_path):
    """Her modelin sıfıra yakın (abs < 1e-4) ağırlık yüzdesini bar grafiğiyle çizer."""
    fig, ax = plt.subplots(figsize=(8, 5))
    models = list(sparsity_dict.keys())
    values = list(sparsity_dict.values())
    
    bars = ax.bar(models, values, color=[COLORS[0], COLORS[1], COLORS[2]], width=0.5, alpha=0.9)
    
    # Barların üzerine yüzde değerini yazma
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', 
                ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax.set_title("FC1 Layer — Weight Sparsity (Zero Weights %)")
    ax.set_ylabel("Sparsity Percentage (%)")
    ax.set_ylim(0, max(values) + 15 if max(values) > 0 else 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_filter_heatmaps(models_dict, save_path):
    """Base, L1 ve L2 modellerinin Conv1 katmanındaki 16 filtreyi ısı haritası olarak çizer."""
    fig, axes = plt.subplots(3, 16, figsize=(18, 4))
    
    for row_idx, (label, model) in enumerate(models_dict.items()):
        # Conv1 filtrelerini al (shape: 32, 3, 3, 3)
        # Görselleştirmek için ilk kanalı (index 0) ve ilk 16 filtreyi seçiyoruz.
        weights = model.conv1.weight.data.cpu().numpy()[:16, 0, :, :]
        
        # Oksitlenmiş (0'lı) filtreleri net göstermek için global min-max ölçekleme
        vmax = np.max(np.abs(weights))
        vmin = -vmax
        
        for col_idx in range(16):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(weights[col_idx], cmap='coolwarm', vmin=vmin, vmax=vmax)
            ax.axis('off')
            if col_idx == 0:
                ax.set_title(label, loc='left', fontsize=12, fontweight='bold', pad=10)

    fig.suptitle("Conv1 Layer — Filter Activation Heatmaps (Coolwarm: Red=Positive, Blue=Negative, White=Zero)", 
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Bonus görseller için mini eğitim başlatılıyor...")
    device = get_device()
    trainloader, testloader = get_dataloaders(batch_size=256)

    # L1 etkisini net görmek için lambda değerini yüksek tutuyoruz.
    configs = [
        {"label": "Base Model",  "l1": 0.0,   "l2": 0.0},
        {"label": "L1 (λ=1e-2)", "l1": 1e-2,  "l2": 0.0},
        {"label": "L2 (λ=5e-2)", "l1": 0.0,   "l2": 5e-2},
    ]

    trained_models = {}
    sparsity_dict = {}

    for cfg in configs:
        print(f"  Training: {cfg['label']}...")
        model = CNN(num_classes=10)
        train_model(
            model, trainloader, testloader,
            epochs=1, device=device,
            l1_lambda=cfg["l1"], l2_weight_decay=cfg["l2"],
            verbose=False,
        )
        trained_models[cfg["label"]] = model
        
        # Sparsity hesaplama (FC1 katmanı için). Eşik (Threshold) = 1e-4
        weights = model.fc1.weight.data.abs()
        zero_weights = (weights < 1e-4).sum().item()
        total_weights = weights.numel()
        sparsity_dict[cfg["label"]] = 100.0 * zero_weights / total_weights

    os.makedirs("ogrenci2", exist_ok=True)
    
    print("Seyreklik (Sparsity) Grafiği üretiliyor...")
    plot_sparsity_bar(sparsity_dict, "ogrenci2/seyreklik_karsilastirma_grafigi.png")
    
    print("Filtre Isı Haritaları (Heatmaps) üretiliyor...")
    plot_filter_heatmaps(trained_models, "ogrenci2/conv1_filtre_isi_haritasi.png")
    
    print("\nTüm bonus görseller başarıyla oluşturuldu!")

if __name__ == "__main__":
    main()
