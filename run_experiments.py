"""
run_experiments.py - Düzenlileştirme Karşılaştırma Deneyleri

Base Model, L1 ve L2 düzenlileştirme yöntemlerini art arda eğitir
ve sonuçları karşılaştırmalı grafiklerle kaydeder.
"""

import torch
from data import get_dataloaders
from models import CNN
from train import train_model
from utils import plot_training_curves, plot_weight_distribution


def get_device():
    """Kullanılabilir en hızlı cihazı seçer."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_regularization_comparison(epochs=10, batch_size=128):
    """
    Farklı düzenlileştirme konfigürasyonlarıyla modeli eğitir ve
    sonuçları karşılaştırmalı grafikler olarak kaydeder.
    """
    device = get_device()
    print(f"Device: {device}\n")

    trainloader, testloader = get_dataloaders(batch_size=batch_size)

    # Deney konfigürasyonları
    configs = [
        {"label": "Base Model",         "l1": 0.0,    "l2": 0.0},
        {"label": "L1 (λ=1e-4)",        "l1": 1e-4,   "l2": 0.0},
        {"label": "L2 / Weight Decay (λ=1e-3)", "l1": 0.0, "l2": 1e-3},
        {"label": "L2 / Weight Decay (λ=1e-2)", "l1": 0.0, "l2": 1e-2},
    ]

    results = {}
    trained_models = {}

    for cfg in configs:
        label = cfg["label"]
        print(f"{'─' * 50}")
        print(f"  Experiment: {label}")
        print(f"{'─' * 50}")

        model = CNN(num_classes=10)

        history = train_model(
            model=model,
            trainloader=trainloader,
            testloader=testloader,
            epochs=epochs,
            device=device,
            l1_lambda=cfg["l1"],
            l2_weight_decay=cfg["l2"],
        )

        results[label] = history
        trained_models[label] = model
        print()

    # ─── Grafik 1: Eğitim eğrileri karşılaştırması ───
    print("Generating training curves comparison...")
    plot_training_curves(results, filename="regularization_comparison.png")

    # ─── Grafik 2: Ağırlık dağılımı histogramı ───
    print("Generating weight distribution histogram...")
    weight_dict = {}
    for label, model in trained_models.items():
        weights = model.fc1.weight.data.cpu().numpy().flatten()
        weight_dict[label] = weights

    plot_weight_distribution(weight_dict, filename="weight_distribution.png")

    print("\nAll experiments completed successfully!")


if __name__ == "__main__":
    run_regularization_comparison(epochs=10)
