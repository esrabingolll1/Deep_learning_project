"""
visualize_extras.py - Ek Analiz Görselleri

CIFAR-10 veri seti örneklerini ve ağırlık dağılımı histogramlarını
üretir. Bu görseller raporun teorik bölümlerini destekler.
"""

import torch
from data import get_dataloaders, denormalize, CLASSES
from models import CNN
from train import train_model
from utils import plot_cifar10_samples, plot_weight_distribution


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def generate_dataset_samples():
    """CIFAR-10 veri setinden 16 örnek görüntüyü kaydeder."""
    print("Generating CIFAR-10 sample grid...")
    trainloader, _ = get_dataloaders(batch_size=16)
    images, labels = next(iter(trainloader))

    # Doğru denormalizasyon
    images = denormalize(images)

    plot_cifar10_samples(images, labels, CLASSES, filename="cifar10_samples.png")


def generate_weight_histograms():
    """
    Base, L1 ve L2 modellerini 1 epoch eğitip ağırlık dağılımlarını
    histogram olarak kaydeder.
    """
    print("\nTraining quick models for weight analysis (1 epoch each)...")
    device = get_device()
    trainloader, testloader = get_dataloaders(batch_size=256)

    configs = [
        {"label": "Base Model",  "l1": 0.0,   "l2": 0.0},
        {"label": "L1 (λ=5e-3)", "l1": 5e-3,  "l2": 0.0},
        {"label": "L2 (λ=5e-2)", "l1": 0.0,   "l2": 5e-2},
    ]

    weight_dict = {}
    for cfg in configs:
        print(f"  Training: {cfg['label']}...")
        model = CNN(num_classes=10)
        train_model(
            model, trainloader, testloader,
            epochs=1, device=device,
            l1_lambda=cfg["l1"], l2_weight_decay=cfg["l2"],
            verbose=False,
        )
        weight_dict[cfg["label"]] = model.fc1.weight.data.cpu().numpy().flatten()

    print("Generating weight histogram...")
    plot_weight_distribution(weight_dict, filename="weight_histogram_analysis.png")


if __name__ == "__main__":
    generate_dataset_samples()
    generate_weight_histograms()
    print("\nAll extra visualizations completed!")
