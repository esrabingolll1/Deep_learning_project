"""
generate_student4_report.py - Öğrenci 4: Data Augmentation & Label Smoothing Deneyleri

Bu script Data Augmentation ve Label Smoothing'in model başarısı ve 
"aşırı güven" (over-confidence) üzerindeki etkisini test etmek için kullanılır.
"""

import torch
import os
from data import get_dataloaders
from models import CNN
from train import train_model
from utils import plot_training_curves

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def run_student4_experiments(epochs=10, batch_size=128):
    device = get_device()
    print(f"Device: {device}\n")

    # İki farklı dataloader oluşturacağız (Augmentation Açık vs Kapalı)
    trainloader_no_aug, testloader = get_dataloaders(batch_size=batch_size, use_augmentation=False)
    trainloader_aug, _ = get_dataloaders(batch_size=batch_size, use_augmentation=True)

    experiments = [
        {"label": "Base (No Aug., No Smooth)", "use_aug": False, "smoothing": 0.0},
        {"label": "Only Augmentation",         "use_aug": True,  "smoothing": 0.0},
        {"label": "Only Label Smoothing (0.1)","use_aug": False, "smoothing": 0.1},
        {"label": "Augmentation + Smoothing",  "use_aug": True,  "smoothing": 0.1},
    ]

    results = {}

    for exp in experiments:
        label = exp["label"]
        print(f"{'─' * 50}")
        print(f"  Experiment: {label}")
        print(f"{'─' * 50}")

        model = CNN(num_classes=10)
        loader = trainloader_aug if exp["use_aug"] else trainloader_no_aug

        history = train_model(
            model=model,
            trainloader=loader,
            testloader=testloader,
            epochs=epochs,
            device=device,
            label_smoothing=exp["smoothing"],
            verbose=True
        )

        results[label] = history
        print()

    print("Generating Student 4 training curves comparison...")
    plot_training_curves(results, filename="student4_augmentation_smoothing_comparison.png")
    print("\nStudent 4 experiments completed successfully!")

if __name__ == "__main__":
    run_student4_experiments(epochs=10)
