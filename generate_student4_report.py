"""
generate_student4_report.py - Öğrenci 4: Week 5 Optimization Deneyleri

Data Augmentation + Label Smoothing kombinasyonunu farklı learning-rate
scheduler ayarlarıyla (CosineAnnealingLR / ReduceLROnPlateau) karşılaştırır.
Yakınsama hızını ve final doğruluğu metriklerini raporlar.
"""

import torch
import os
import json
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
    os.makedirs("ogrenci4", exist_ok=True)

    # İki farklı dataloader oluşturacağız (Augmentation Açık vs Kapalı)
    trainloader_no_aug, testloader = get_dataloaders(batch_size=batch_size, use_augmentation=False)
    trainloader_aug, _ = get_dataloaders(batch_size=batch_size, use_augmentation=True)

    experiments = [
        {
            "label": "Cosine | Base (No Aug, No Smooth)",
            "use_aug": False,
            "smoothing": 0.0,
            "scheduler": "cosine",
        },
        {
            "label": "Cosine | Aug+Smooth",
            "use_aug": True,
            "smoothing": 0.1,
            "scheduler": "cosine",
        },
        {
            "label": "Plateau | Base (No Aug, No Smooth)",
            "use_aug": False,
            "smoothing": 0.0,
            "scheduler": "plateau",
        },
        {
            "label": "Plateau | Aug+Smooth",
            "use_aug": True,
            "smoothing": 0.1,
            "scheduler": "plateau",
        },
    ]

    results = {}
    analysis = {}

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
            scheduler_name=exp["scheduler"],
            verbose=True
        )

        results[label] = history
        test_curve = history["test_acc"]
        best_test = max(test_curve)
        final_test = test_curve[-1]
        first_70_epoch = next((i + 1 for i, v in enumerate(test_curve) if v >= 70.0), None)
        analysis[label] = {
            "best_test_acc": round(best_test, 2),
            "final_test_acc": round(final_test, 2),
            "epoch_reach_70_acc": first_70_epoch,
            "epochs_ran": len(test_curve),
        }
        print()

    print("Generating Student 4 training curves comparison...")
    plot_training_curves(results, filename="student4_scheduler_aug_smooth_comparison.png")

    analysis_path = "ogrenci4/week5_scheduler_analysis.json"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"Saved analysis summary: {analysis_path}")

    print("\nStudent 4 experiments completed successfully!")

if __name__ == "__main__":
    run_student4_experiments(epochs=10)
