"""
generate_student1_capacity_report.py - Öğrenci 1 Kapasite Ablation Deneyi

Model kapasitesinin (parametre sayısı) bias-variance dengesine etkisini
sistematik olarak göstermek için Small/Base/Large CNN varyantlarını
aynı eğitim protokolü altında karşılaştırır.
"""

import os
import json
import shutil
import argparse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from data import get_dataloaders
from models import CNN, count_cnn_parameters
from train import train_model
from utils import plot_training_curves


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _subset_loader(loader, ratio):
    """DataLoader'ı belirtilen oranla küçültür."""
    if ratio >= 1.0:
        return loader
    dataset = loader.dataset
    subset_size = max(1, int(len(dataset) * ratio))
    subset = Subset(dataset, list(range(subset_size)))
    return DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        pin_memory=getattr(loader, "pin_memory", False),
    )


def run_capacity_ablation(epochs=10, batch_size=128, subset_ratio=1.0):
    device = get_device()
    os.makedirs("ogrenci1", exist_ok=True)
    trainloader, testloader = get_dataloaders(batch_size=batch_size)
    trainloader = _subset_loader(trainloader, subset_ratio)
    testloader = _subset_loader(testloader, subset_ratio)

    configs = [
        {
            "label": "Small Capacity",
            "base_channels": 8,
            "fc_hidden_dim": 128,
        },
        {
            "label": "Base Capacity",
            "base_channels": 16,
            "fc_hidden_dim": 256,
        },
        {
            "label": "Large Capacity",
            "base_channels": 32,
            "fc_hidden_dim": 512,
        },
    ]

    results = {}
    summary = {}

    for cfg in configs:
        label = cfg["label"]
        print(f"\n{'=' * 55}\nCapacity Experiment: {label}\n{'=' * 55}")

        model = CNN(
            num_classes=10,
            base_channels=cfg["base_channels"],
            fc_hidden_dim=cfg["fc_hidden_dim"],
        )
        param_count = count_cnn_parameters(
            num_classes=10,
            base_channels=cfg["base_channels"],
            fc_hidden_dim=cfg["fc_hidden_dim"],
        )

        history = train_model(
            model=model,
            trainloader=trainloader,
            testloader=testloader,
            epochs=epochs,
            device=device,
            optimizer_name="adam",
            scheduler_name="cosine",
            verbose=True,
        )

        results[label] = history
        best_test = max(history["test_acc"])
        final_train = history["train_acc"][-1]
        final_test = history["test_acc"][-1]
        gap = final_train - final_test

        summary[label] = {
            "base_channels": cfg["base_channels"],
            "fc_hidden_dim": cfg["fc_hidden_dim"],
            "parameter_count": int(param_count),
            "best_test_acc": round(best_test, 2),
            "final_train_acc": round(final_train, 2),
            "final_test_acc": round(final_test, 2),
            "generalization_gap": round(gap, 2),
        }

    plot_training_curves(results, filename="student1_capacity_ablation_curves.png")
    shutil.copyfile(
        "results/student1_capacity_ablation_curves.png",
        "ogrenci1/student1_capacity_ablation_curves.png",
    )

    labels = list(summary.keys())
    gaps = [summary[l]["generalization_gap"] for l in labels]
    colors = ["#4CAF50" if g <= 0 else "#F44336" for g in gaps]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, gaps, color=colors, alpha=0.9)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.title("Generalization Gap by Model Capacity")
    plt.ylabel("Train Acc - Test Acc (%)")
    plt.xticks(rotation=10)

    for bar, value in zip(bars, gaps):
        y = value + 0.1 if value >= 0 else value - 0.6
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{value:.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("ogrenci1/generalization_gap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    out_path = "ogrenci1/capacity_ablation_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {out_path}")
    print("Saved: results/student1_capacity_ablation_curves.png")
    print("Saved: ogrenci1/student1_capacity_ablation_curves.png")
    print("Saved: ogrenci1/generalization_gap_bar.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Student 1 model capacity ablation (Small/Base/Large)."
    )
    parser.add_argument("--epochs", type=int, default=10, help="Epoch count per capacity config")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--subset_ratio",
        type=float,
        default=1.0,
        help="Fraction of train/test data to use (0,1]. Example: 0.2 for fast debug run.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast sanity mode: epochs=3, subset_ratio=0.2, batch_size=256",
    )

    args = parser.parse_args()
    if args.quick:
        run_capacity_ablation(epochs=3, batch_size=256, subset_ratio=0.2)
    else:
        run_capacity_ablation(
            epochs=args.epochs,
            batch_size=args.batch_size,
            subset_ratio=args.subset_ratio,
        )
