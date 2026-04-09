"""
generate_student3_init_report.py - Öğrenci 3 He/Xavier Initialization Deneyi

Dropout + BatchNorm tabanında farklı ilklendirme şemalarının (default/he/xavier)
öğrenme dinamiğine etkisini karşılaştırır.
"""

import argparse
import json
import os
import shutil
import torch
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


def run_student3_init_experiments(epochs=8, batch_size=128, subset_ratio=1.0):
    device = get_device()
    os.makedirs("ogrenci3", exist_ok=True)

    trainloader, testloader = get_dataloaders(batch_size=batch_size, use_augmentation=True)
    trainloader = _subset_loader(trainloader, subset_ratio)
    testloader = _subset_loader(testloader, subset_ratio)

    configs = [
        {"label": "Default Init + Dropout+BN", "init_scheme": "default"},
        {"label": "He Init + Dropout+BN", "init_scheme": "he"},
        {"label": "Xavier Init + Dropout+BN", "init_scheme": "xavier"},
    ]

    results = {}
    summary = {}

    for cfg in configs:
        print(f"\n{'=' * 55}\nExperiment: {cfg['label']}\n{'=' * 55}")
        model = CNN(
            num_classes=10,
            dropout_rate=0.5,
            use_batchnorm=True,
            init_scheme=cfg["init_scheme"],
        )
        params = count_cnn_parameters(
            num_classes=10,
            dropout_rate=0.5,
            use_batchnorm=True,
            init_scheme=cfg["init_scheme"],
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
        results[cfg["label"]] = history
        summary[cfg["label"]] = {
            "init_scheme": cfg["init_scheme"],
            "parameter_count": int(params),
            "best_test_acc": round(max(history["test_acc"]), 2),
            "final_train_acc": round(history["train_acc"][-1], 2),
            "final_test_acc": round(history["test_acc"][-1], 2),
            "generalization_gap": round(history["train_acc"][-1] - history["test_acc"][-1], 2),
            "epochs_ran": len(history["test_acc"]),
        }

    plot_training_curves(results, filename="student3_init_comparison_curves.png")
    shutil.copyfile(
        "results/student3_init_comparison_curves.png",
        "ogrenci3/student3_init_comparison_curves.png",
    )

    summary_path = "ogrenci3/student3_init_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {summary_path}")
    print("Saved: results/student3_init_comparison_curves.png")
    print("Saved: ogrenci3/student3_init_comparison_curves.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Student 3 init ablation on Dropout+BN")
    parser.add_argument("--epochs", type=int, default=8, help="Epoch count")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--subset_ratio", type=float, default=1.0, help="Data fraction (0,1]")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast mode: epochs=4, batch_size=256, subset_ratio=0.4",
    )
    args = parser.parse_args()

    if args.quick:
        run_student3_init_experiments(epochs=4, batch_size=256, subset_ratio=0.4)
    else:
        run_student3_init_experiments(
            epochs=args.epochs,
            batch_size=args.batch_size,
            subset_ratio=args.subset_ratio,
        )
