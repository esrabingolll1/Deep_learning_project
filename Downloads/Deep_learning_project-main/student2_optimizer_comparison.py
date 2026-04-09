import argparse
import os
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

from data import get_dataloaders
from models import CNN
from train import train_model

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

SAVE_DIR = "ogrenci2"
COLORS = {
    "Adam":     "#2196F3",
    "Momentum": "#F44336",
}
LINE_STYLES = {
    "Base": "-",
    "L1":   "--",
    "L2":   ":",
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_test_accuracy_curves(results, save_path):
    reg_types = ["Base", "L1", "L2"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, reg in zip(axes, reg_types):
        for opt in ["Adam", "Momentum"]:
            key = f"{reg} + {opt}"
            if key not in results:
                continue
            history = results[key]
            epochs = range(1, len(history["test_acc"]) + 1)
            ax.plot(
                epochs, history["test_acc"],
                color=COLORS[opt],
                linestyle=LINE_STYLES[reg],
                marker="o" if opt == "Adam" else "s",
                label=opt,
            )
        ax.set_title(f"Regularization: {reg}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test Accuracy (%)")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Adam vs SGD+Momentum — Test Accuracy",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → {save_path}")


def plot_train_loss_curves(results, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    reg_types = ["Base", "L1", "L2"]

    for ax, reg in zip(axes, reg_types):
        for opt in ["Adam", "Momentum"]:
            key = f"{reg} + {opt}"
            if key not in results:
                continue
            history = results[key]
            epochs = range(1, len(history["train_loss"]) + 1)
            ax.plot(
                epochs, history["train_loss"],
                color=COLORS[opt],
                linestyle=LINE_STYLES[reg],
                marker="o" if opt == "Adam" else "s",
                label=opt,
            )
        ax.set_title(f"Regularization: {reg}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train Loss")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Adam vs SGD+Momentum — Training Loss",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → {save_path}")


def plot_final_accuracy_bar(results, save_path):
    reg_types = ["Base", "L1", "L2"]
    x = np.arange(len(reg_types))
    width = 0.35

    adam_accs = [results.get(f"{r} + Adam",     {}).get("test_acc", [0])[-1] for r in reg_types]
    mom_accs  = [results.get(f"{r} + Momentum", {}).get("test_acc", [0])[-1] for r in reg_types]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_adam = ax.bar(x - width / 2, adam_accs, width, label="Adam",         color=COLORS["Adam"],     alpha=0.88)
    bars_mom  = ax.bar(x + width / 2, mom_accs,  width, label="SGD+Momentum", color=COLORS["Momentum"], alpha=0.88)

    for bars in (bars_adam, bars_mom):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.4,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

    ax.set_title("Final Test Accuracy — Adam vs SGD+Momentum", fontsize=14)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(reg_types)
    ax.set_ylim(0, max(max(adam_accs), max(mom_accs)) + 10)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → {save_path}")


def plot_convergence_speed(results, save_path, target_acc=50.0):
    labels, epochs_to_target = [], []

    for key, history in results.items():
        accs = history["test_acc"]
        reached = next((i + 1 for i, a in enumerate(accs) if a >= target_acc), None)
        labels.append(key)
        epochs_to_target.append(reached if reached is not None else len(accs) + 1)

    colors = [COLORS["Adam"] if "Adam" in l else COLORS["Momentum"] for l in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, epochs_to_target, color=colors, alpha=0.88)

    for bar, val in zip(bars, epochs_to_target):
        label_txt = f"Epoch {val}" if val <= max(epochs_to_target) - 1 else "Ulaşamadı"
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                label_txt, va="center", fontsize=10)

    ax.set_title(f"Convergence Speed — {target_acc}% Test Accuracy'ye Ulaşma Süresi")
    ax.set_xlabel("Epoch Sayısı")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_elements = [
        Patch(facecolor=COLORS["Adam"],     label="Adam"),
        Patch(facecolor=COLORS["Momentum"], label="SGD+Momentum"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--l1",         type=float, default=1e-4)
    parser.add_argument("--l2",         type=float, default=1e-3)
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\n{'═' * 60}")
    print(f"  Öğrenci 2 — Adam vs SGD+Momentum Karşılaştırması")
    print(f"  Device     : {device}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  LR         : {args.lr}")
    print(f"  L1 lambda  : {args.l1}")
    print(f"  L2 decay   : {args.l2}")
    print(f"{'═' * 60}\n")

    trainloader, testloader = get_dataloaders(batch_size=args.batch_size)

    experiments = [
        {"label": "Base + Adam",     "l1": 0.0,     "l2": 0.0,     "optimizer": "adam"},
        {"label": "Base + Momentum", "l1": 0.0,     "l2": 0.0,     "optimizer": "sgd"},
        {"label": "L1 + Adam",       "l1": args.l1, "l2": 0.0,     "optimizer": "adam"},
        {"label": "L1 + Momentum",   "l1": args.l1, "l2": 0.0,     "optimizer": "sgd"},
        {"label": "L2 + Adam",       "l1": 0.0,     "l2": args.l2, "optimizer": "adam"},
        {"label": "L2 + Momentum",   "l1": 0.0,     "l2": args.l2, "optimizer": "sgd"},
    ]

    results = {}

    for exp in experiments:
        print(f"{'─' * 60}")
        print(f"  Deney: {exp['label']}")
        print(f"{'─' * 60}")

        set_seed(42)
        model = CNN(num_classes=10)

        history = train_model(
            model=model,
            trainloader=trainloader,
            testloader=testloader,
            epochs=args.epochs,
            device=device,
            lr=args.lr,
            l1_lambda=exp["l1"],
            l2_weight_decay=exp["l2"],
            optimizer_name=exp["optimizer"],
            scheduler_name="cosine",
            verbose=True,
        )

        results[exp["label"]] = history
        print(f"  Final Test Acc: {history['test_acc'][-1]:.2f}%\n")

    print(f"\n{'═' * 60}")
    print(f"  SONUÇ ÖZETİ")
    print(f"{'═' * 60}")
    for label, history in results.items():
        print(f"  {label:<28} → Test Acc: {history['test_acc'][-1]:.2f}%")

    print(f"\n{'─' * 60}")
    print("  Grafikler üretiliyor...")

    plot_test_accuracy_curves(results, save_path=os.path.join(SAVE_DIR, "test_accuracy_curves.png"))
    plot_train_loss_curves(results,    save_path=os.path.join(SAVE_DIR, "train_loss_curves.png"))
    plot_final_accuracy_bar(results,   save_path=os.path.join(SAVE_DIR, "final_accuracy_bar.png"))
    plot_convergence_speed(results,    save_path=os.path.join(SAVE_DIR, "convergence_speed.png"), target_acc=50.0)

    print(f"\n  Tüm grafikler '{SAVE_DIR}/' klasörüne kaydedildi.")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()