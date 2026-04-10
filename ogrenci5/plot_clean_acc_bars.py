"""Eğitim logundan temiz test doğruluğu çubuk grafiği üretir."""
import json
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    path = os.path.join(HERE, "student5_report_data.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    accs = data["regularization_training_clean_test_acc"]
    labels = list(accs.keys())
    vals = list(accs.values())

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = plt.cm.tab10(range(len(labels)))
    ax.bar(range(len(labels)), vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel("Test accuracy (%) — temiz")
    ax.set_title("Düzenleme modelleri — eğitim sonu temiz doğruluk (log 259520)")
    ax.set_ylim(0, 100)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.0, f"{v:.1f}", ha="center", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(HERE, "clean_accuracy_regularization.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
