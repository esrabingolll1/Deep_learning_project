"""
main.py - Proje Giriş Noktası

Komut satırından tek bir model eğitmek için kullanılır.
Tüm düzenlileştirme parametreleri argüman olarak verilebilir.

Örnekler:
    python main.py --epochs 10
    python main.py --epochs 15 --l2 0.001
    python main.py --epochs 15 --use_dropout --use_batchnorm
    python main.py --epochs 15 --label_smoothing 0.1
"""

import argparse
import torch
from data import get_dataloaders
from models import CNN
from train import train_model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(
        description="CIFAR-10 Regularization & Adversarial Robustness Project"
    )
    parser.add_argument('--epochs',     type=int,   default=10,   help='Number of training epochs')
    parser.add_argument('--batch_size', type=int,   default=128,  help='Batch size')
    parser.add_argument('--lr',         type=float, default=0.001,help='Learning rate')
    parser.add_argument('--l1',         type=float, default=0.0,  help='L1 regularization lambda')
    parser.add_argument('--l2',         type=float, default=0.0,  help='L2 weight decay')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing factor')
    parser.add_argument('--use_dropout',   action='store_true', help='Enable Dropout(0.5)')
    parser.add_argument('--use_batchnorm', action='store_true', help='Enable Batch Normalization')

    args = parser.parse_args()
    device = get_device()

    print(f"{'═' * 55}")
    print(f"  Device        : {device}")
    print(f"  Epochs        : {args.epochs}")
    print(f"  Batch Size    : {args.batch_size}")
    print(f"  Learning Rate : {args.lr}")
    print(f"  L1 Lambda     : {args.l1}")
    print(f"  L2 Decay      : {args.l2}")
    print(f"  Label Smooth  : {args.label_smoothing}")
    print(f"  Dropout       : {args.use_dropout}")
    print(f"  BatchNorm     : {args.use_batchnorm}")
    print(f"{'═' * 55}\n")

    # Veri yükleme
    print("Loading CIFAR-10 dataset...")
    trainloader, testloader = get_dataloaders(batch_size=args.batch_size)

    # Model oluşturma
    model = CNN(
        num_classes=10,
        use_dropout=args.use_dropout,
        use_batchnorm=args.use_batchnorm,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Eğitim
    history = train_model(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        epochs=args.epochs,
        device=device,
        lr=args.lr,
        l1_lambda=args.l1,
        l2_weight_decay=args.l2,
        label_smoothing=args.label_smoothing,
    )

    print(f"\n{'═' * 55}")
    print(f"  Training finished!")
    print(f"  Final Train Acc : {history['train_acc'][-1]:.2f}%")
    print(f"  Final Test Acc  : {history['test_acc'][-1]:.2f}%")
    print(f"{'═' * 55}")


if __name__ == "__main__":
    main()
