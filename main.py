"""CLI: tek model eğitimi (CIFAR-10, düzenlileştirme argümanları)."""

import argparse
import random
import numpy as np
import torch
from data import get_dataloaders
from models import CNN
from train import train_model

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    parser.add_argument('--dropout_rate',  type=float, default=0.0, help='Dropout rate (e.g. 0.2, 0.5)')
    parser.add_argument('--use_batchnorm', action='store_true', help='Enable Batch Normalization')
    parser.add_argument('--init_scheme', type=str, default='default', choices=['default', 'he', 'xavier'], help='Weight initialization scheme')
    parser.add_argument('--base_channels', type=int, default=16, help='Base conv channel width (capacity control)')
    parser.add_argument('--fc_hidden_dim', type=int, default=256, help='Hidden dimension of FC layer (capacity control)')
    parser.add_argument('--adv_train',     action='store_true', help='Enable Adversarial Training (FGSM)')
    parser.add_argument('--save_model',    action='store_true', help='Save trained model weights for tests')
    parser.add_argument('--no_augmentation',action='store_true', help='Disable Data Augmentation')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'plateau', 'none'], help='LR scheduler type')
    parser.add_argument('--grad_clip_norm', type=float, default=0.0, help='Gradient clipping max norm (0 disables)')
    parser.add_argument('--plot_path', type=str, default='', help='Optional path to save train/test curves PNG')

    args = parser.parse_args()
    set_seed(42)
    device = get_device()

    print(f"{'═' * 55}")
    print(f"  Device        : {device}")
    print(f"  Epochs        : {args.epochs}")
    print(f"  Batch Size    : {args.batch_size}")
    print(f"  Learning Rate : {args.lr}")
    print(f"  L1 Lambda     : {args.l1}")
    print(f"  L2 Decay      : {args.l2}")
    print(f"  Label Smooth  : {args.label_smoothing}")
    print(f"  Dropout Rate  : {args.dropout_rate}")
    print(f"  BatchNorm     : {args.use_batchnorm}")
    print(f"  Init Scheme   : {args.init_scheme}")
    print(f"  Base Channels : {args.base_channels}")
    print(f"  FC Hidden Dim : {args.fc_hidden_dim}")
    print(f"  Adv Train     : {args.adv_train}")
    print(f"  Augmentation  : {not args.no_augmentation}")
    print(f"  Optimizer     : {args.optimizer}")
    print(f"  Scheduler     : {args.scheduler}")
    print(f"  Grad Clip     : {args.grad_clip_norm}")
    print(f"  Plot Path     : {args.plot_path if args.plot_path else '(disabled)'}")
    print(f"{'═' * 55}\n")

    print("Loading CIFAR-10 dataset...")
    trainloader, testloader = get_dataloaders(
        batch_size=args.batch_size, 
        use_augmentation=not args.no_augmentation
    )

    model = CNN(
        num_classes=10,
        dropout_rate=args.dropout_rate,
        use_batchnorm=args.use_batchnorm,
        init_scheme=args.init_scheme,
        base_channels=args.base_channels,
        fc_hidden_dim=args.fc_hidden_dim,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

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
        adv_train=args.adv_train,
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
        grad_clip_norm=args.grad_clip_norm,
        plot_path=args.plot_path if args.plot_path else None,
    )

    print(f"\n{'═' * 55}")
    print(f"  Training finished!")
    print(f"  Final Train Acc : {history['train_acc'][-1]:.2f}%")
    print(f"  Final Test Acc  : {history['test_acc'][-1]:.2f}%")
    
    if args.save_model:
        import os
        os.makedirs('results', exist_ok=True)
        save_path = f"results/model_final.pth"
        torch.save(model.state_dict(), save_path)
        print(f"  Model Saved To  : {save_path}")
        
    print(f"{'═' * 55}")


if __name__ == "__main__":
    main()
