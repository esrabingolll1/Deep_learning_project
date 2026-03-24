"""
train.py - Eğitim ve Doğrulama Döngüsü

Modeli eğiten ve her epoch sonunda doğrulama başarısını ölçen
fonksiyonları içerir. L1 ve L2 düzenlileştirme desteği mevcuttur.
"""

import torch
import torch.nn as nn
import torch.optim as optim


def train_model(model, trainloader, testloader, epochs, device,
                lr=0.001, l1_lambda=0.0, l2_weight_decay=0.0,
                label_smoothing=0.0, verbose=True):
    """
    Modeli eğitir ve epoch bazında metrikleri döndürür.

    Args:
        model: Eğitilecek nn.Module modeli.
        trainloader: Eğitim veri yükleyicisi.
        testloader: Doğrulama (test) veri yükleyicisi.
        epochs: Toplam eğitim epoch sayısı.
        device: Hesaplama cihazı (cpu / mps / cuda).
        lr: Öğrenme oranı (learning rate).
        l1_lambda: L1 düzenlileştirme katsayısı (0 = kapalı).
        l2_weight_decay: L2 düzenlileştirme katsayısı (0 = kapalı).
        label_smoothing: Label Smoothing katsayısı (0 = kapalı).
        verbose: True ise her epoch'ta sonuçları ekrana yazdırır.

    Returns:
        dict: {
            'train_loss': [...],
            'train_acc':  [...],
            'test_acc':   [...]
        }
    """
    model.to(device)

    # Cross-Entropy kaybı (opsiyonel Label Smoothing desteği)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Optimizer: L2 düzenlileştirme weight_decay ile uygulanır
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_weight_decay)

    history = {
        'train_loss': [],
        'train_acc':  [],
        'test_acc':   [],
    }

    for epoch in range(epochs):
        # ─────────── Eğitim Aşaması ───────────
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # L1 düzenlileştirme (manuel hesaplama)
            if l1_lambda > 0:
                l1_penalty = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_penalty

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        # ─────────── Doğrulama Aşaması ───────────
        model.eval()
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total_test += labels.size(0)
                correct_test += predicted.eq(labels).sum().item()

        # Metrikleri kaydet
        epoch_loss = running_loss / len(trainloader)
        train_acc  = 100.0 * correct_train / total_train
        test_acc   = 100.0 * correct_test / total_test

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        if verbose:
            print(f"  Epoch [{epoch+1:>2}/{epochs}]  "
                  f"Loss: {epoch_loss:.4f}  |  "
                  f"Train Acc: {train_acc:.2f}%  |  "
                  f"Test Acc: {test_acc:.2f}%")

    return history
