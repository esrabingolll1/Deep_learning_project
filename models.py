"""
models.py - CNN Model Mimarisi

4 katmanlı bir Convolutional Neural Network (CNN) tanımlar.
Dropout ve Batch Normalization katmanları isteğe bağlı olarak
eklenebilir; böylece aynı mimari üzerinde farklı düzenlileştirme
yöntemlerinin etkisi karşılaştırılabilir.
"""

import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    CIFAR-10 sınıflandırması için 4 katmanlı CNN.

    Mimari:
        Conv(3→32) → Conv(32→64) → MaxPool →
        Conv(64→128) → Conv(128→128) → MaxPool →
        FC(8192→512) → FC(512→10)

    Args:
        num_classes: Çıkış sınıf sayısı (varsayılan: 10).
        use_dropout: True ise FC katmanları arasına Dropout(0.5) eklenir.
        use_batchnorm: True ise her Conv katmanından sonra BatchNorm eklenir.
    """

    def __init__(self, num_classes=10, use_dropout=False, use_batchnorm=False):
        super().__init__()

        # --- Convolutional Block 1 ---
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.pool  = nn.MaxPool2d(2, 2)

        # --- Convolutional Block 2 ---
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128) if use_batchnorm else nn.Identity()
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(128) if use_batchnorm else nn.Identity()

        # --- Fully Connected ---
        self.fc1     = nn.Linear(128 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()
        self.fc2     = nn.Linear(512, num_classes)

    def forward(self, x):
        # Block 1  (ReLU aktivasyonu + havuzlama)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        # Düzleştirme → Fully Connected
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
