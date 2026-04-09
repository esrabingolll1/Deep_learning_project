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

    def __init__(
        self,
        num_classes=10,
        dropout_rate=0.0,
        use_batchnorm=False,
        base_channels=16,
        fc_hidden_dim=256,
        init_scheme="default",
    ):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        # --- Convolutional Block 1 ---
        self.conv1 = nn.Conv2d(3, c1, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(c1) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(c2) if use_batchnorm else nn.Identity()
        self.pool  = nn.MaxPool2d(2, 2)

        # --- Convolutional Block 2 ---
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(c3) if use_batchnorm else nn.Identity()
        self.conv4 = nn.Conv2d(c3, c3, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(c3) if use_batchnorm else nn.Identity()

        # --- Fully Connected ---
        self.fc1     = nn.Linear(c3 * 8 * 8, fc_hidden_dim)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        self.fc2     = nn.Linear(fc_hidden_dim, num_classes)

        self._apply_init(init_scheme)

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

    def _apply_init(self, init_scheme):
        """Conv/Linear katmanlarına opsiyonel ağırlık ilklendirme uygular."""
        scheme = init_scheme.lower()
        if scheme == "default":
            return

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if scheme == "he":
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                elif scheme == "xavier":
                    nn.init.xavier_normal_(module.weight)
                else:
                    raise ValueError(f"Unknown init_scheme: {init_scheme}")

                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def count_cnn_parameters(
    num_classes=10,
    dropout_rate=0.0,
    use_batchnorm=False,
    base_channels=16,
    fc_hidden_dim=256,
    init_scheme="default",
) -> int:
    """
    Verilen CNN konfigürasyonu için toplam parametre sayısını hesaplar.

    Teknik yorum (Capacity ve overfitting):
    Parametre sayısı (model kapasitesi) arttıkça modelin veriyi ezberleme (memorization)
    eğilimi de genellikle artar; bu da bias azalırken variance büyümesine ve dolayısıyla
    overfitting riskinin yükselmesine yol açabilir. Düzenlileştirme (L1/L2, dropout vb.)
    bu trade-off dengesini daha kontrollü tutmayı amaçlar.
    """
    model = CNN(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        use_batchnorm=use_batchnorm,
        base_channels=base_channels,
        fc_hidden_dim=fc_hidden_dim,
        init_scheme=init_scheme,
    )
    return sum(p.numel() for p in model.parameters())
