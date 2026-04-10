"""CIFAR-10 için CNN; isteğe bağlı Dropout ve BatchNorm."""

import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
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

        self.conv1 = nn.Conv2d(3, c1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(c1) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(c2) if use_batchnorm else nn.Identity()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(c3) if use_batchnorm else nn.Identity()
        self.conv4 = nn.Conv2d(c3, c3, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(c3) if use_batchnorm else nn.Identity()

        self.fc1 = nn.Linear(c3 * 8 * 8, fc_hidden_dim)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        self.fc2 = nn.Linear(fc_hidden_dim, num_classes)

        self._apply_init(init_scheme)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def _apply_init(self, init_scheme):
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
    model = CNN(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        use_batchnorm=use_batchnorm,
        base_channels=base_channels,
        fc_hidden_dim=fc_hidden_dim,
        init_scheme=init_scheme,
    )
    return sum(p.numel() for p in model.parameters())
