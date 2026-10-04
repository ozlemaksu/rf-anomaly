import torch.nn as nn


class CNN1D(nn.Module):
    """Ham IQ girdisi: (N, 2, L). AdaptiveAvgPool sayesinde L'den bagimsiz (L >= 16)."""

    def __init__(self, n_classes=2, base=32, dropout=0.3):
        super().__init__()

        def block(cin, cout, k):
            return nn.Sequential(
                nn.Conv1d(cin, cout, k, padding=k // 2),
                nn.BatchNorm1d(cout),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )

        self.features = nn.Sequential(
            block(2, base, 7),
            block(base, base * 2, 5),
            block(base * 2, base * 4, 3),
            block(base * 4, base * 4, 3),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(base * 4, n_classes)
        )

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


class CNN2D(nn.Module):
    """Spektrogram girdisi: (N, C, F, T). Boyuttan bagimsiz (F, T >= 16)."""

    def __init__(self, in_ch=1, n_classes=2, base=16, dropout=0.3):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_ch, base),
            block(base, base * 2),
            block(base * 2, base * 4),
            block(base * 4, base * 4),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(base * 4, n_classes)
        )

    def forward(self, x):
        return self.head(self.pool(self.features(x)))
