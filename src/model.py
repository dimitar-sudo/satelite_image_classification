"""CNN architecture for the satellite terrain classifier.

A compact VGG-style network trained from scratch (no pretrained weights / no
transfer learning) for 64x64 RGB tiles -> 10 land-use classes.
"""
import torch.nn as nn


class SatelliteCNN(nn.Module):
    """VGG-style CNN built from scratch for 64x64 RGB -> ``num_classes``.

    Each block is two 3x3 convolutions with batch normalization and ReLU,
    followed by 2x2 max pooling. Four blocks reduce the 64x64 input to a 4x4
    feature map before a two-layer classifier head with dropout.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32),     # 64 -> 32
            block(32, 64),    # 32 -> 16
            block(64, 128),   # 16 -> 8
            block(128, 256),  # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
