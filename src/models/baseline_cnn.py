"""
baseline_cnn.py

Simple 4-block CNN — no BatchNorm, minimal Dropout, Flatten (not GAP).
Purpose: establish a working end-to-end pipeline and a reference score.
This is intentionally NOT accuracy-optimized — that's what improved_cnn.py
is for. Keeping this simple makes the baseline-vs-improved comparison in
Phase 4 mean something (architecture is the only variable that changes).
"""

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int, img_size: int = 128):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # img_size / 2

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # img_size / 4

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # img_size / 8

            # Block 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # img_size / 16
        )

        flattened_size = 128 * (img_size // 16) * (img_size // 16)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Shape sanity check
    model = BaselineCNN(num_classes=38, img_size=128)
    dummy_input = torch.randn(4, 3, 128, 128)
    output = model(dummy_input)
    print(f"Input shape : {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # expect [4, 38]

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")