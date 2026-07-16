"""
improved_cnn.py

Deepened CNN — BatchNorm after every conv, 5 blocks (32->512 filters),
GlobalAveragePooling instead of Flatten, stronger Dropout.

Given the baseline already hit ~96% test accuracy, this architecture isn't
chasing a big raw-accuracy jump — it's targeting:
  1. Better train/val consistency (regularization doing its job)
  2. Faster, more stable convergence (BatchNorm)
  3. Fewer parameters in the classifier head (GAP vs Flatten), which should
     help generalization on the smaller classes specifically

Same training procedure as baseline_cnn.py (via train.py's run_training) —
architecture is the only variable that changes, so the Phase 3 vs Phase 4
comparison is fair.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU -> MaxPool, the repeating unit of this network."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class ImprovedCNN(nn.Module):
    def __init__(self, num_classes: int, img_size: int = 128, dropout: float = 0.4):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32),     # img_size / 2
            ConvBlock(32, 64),    # img_size / 4
            ConvBlock(64, 128),   # img_size / 8
            ConvBlock(128, 256),  # img_size / 16
            ConvBlock(256, 512),  # img_size / 32
        )

        # GlobalAveragePooling collapses each of the 512 feature maps to a
        # single value, regardless of spatial size — far fewer parameters
        # than Flatten, and less prone to overfitting on small classes.
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Shape sanity check + side-by-side param count vs baseline
    model = ImprovedCNN(num_classes=38, img_size=128)
    dummy_input = torch.randn(4, 3, 128, 128)
    output = model(dummy_input)
    print(f"Input shape : {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # expect [4, 38]

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    try:
        from baseline_cnn import BaselineCNN
        baseline = BaselineCNN(num_classes=38, img_size=128)
        n_baseline_params = sum(p.numel() for p in baseline.parameters())
        print(f"(Baseline had : {n_baseline_params:,} parameters)")
    except ImportError:
        pass