"""Model definition for the Ray Train CIFAR-10 example.

Uses a torchvision ResNet with the stem adapted for 32x32 CIFAR-10 inputs (a 3x3
stride-1 first conv and no max-pool, the standard CIFAR adaptation), so the network
does not throw away spatial resolution before it has learned anything.
"""

import torch.nn as nn
from torchvision.models import resnet18


def build_resnet(num_classes: int = 10) -> nn.Module:
    model = resnet18(weights=None, num_classes=num_classes)
    # CIFAR-10 is 32x32; replace the ImageNet 7x7/stride-2 stem + maxpool.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
