from __future__ import annotations

import torch
from torch import nn
from torchvision import models

class DermCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        # Use a pre-trained ResNet50 for clinical-grade feature extraction
        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=weights)
        
        # Replace the final fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
