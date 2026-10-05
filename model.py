"""ResNet-50 image encoder and category classifier."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


class SareeResNet50(nn.Module):
    """A 256-D L2-normalized embedding plus a broad-category classification head."""
    def __init__(self, num_classes: int, pretrained: bool = True, embedding_dim: int = 256):
        super().__init__()
        # Pin the weight enum so the initialization does not change with torchvision's DEFAULT alias.
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.embedding_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.embedding_dim = embedding_dim

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        embedding = F.normalize(self.embedding_head(features), p=2, dim=1)
        logits = self.classifier(embedding)
        return embedding, logits


def build_model(num_classes: int, pretrained: bool = True, embedding_dim: int = 256) -> SareeResNet50:
    return SareeResNet50(num_classes=num_classes, pretrained=pretrained, embedding_dim=embedding_dim)
