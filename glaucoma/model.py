"""Neural models for single-modality and late-fusion classification."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


INPUT_KEYS = {
    "fundus": ("fundus",),
    "visual_field": ("visual_field",),
    "oct": ("oct",),
    "multimodal": ("fundus", "visual_field", "oct"),
    "attentive_multimodal": ("fundus", "visual_field", "oct"),
}


def encoder(pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    network = resnet18(weights=weights)
    network.fc = nn.Identity()
    return network


class GlaucomaClassifier(nn.Module):
    def __init__(self, modality: str, pretrained: bool = True) -> None:
        super().__init__()
        if modality not in INPUT_KEYS:
            raise ValueError(f"Unknown modality: {modality}")
        self.modality = modality
        self.input_keys = INPUT_KEYS[modality]
        self.encoders = nn.ModuleDict({key: encoder(pretrained) for key in self.input_keys})
        feature_count = 512 * len(self.input_keys)
        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(feature_count, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = [self.encoders[key](batch[key]) for key in self.input_keys]
        return self.classifier(torch.cat(features, dim=1)).squeeze(1)


class AttentiveFusionClassifier(nn.Module):
    """Modality-dropout transformer fusion with learned attention pooling."""

    def __init__(self, pretrained: bool = True, modality_dropout: float = 0.25) -> None:
        super().__init__()
        self.modality = "attentive_multimodal"
        self.input_keys = INPUT_KEYS[self.modality]
        self.encoders = nn.ModuleDict({key: encoder(pretrained) for key in self.input_keys})
        self.projections = nn.ModuleDict({key: nn.Linear(512, 256) for key in self.input_keys})
        self.modality_embedding = nn.Parameter(torch.randn(1, 3, 256) * 0.02)
        layer = nn.TransformerEncoderLayer(256, 4, 512, dropout=0.2, batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, num_layers=2)
        self.attention = nn.Linear(256, 1)
        self.classifier = nn.Sequential(nn.LayerNorm(256), nn.Dropout(0.3), nn.Linear(256, 1))
        self.modality_dropout = modality_dropout

    def forward(self, batch: dict[str, torch.Tensor], availability: torch.Tensor | None = None) -> torch.Tensor:
        tokens = torch.stack(
            [self.projections[key](self.encoders[key](batch[key])) for key in self.input_keys], dim=1
        ) + self.modality_embedding
        batch_size = tokens.shape[0]
        if availability is None:
            availability = torch.ones(batch_size, 3, dtype=torch.bool, device=tokens.device)
        if self.training and self.modality_dropout > 0:
            dropped = torch.rand(batch_size, 3, device=tokens.device) < self.modality_dropout
            dropped[:, 2] = False  # retain at least structural OCT during development training
            availability = availability & ~dropped
        encoded = self.fusion(tokens, src_key_padding_mask=~availability)
        logits = self.attention(encoded).squeeze(-1).masked_fill(~availability, -1e4)
        pooled = (encoded * torch.softmax(logits, dim=1).unsqueeze(-1)).sum(dim=1)
        return self.classifier(pooled).squeeze(1)


def build_model(modality: str, pretrained: bool = True) -> nn.Module:
    if modality == "attentive_multimodal":
        return AttentiveFusionClassifier(pretrained=pretrained)
    return GlaucomaClassifier(modality, pretrained=pretrained)
