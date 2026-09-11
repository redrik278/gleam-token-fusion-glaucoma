"""Missing-modality-aware cross-modal attention model."""

from __future__ import annotations

import torch
from torch import nn

from glaucoma.model import encoder


MODALITIES = ("fundus", "visual_field", "oct")


class CrossModalAttentionClassifier(nn.Module):
    def __init__(self, pretrained: bool = True, layers: int = 2, heads: int = 8, modality_dropout: float = 0.25) -> None:
        super().__init__()
        self.encoders = nn.ModuleDict({name: encoder(pretrained) for name in MODALITIES})
        self.modality_embeddings = nn.Parameter(torch.randn(1, len(MODALITIES), 512) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512) * 0.02)
        block = nn.TransformerEncoderLayer(
            d_model=512, nhead=heads, dim_feedforward=1024, dropout=0.2,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(block, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(512)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 1))
        self.modality_dropout = modality_dropout

    def _availability(self, batch_size: int, device: torch.device, supplied: torch.Tensor | None) -> torch.Tensor:
        if supplied is not None:
            return supplied.to(device=device, dtype=torch.bool)
        available = torch.ones(batch_size, len(MODALITIES), device=device, dtype=torch.bool)
        if self.training and self.modality_dropout > 0:
            dropped = torch.rand(batch_size, len(MODALITIES), device=device) < self.modality_dropout
            all_dropped = dropped.all(dim=1)
            if all_dropped.any():
                keep = torch.randint(0, len(MODALITIES), (int(all_dropped.sum()),), device=device)
                dropped[all_dropped] = True
                dropped[all_dropped, keep] = False
            available = ~dropped
        return available

    def forward(self, batch: dict[str, torch.Tensor], availability: torch.Tensor | None = None) -> torch.Tensor:
        features = torch.stack([self.encoders[name](batch[name]) for name in MODALITIES], dim=1)
        features = features + self.modality_embeddings
        available = self._availability(features.shape[0], features.device, availability)
        features = features * available.unsqueeze(-1)
        cls = self.cls_token.expand(features.shape[0], -1, -1)
        tokens = torch.cat([cls, features], dim=1)
        padding_mask = torch.cat([
            torch.zeros(features.shape[0], 1, device=features.device, dtype=torch.bool), ~available
        ], dim=1)
        fused = self.fusion(tokens, src_key_padding_mask=padding_mask)
        return self.classifier(self.norm(fused[:, 0])).squeeze(1)
