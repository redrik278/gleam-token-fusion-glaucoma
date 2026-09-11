"""Eye-aligned image/report fusion with availability-aware attention."""

from __future__ import annotations

import torch
from torch import nn
from glaucoma.model import encoder

MODALITIES = ("fundus", "visual_field", "oct")
EYES = ("od", "os")


class EyeAlignedReportFusion(nn.Module):
    def __init__(self, report_features: int, pretrained: bool = True, dimension: int = 192) -> None:
        super().__init__()
        self.encoders = nn.ModuleDict({m: encoder(pretrained) for m in MODALITIES})
        self.project = nn.ModuleDict({m: nn.Linear(512, dimension) for m in MODALITIES})
        self.report_eye = nn.ModuleDict({e: nn.Sequential(nn.Linear(report_features,dimension),nn.LayerNorm(dimension),nn.GELU()) for e in EYES})
        self.type_embedding = nn.Parameter(torch.randn(1,4,dimension)*.02)
        eye_layer=nn.TransformerEncoderLayer(dimension,4,dimension*2,.15,activation="gelu",batch_first=True,norm_first=True)
        self.eye_fusion=nn.TransformerEncoder(eye_layer,2,enable_nested_tensor=False)
        self.eye_attention=nn.Linear(dimension,1)
        self.eye_embedding=nn.Parameter(torch.randn(1,2,dimension)*.02)
        patient_layer=nn.TransformerEncoderLayer(dimension,4,dimension*2,.15,activation="gelu",batch_first=True,norm_first=True)
        self.patient_fusion=nn.TransformerEncoder(patient_layer,1,enable_nested_tensor=False)
        self.patient_attention=nn.Linear(dimension,1)
        self.head=nn.Sequential(nn.LayerNorm(dimension),nn.Dropout(.25),nn.Linear(dimension,1))

    def forward(self, batch: dict[str,torch.Tensor]) -> torch.Tensor:
        report=batch["report"]
        availability=batch["availability"]
        eye_vectors=[]
        for eye_index,eye in enumerate(EYES):
            image_tokens=[self.project[m](self.encoders[m](batch[f"{m}_{eye}"])) for m in MODALITIES]
            tokens=torch.stack(image_tokens+[self.report_eye[eye](report)],1)+self.type_embedding
            mask=torch.cat([~availability[:,eye_index*3:(eye_index+1)*3],torch.zeros(len(report),1,dtype=torch.bool,device=report.device)],1)
            encoded=self.eye_fusion(tokens,src_key_padding_mask=mask)
            weights=self.eye_attention(encoded).squeeze(-1).masked_fill(mask,-1e4).softmax(1)
            eye_vectors.append((encoded*weights.unsqueeze(-1)).sum(1))
        eyes=torch.stack(eye_vectors,1)+self.eye_embedding
        eyes=self.patient_fusion(eyes)
        weights=self.patient_attention(eyes).squeeze(-1).softmax(1)
        patient=(eyes*weights.unsqueeze(-1)).sum(1)
        return self.head(patient).squeeze(1)
