"""Eye-aligned multimodal dataset with training-fold-only tabular scaling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_KEYS = tuple(f"{modality}_{eye}" for eye in ("od", "os") for modality in ("fundus", "visual_field", "oct"))


class EyeAlignedDataset(Dataset):
    def __init__(self, manifest: str | Path, split: str, train: bool = False, zero_report: bool = False) -> None:
        source = pd.read_csv(manifest)
        self.data = source[source.split.eq(split)].reset_index(drop=True)
        self.zero_report = zero_report
        feature_file = Path(manifest).parent / "structured_feature_columns.json"
        self.features = json.loads(feature_file.read_text(encoding="utf-8"))
        reference = source[source.split.eq("train")][self.features].apply(pd.to_numeric, errors="coerce")
        self.median = reference.median().fillna(0.0)
        self.mean = reference.fillna(self.median).mean()
        self.std = reference.fillna(self.median).std().replace(0, 1).fillna(1)
        fundus_ops = [transforms.RandomResizedCrop(384, scale=(.88, 1.0), ratio=(.95, 1.05)),
                      transforms.RandomRotation(5), transforms.ColorJitter(.12, .12, .06, .02)] if train else [transforms.Resize((384,384))]
        self.fundus = transforms.Compose(fundus_ops + [
            transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        self.document = transforms.Compose([
            transforms.Resize((384,384)), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])
        ])
        oct_ops = [transforms.Resize((384,384))]
        if train: oct_ops.append(transforms.ColorJitter(brightness=.08, contrast=.08))
        self.oct = transforms.Compose(oct_ops + [transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

    def __len__(self) -> int: return len(self.data)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.data.iloc[index]
        result: dict[str, object] = {"patient_id": row.patient_id, "label": torch.tensor(float(row.final_label))}
        available = []
        for eye in ("od", "os"):
            for modality, transform in (("fundus",self.fundus),("visual_field",self.document),("oct",self.oct)):
                key = f"{modality}_{eye}"
                result[key] = transform(Image.open(row[f"{key}_input"]).convert("RGB"))
                available.append(float(row[f"{key}_available"]))
        values = pd.to_numeric(row[self.features], errors="coerce")
        missing = values.isna().astype(float)
        scaled = (values.fillna(self.median)-self.mean)/self.std
        report = np.r_[scaled.to_numpy(float), missing.to_numpy(float)]
        if self.zero_report: report = np.zeros_like(report)
        result["report"] = torch.tensor(report, dtype=torch.float32)
        result["availability"] = torch.tensor(available, dtype=torch.bool)
        return result
