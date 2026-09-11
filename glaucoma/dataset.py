"""PyTorch dataset for prepared multimodal samples."""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MultimodalGlaucomaDataset(Dataset):
    def __init__(self, manifest: str | Path, split: str, train: bool = False) -> None:
        data = pd.read_csv(manifest)
        self.data = data[data["split"].eq(split)].reset_index(drop=True)
        operations = [transforms.Resize((224, 224))]
        if train:
            operations.extend([transforms.RandomRotation(3)])
        operations.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.transform = transforms.Compose(operations)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.data.iloc[index]
        return {
            "fundus": self.transform(Image.open(row["fundus_input"]).convert("RGB")),
            "visual_field": self.transform(Image.open(row["visual_field_input"]).convert("RGB")),
            "oct": self.transform(Image.open(row["oct_input"]).convert("RGB")),
            "label": torch.tensor(float(row["final_label"]), dtype=torch.float32),
            "patient_id": row["patient_id"],
        }
