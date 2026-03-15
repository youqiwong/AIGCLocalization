from typing import Any, Dict

import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import Stage1Transform
from .utils import decode_image_like, decode_mask_like, load_jsonl


class MagicBrushDataset(Dataset):
    def __init__(self, manifest_path: str, image_size: int):
        self.samples = load_jsonl(manifest_path)
        self.transform = Stage1Transform(image_size=image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image = decode_image_like(sample["image"])
        if sample["mask"] == "__ZERO__":
            mask = Image.new("L", image.size, color=0)
        else:
            mask = decode_mask_like(sample["mask"])
        out = self.transform(image=image, mask=mask)

        # image: Tensor[3,H,W], mask: Tensor[1,H,W], label: Tensor[]
        return {
            "image": out["image"],
            "mask": out["mask"],
            "label": torch.tensor(float(sample["label"]), dtype=torch.float32),
            "meta": {
                "sample_id": sample["sample_id"],
                "turn_index": int(sample["turn_index"]),
                "source_group_id": sample["source_group_id"],
            },
        }
