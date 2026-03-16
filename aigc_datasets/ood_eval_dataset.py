from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .magicbrush_dataset import mask_to_edge
from .transforms import Stage1Transform


class OODEvalDataset(Dataset):
    def __init__(
        self,
        forged_txt_path: str,
        clean_dir: str,
        image_size: int,
        processor_name_or_path: str,
        edge_kernel_size: int = 5,
        dataset_name: str = "OOD",
    ):
        self.dataset_name = dataset_name
        self.samples = self._build_samples(Path(forged_txt_path), Path(clean_dir))
        self.transform = Stage1Transform(image_size=image_size)
        self.edge_kernel_size = edge_kernel_size
        from transformers import AutoImageProcessor

        self.image_processor = AutoImageProcessor.from_pretrained(processor_name_or_path, trust_remote_code=True)

    def _build_samples(self, forged_txt_path: Path, clean_dir: Path) -> List[Dict[str, Any]]:
        if not forged_txt_path.exists():
            raise FileNotFoundError(f"forged txt not found: {forged_txt_path}")
        if not clean_dir.exists():
            raise FileNotFoundError(f"clean dir not found: {clean_dir}")

        samples: List[Dict[str, Any]] = []
        with open(forged_txt_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                forged_rel, mask_rel = [part.strip() for part in line.split(",", 1)]
                image_path = (forged_txt_path.parent / forged_rel).resolve()
                mask_path = (forged_txt_path.parent / mask_rel).resolve()
                samples.append(
                    {
                        "sample_id": f"{self.dataset_name}_forged_{idx}",
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "label": 1.0,
                    }
                )

        valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        clean_paths = sorted(p for p in clean_dir.rglob("*") if p.is_file() and p.suffix.lower() in valid_suffixes)
        for idx, image_path in enumerate(clean_paths):
            samples.append(
                {
                    "sample_id": f"{self.dataset_name}_clean_{idx}",
                    "image_path": image_path.resolve(),
                    "mask_path": None,
                    "label": 0.0,
                }
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        if sample["mask_path"] is None:
            mask = Image.new("L", image.size, color=0)
        else:
            mask = Image.open(sample["mask_path"]).convert("L")
        image, mask = self.transform.resize_pair(image=image, mask=mask)
        image_inputs = self.image_processor(images=image, do_resize=False, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"]
        image_grid_thw = image_inputs["image_grid_thw"]
        mask_tensor = (TF.to_tensor(mask)[:1] > 0.5).float()
        edge_tensor = mask_to_edge(mask_tensor, kernel_size=self.edge_kernel_size)

        return {
            "image": TF.to_tensor(image),
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "mask": mask_tensor,
            "edge_gt": edge_tensor,
            "label": torch.tensor(sample["label"], dtype=torch.float32),
            "meta": {
                "sample_id": sample["sample_id"],
                "dataset_name": self.dataset_name,
                "image_path": str(sample["image_path"]),
                "mask_path": "" if sample["mask_path"] is None else str(sample["mask_path"]),
            },
        }
