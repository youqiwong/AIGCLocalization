from typing import Any, Dict, List, Optional

import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .transforms import Stage1Transform
from .utils import decode_image_like, decode_mask_like, load_jsonl


class _ParquetTurnTableReader:
    def __init__(self):
        self._pf_cache: Dict[str, Any] = {}

    def _get_pf(self, parquet_path: str):
        if parquet_path not in self._pf_cache:
            self._pf_cache[parquet_path] = pq.ParquetFile(parquet_path)
        return self._pf_cache[parquet_path]

    def get_row(
        self,
        parquet_path: str,
        row_group: int,
        row_index_in_group: int,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        pf = self._get_pf(parquet_path)
        table = pf.read_row_group(row_group, columns=columns, use_threads=False)
        col_names = columns if columns is not None else table.column_names
        row = {}
        for col in col_names:
            row[col] = table.column(col)[row_index_in_group].as_py()
        return row


class MagicBrushDataset(Dataset):
    def __init__(self, manifest_path: str, image_size: int, processor_name_or_path: str):
        self.samples = load_jsonl(manifest_path)
        self.transform = Stage1Transform(image_size=image_size)
        self.turn_reader = _ParquetTurnTableReader()
        from transformers import AutoImageProcessor

        self.image_processor = AutoImageProcessor.from_pretrained(processor_name_or_path, trust_remote_code=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        if "storage" in sample and sample["storage"].get("type") == "magicbrush_parquet_turn_table":
            st = sample["storage"]
            image_field = st["image_field"]
            mask_field = st.get("mask_field", "mask_img")
            columns = [image_field]
            if st.get("mask_mode") != "zero":
                columns.append(mask_field)
            row = self.turn_reader.get_row(
                parquet_path=st["parquet_path"],
                row_group=int(st["row_group"]),
                row_index_in_group=int(st["row_index_in_group"]),
                columns=columns,
            )
            image = decode_image_like(row[image_field])
            if st.get("mask_mode") == "zero":
                mask = Image.new("L", image.size, color=0)
            else:
                if row.get(mask_field) is None:
                    raise ValueError(f"mask missing for sample_id={sample['sample_id']}")
                mask = decode_mask_like(row[mask_field])
        else:
            image = decode_image_like(sample["image"])
            if sample["mask"] == "__ZERO__":
                mask = Image.new("L", image.size, color=0)
            else:
                mask = decode_mask_like(sample["mask"])
        image, mask = self.transform.resize_pair(image=image, mask=mask)
        image_inputs = self.image_processor(images=image, do_resize=False, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"]
        image_grid_thw = image_inputs["image_grid_thw"]
        out = {
            "image": TF.to_tensor(image),
            "mask": (TF.to_tensor(mask)[:1] > 0.5).float(),
        }

        return {
            "image": out["image"],
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "mask": out["mask"],
            "label": torch.tensor(float(sample["label"]), dtype=torch.float32),
            "meta": {
                "sample_id": sample["sample_id"],
                "turn_index": int(sample["turn_index"]),
                "source_group_id": sample["source_group_id"],
            },
        }


def collate_magicbrush_batch(batch):
    pixel_values = [item["pixel_values"].squeeze(0) for item in batch]
    image_grid_thw = [item["image_grid_thw"].reshape(-1, 3) for item in batch]
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "pixel_values": torch.cat(pixel_values, dim=0),
        "image_grid_thw": torch.cat(image_grid_thw, dim=0),
        "mask": torch.stack([item["mask"] for item in batch], dim=0),
        "label": torch.stack([item["label"] for item in batch], dim=0),
        "meta": [item["meta"] for item in batch],
    }
