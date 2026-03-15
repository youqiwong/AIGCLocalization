from typing import Any, Dict

import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import Stage1Transform
from .utils import decode_image_like, decode_mask_like, load_jsonl


class _ParquetTurnTableReader:
    def __init__(self):
        self._pf_cache: Dict[str, Any] = {}
        self._rows_cache: Dict[str, Any] = {}

    def _get_pf(self, parquet_path: str):
        if parquet_path not in self._pf_cache:
            self._pf_cache[parquet_path] = pq.ParquetFile(parquet_path)
        return self._pf_cache[parquet_path]

    def get_row(self, parquet_path: str, row_group: int, row_index_in_group: int) -> Dict[str, Any]:
        key = f"{parquet_path}::rg{row_group}"
        if key not in self._rows_cache:
            pf = self._get_pf(parquet_path)
            table = pf.read_row_group(row_group, use_threads=True)
            self._rows_cache[key] = table.to_pylist()
        rows = self._rows_cache[key]
        return rows[row_index_in_group]


class MagicBrushDataset(Dataset):
    def __init__(self, manifest_path: str, image_size: int):
        self.samples = load_jsonl(manifest_path)
        self.transform = Stage1Transform(image_size=image_size)
        self.turn_reader = _ParquetTurnTableReader()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        if "storage" in sample and sample["storage"].get("type") == "magicbrush_parquet_turn_table":
            st = sample["storage"]
            row = self.turn_reader.get_row(
                parquet_path=st["parquet_path"],
                row_group=int(st["row_group"]),
                row_index_in_group=int(st["row_index_in_group"]),
            )
            image = decode_image_like(row[st["image_field"]])
            if st.get("mask_mode") == "zero":
                mask = Image.new("L", image.size, color=0)
            else:
                mask_field = st.get("mask_field", "mask_img")
                if row.get(mask_field) is None:
                    raise ValueError(f"mask missing for sample_id={sample['sample_id']}")
                mask = decode_mask_like(row[mask_field])
        else:
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
