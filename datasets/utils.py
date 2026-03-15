import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from PIL import Image


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def save_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _decode_bytes(blob: bytes) -> Image.Image:
    return Image.open(io.BytesIO(blob)).convert("RGB")


def _decode_base64(s: str) -> Image.Image:
    if "," in s and s.strip().startswith("data:"):
        s = s.split(",", 1)[1]
    return _decode_bytes(base64.b64decode(s))


def decode_image_like(value: Any) -> Image.Image:
    if value is None:
        raise ValueError("image field is None")
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, bytes):
        return _decode_bytes(value)
    if isinstance(value, str):
        v = value.strip()
        if v.startswith("data:image"):
            return _decode_base64(v)
        p = Path(v)
        if p.exists():
            return Image.open(p).convert("RGB")
        raise ValueError(f"unsupported image string: {value[:80]}")
    if isinstance(value, dict):
        if "bytes" in value and value["bytes"] is not None:
            b = value["bytes"]
            if isinstance(b, str):
                return _decode_base64(b)
            if isinstance(b, bytes):
                return _decode_bytes(b)
        if "path" in value and value["path"]:
            return Image.open(value["path"]).convert("RGB")
        if "array" in value:
            from numpy import array, uint8

            arr = array(value["array"], dtype=uint8)
            return Image.fromarray(arr).convert("RGB")
    raise ValueError(f"unsupported image payload type: {type(value)}")


def decode_mask_like(value: Any) -> Image.Image:
    if value == "__ZERO__":
        raise ValueError("__ZERO__ mask requires image size context")
    mask = decode_image_like(value).convert("L")
    return mask
