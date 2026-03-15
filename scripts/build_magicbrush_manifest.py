#!/usr/bin/env python3
import argparse
import base64
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.utils import save_jsonl


def _col_priority(names: Iterable[str], keys: List[str]) -> List[str]:
    scored = []
    for n in names:
        s = n.lower()
        score = sum(1 for k in keys if k in s)
        if score > 0:
            scored.append((score, n))
    scored.sort(reverse=True)
    return [x[1] for x in scored]


def _to_jsonable(v: Any) -> Any:
    if isinstance(v, bytes):
        return "data:application/octet-stream;base64," + base64.b64encode(v).decode("utf-8")
    if isinstance(v, dict):
        return {k: _to_jsonable(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_to_jsonable(x) for x in v]
    return v


def _is_image_like(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (bytes, str, dict)):
        return True
    return False


def _ensure_list(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def _extract_row_id(row: Dict[str, Any], row_idx: int, split: str) -> str:
    for key in ["id", "sample_id", "image_id", "img_id", "uid", "key"]:
        if key in row and row[key] is not None:
            return f"{split}-{row[key]}"
    return f"{split}-row{row_idx}"


def _infer_fields(columns: List[str]) -> Dict[str, List[str]]:
    source_candidates = _col_priority(columns, ["input", "source", "original", "init", "image"])
    output_candidates = _col_priority(columns, ["output", "edited", "target", "image"])
    mask_candidates = _col_priority(columns, ["mask"])
    return {
        "source": source_candidates,
        "output": output_candidates,
        "mask": mask_candidates,
    }


def _extract_turn_payloads(row: Dict[str, Any], fields: Dict[str, List[str]]) -> Tuple[Any, List[Any], List[Any]]:
    source_image = None
    for c in fields["source"]:
        if c in row and _is_image_like(row[c]):
            source_image = row[c]
            break
    if source_image is None:
        for c, v in row.items():
            if "image" in c.lower() and _is_image_like(v):
                source_image = v
                break

    output_images: List[Any] = []
    output_masks: List[Any] = []

    for c in fields["output"]:
        if c not in row:
            continue
        v = row[c]
        if isinstance(v, list):
            if v and _is_image_like(v[0]):
                output_images = v
                break
        elif _is_image_like(v) and c not in fields["source"]:
            output_images = [v]
            break

    for c in fields["mask"]:
        if c not in row:
            continue
        v = row[c]
        if isinstance(v, list):
            output_masks = v
            break
        if _is_image_like(v):
            output_masks = [v]
            break

    # fallback: infer nested turn struct like turns=[{image,mask}, ...]
    if not output_images:
        for _, v in row.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                im_keys = [k for k in v[0].keys() if "image" in k.lower() or "output" in k.lower()]
                m_keys = [k for k in v[0].keys() if "mask" in k.lower()]
                if im_keys:
                    output_images = [turn.get(im_keys[0]) for turn in v]
                    output_masks = [turn.get(m_keys[0]) if m_keys else None for turn in v]
                    break

    return source_image, _ensure_list(output_images), _ensure_list(output_masks)


def _build_records_from_file(path: Path, split: str) -> List[Dict[str, Any]]:
    pf = pq.ParquetFile(path)
    schema_cols = [f.name for f in pf.schema_arrow]
    fields = _infer_fields(schema_cols)
    records: List[Dict[str, Any]] = []

    # Explicit support for MagicBrush turn-table schema:
    # img_id, turn_index, source_img, target_img, mask_img
    has_turn_table = {"img_id", "turn_index", "source_img", "target_img", "mask_img"}.issubset(set(schema_cols))

    if has_turn_table:
        groups: Dict[str, Dict[str, Any]] = {}
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, use_threads=True)
            rows = table.to_pylist()
            for row in rows:
                img_id = str(row["img_id"])
                source_group_id = f"{split}-{img_id}"
                turn_index = int(row["turn_index"])
                src = row.get("source_img")
                tgt = row.get("target_img")
                msk = row.get("mask_img")
                if tgt is None:
                    raise ValueError(f"target_img missing for {source_group_id} turn={turn_index}")
                if msk is None:
                    raise ValueError(f"mask_img missing for {source_group_id} turn={turn_index}")

                if img_id not in groups:
                    groups[img_id] = {
                        "source_group_id": source_group_id,
                        "clean_source": src,
                        "clean_turn": turn_index,
                        "turns": [],
                    }
                else:
                    # Prefer source image from earliest turn as the clean input image.
                    if turn_index < groups[img_id]["clean_turn"]:
                        groups[img_id]["clean_source"] = src
                        groups[img_id]["clean_turn"] = turn_index

                groups[img_id]["turns"].append((turn_index, tgt, msk))

        for g in groups.values():
            clean_source = g["clean_source"]
            if clean_source is None:
                raise ValueError(f"source_img missing for {g['source_group_id']}")
            records.append(
                {
                    "sample_id": f"{g['source_group_id']}-clean",
                    "image": _to_jsonable(clean_source),
                    "mask": "__ZERO__",
                    "label": 0,
                    "turn_index": 0,
                    "source_group_id": g["source_group_id"],
                    "dataset": "MagicBrush",
                    "split": split,
                }
            )
            for turn_index, tgt, msk in sorted(g["turns"], key=lambda x: x[0]):
                sid = f"{g['source_group_id']}-turn{turn_index}"
                records.append(
                    {
                        "sample_id": sid,
                        "image": _to_jsonable(tgt),
                        "mask": _to_jsonable(msk),
                        "label": 1,
                        "turn_index": int(turn_index),
                        "source_group_id": g["source_group_id"],
                        "dataset": "MagicBrush",
                        "split": split,
                    }
                )
        return records

    global_row_idx = 0
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, use_threads=True)
        rows = table.to_pylist()
        for row in rows:
            source, outputs, masks = _extract_turn_payloads(row, fields)
            if source is None:
                global_row_idx += 1
                continue
            source_group_id = _extract_row_id(row, global_row_idx, split)
            clean_record = {
                "sample_id": f"{source_group_id}-clean",
                "image": _to_jsonable(source),
                "mask": "__ZERO__",
                "label": 0,
                "turn_index": 0,
                "source_group_id": source_group_id,
                "dataset": "MagicBrush",
                "split": split,
            }
            records.append(clean_record)

            if outputs and len(masks) not in (0, len(outputs)):
                raise ValueError(
                    f"mask/output length mismatch for {source_group_id}: outputs={len(outputs)} masks={len(masks)}"
                )

            if outputs:
                if not masks:
                    raise ValueError(f"edited sample has no masks: {source_group_id}")
                for i, (img_v, mask_v) in enumerate(zip(outputs, masks), start=1):
                    sid = f"{source_group_id}-turn{i}"
                    if mask_v is None:
                        raise ValueError(f"mask missing for edited sample id={sid}")
                    records.append(
                        {
                            "sample_id": sid,
                            "image": _to_jsonable(img_v),
                            "mask": _to_jsonable(mask_v),
                            "label": 1,
                            "turn_index": i,
                            "source_group_id": source_group_id,
                            "dataset": "MagicBrush",
                            "split": split,
                        }
                    )

            global_row_idx += 1
    return records


def _subsample_debug(records: List[Dict[str, Any]], clean_n: int, edit_n: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    clean = [r for r in records if r["label"] == 0]
    edit = [r for r in records if r["label"] == 1]
    if len(clean) < clean_n or len(edit) < edit_n:
        raise ValueError(f"insufficient samples: clean={len(clean)} edit={len(edit)} need {clean_n}/{edit_n}")
    return rng.sample(edit, edit_n) + rng.sample(clean, clean_n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--mode", choices=["debug", "full"], default="debug")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_files = sorted(data_root.glob("train-*.parquet"))
    dev_files = sorted(data_root.glob("dev-*.parquet"))
    if not train_files or not dev_files:
        raise FileNotFoundError(f"no train/dev parquet found under {data_root}")

    train_records: List[Dict[str, Any]] = []
    for p in train_files:
        train_records.extend(_build_records_from_file(p, split="train"))

    dev_records: List[Dict[str, Any]] = []
    for p in dev_files:
        dev_records.extend(_build_records_from_file(p, split="dev"))

    if args.mode == "debug":
        train_out = _subsample_debug(train_records, clean_n=8000, edit_n=8000, seed=args.seed)
        val_out = _subsample_debug(dev_records, clean_n=1000, edit_n=1000, seed=args.seed + 1)
        test_out = _subsample_debug(dev_records, clean_n=2000, edit_n=2000, seed=args.seed + 2)
        save_jsonl(str(out_dir / "debug_train.jsonl"), train_out)
        save_jsonl(str(out_dir / "debug_val.jsonl"), val_out)
        save_jsonl(str(out_dir / "debug_test.jsonl"), test_out)
    else:
        save_jsonl(str(out_dir / "full_train.jsonl"), train_records)
        save_jsonl(str(out_dir / "full_val.jsonl"), dev_records)
        save_jsonl(str(out_dir / "full_test.jsonl"), dev_records)

    stats = {
        "mode": args.mode,
        "train_total": len(train_records),
        "dev_total": len(dev_records),
    }
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
