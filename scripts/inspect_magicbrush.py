#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pyarrow.parquet as pq


def _type_of_value(v: Any) -> str:
    if v is None:
        return "None"
    if isinstance(v, bytes):
        return "bytes"
    if isinstance(v, dict):
        return f"dict(keys={list(v.keys())[:8]})"
    if isinstance(v, list):
        if not v:
            return "list(empty)"
        return f"list(len={len(v)}, elem={type(v[0]).__name__})"
    return type(v).__name__


def inspect_one_parquet(path: Path) -> Dict[str, Any]:
    pf = pq.ParquetFile(path)
    table = pf.read_row_groups([0], use_threads=False)
    schema = table.schema
    row = table.slice(0, 1).to_pylist()[0] if table.num_rows > 0 else {}
    columns = []
    for field in schema:
        sample_val = row.get(field.name, None)
        columns.append(
            {
                "name": field.name,
                "arrow_type": str(field.type),
                "sample_python_type": _type_of_value(sample_val),
            }
        )

    image_like_cols = []
    for c in columns:
        t = (c["arrow_type"] + " " + c["name"]).lower()
        if any(k in t for k in ["image", "img", "mask", "pixel", "jpg", "png"]):
            image_like_cols.append(c["name"])

    return {
        "path": str(path),
        "num_row_groups": pf.num_row_groups,
        "num_rows_row_group0": table.num_rows,
        "columns": columns,
        "image_like_columns_guess": image_like_cols,
        "sample_row_preview": row,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    train_files = sorted(data_root.glob("train-*.parquet"))
    dev_files = sorted(data_root.glob("dev-*.parquet"))
    if not train_files or not dev_files:
        raise FileNotFoundError(f"parquet files not found under {data_root}")

    train_report = inspect_one_parquet(train_files[0])
    dev_report = inspect_one_parquet(dev_files[0])
    report = {"train_example": train_report, "dev_example": dev_report}

    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
