#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def find_vision(model: torch.nn.Module):
    for name in ["visual", "vision_tower", "vision_model", "model.vision_tower", "model.visual"]:
        cur = model
        ok = True
        for p in name.split("."):
            if not hasattr(cur, p):
                ok = False
                break
            cur = getattr(cur, p)
        if ok:
            return name, cur
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--max-lines", type=int, default=300)
    parser.add_argument("--only-linear", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModel

    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    vision_path, vision = find_vision(model)
    if vision is None:
        raise RuntimeError("vision module not found")

    print(f"[VisionPath] {vision_path}")
    blocks = None
    for maybe in ["blocks", "layers", "encoder.layers"]:
        cur = vision
        ok = True
        for p in maybe.split("."):
            if not hasattr(cur, p):
                ok = False
                break
            cur = getattr(cur, p)
        if ok and hasattr(cur, "__len__"):
            blocks = (maybe, cur)
            break
    if blocks is not None:
        print(f"[VisionBlocks] pattern={blocks[0]} count={len(blocks[1])}")

    printed = 0
    for n, m in vision.named_modules():
        if args.only_linear and not isinstance(m, torch.nn.Linear):
            continue
        print(f"{n}\t{m.__class__.__name__}")
        printed += 1
        if printed >= args.max_lines:
            print(f"... truncated at {args.max_lines} lines")
            break


if __name__ == "__main__":
    main()
