#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision.transforms import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.stage1_model import Stage1ForgeryModel
from utils.checkpoint import load_checkpoint, load_stage1_checkpoint_into_model
from utils.vis import save_triplet_vis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/infer_stage1")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = Stage1ForgeryModel(cfg["model"]).to(device)
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    load_stage1_checkpoint_into_model(model, ckpt)
    model.eval()

    img = Image.open(args.image).convert("RGB")
    h, w = img.height, img.width
    resized = F.resize(img, [cfg["data"]["image_size"], cfg["data"]["image_size"]], antialias=True)
    image_vis = F.to_tensor(resized).unsqueeze(0)
    from transformers import AutoImageProcessor

    image_processor = AutoImageProcessor.from_pretrained(cfg["model"]["backbone"]["name_or_path"], trust_remote_code=True)
    image_inputs = image_processor(images=resized, do_resize=False, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(device)
    image_grid_thw = image_inputs["image_grid_thw"].to(device)
    with torch.no_grad():
        out = model(pixel_values, image_grid_thw, out_hw=image_vis.shape[-2:])
    p_edit = float(out["p_edit"][0].item())
    mask = out["mask0"][0:1].cpu()
    heatmap = out["heatmap"][0:1].cpu()
    save_triplet_vis(
        image=image_vis.cpu(),
        gt_mask=torch.zeros_like(mask),
        heatmap=heatmap,
        pred_mask=mask,
        path=str(out_dir / "prediction.png"),
        max_items=1,
    )
    regions = out["candidate_regions"][0].detach().cpu().tolist()
    print(json.dumps({"p_edit": p_edit, "candidate_regions": regions, "image_hw": [h, w]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
