#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aigc_datasets.magicbrush_dataset import MagicBrushDataset
from models.stage1_model import Stage1ForgeryModel
from utils.checkpoint import load_checkpoint
from utils.metrics import binary_auc_ap, cls_metrics, pixel_metrics
from utils.vis import save_triplet_vis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint or str(Path(cfg["output_dir"]) / "best_by_iou.pt")

    ds = MagicBrushDataset(manifest_path=cfg["data"]["manifests"][args.split], image_size=cfg["data"]["image_size"])
    loader = DataLoader(
        ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    model = Stage1ForgeryModel(cfg["model"]).to(device)
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    probs, labels, pred_masks, gt_masks = [], [], [], []
    vis_saved = False
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval-{args.split}"):
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            mask = batch["mask"].to(device)
            out = model(image)
            probs.append(out["p_edit"].cpu())
            labels.append(label.cpu())
            pred_masks.append(out["mask0"].cpu())
            gt_masks.append(mask.cpu())
            if not vis_saved:
                save_triplet_vis(
                    image.cpu(),
                    gt_mask=mask.cpu(),
                    heatmap=out["heatmap"].cpu(),
                    pred_mask=out["mask0"].cpu(),
                    path=str(Path(cfg["output_dir"]) / f"{args.split}_vis.png"),
                )
                vis_saved = True

    prob = torch.cat(probs, dim=0)
    label = torch.cat(labels, dim=0)
    pred_mask = torch.cat(pred_masks, dim=0)
    gt_mask = torch.cat(gt_masks, dim=0)

    metrics = {}
    metrics.update(binary_auc_ap(prob, label))
    metrics.update(cls_metrics(prob, label))
    metrics.update(pixel_metrics(pred_mask, gt_mask))
    print(json.dumps({args.split: metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
