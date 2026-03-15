#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.magicbrush_dataset import MagicBrushDataset
from losses import bce_dice_loss, detection_bce_loss, focal_heatmap_loss
from models.stage1_model import Stage1ForgeryModel
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.metrics import binary_auc_ap, cls_metrics, pixel_metrics
from utils.vis import save_triplet_vis


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loader(manifest: str, image_size: int, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    ds = MagicBrushDataset(manifest_path=manifest, image_size=image_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def run_eval(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_prob, all_label, all_pred_mask, all_gt_mask = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            mask = batch["mask"].to(device)
            out = model(image)
            all_prob.append(out["p_edit"].detach().cpu())
            all_label.append(label.detach().cpu())
            all_pred_mask.append(out["mask0"].detach().cpu())
            all_gt_mask.append(mask.detach().cpu())
    prob = torch.cat(all_prob, dim=0)
    label = torch.cat(all_label, dim=0)
    pred_mask = torch.cat(all_pred_mask, dim=0)
    gt_mask = torch.cat(all_gt_mask, dim=0)
    m = {}
    m.update(binary_auc_ap(prob, label))
    m.update(cls_metrics(prob, label))
    m.update(pixel_metrics(pred_mask, gt_mask))
    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(int(cfg.get("seed", 42)))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    train_loader = build_loader(
        manifest=cfg["data"]["manifests"]["train"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        shuffle=True,
    )
    val_loader = build_loader(
        manifest=cfg["data"]["manifests"]["val"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        shuffle=False,
    )

    model = Stage1ForgeryModel(cfg["model"]).to(device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"].get("amp", True) and device.type == "cuda"))

    start_epoch = 0
    step = 0
    best_iou = -1.0
    best_step = -1
    val_every_steps = int(cfg["train"].get("val_every_steps", 500))
    resume = cfg["train"].get("resume", "")
    if resume:
        ckpt = load_checkpoint(resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        step = int(ckpt.get("step", 0))
        best_iou = float(ckpt.get("best_iou", -1.0))
        best_step = int(ckpt.get("best_step", -1))
    for epoch in range(start_epoch, int(cfg["train"]["epochs"])):
        model.train()
        last_eval_step = -1
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=True)
        for batch in pbar:
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            gt_mask = batch["mask"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(image)
                heat_target = F.interpolate(gt_mask, size=out["heatmap"].shape[-2:], mode="nearest")
                l_det = detection_bce_loss(out["p_edit"], label)
                l_heat = focal_heatmap_loss(out["heatmap"], heat_target)
                l_mask = bce_dice_loss(out["mask0"], gt_mask)
                loss = (
                    float(cfg["train"]["lambda_det"]) * l_det
                    + float(cfg["train"]["lambda_heat"]) * l_heat
                    + float(cfg["train"]["lambda_mask"]) * l_mask
                )
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            pbar.set_postfix(
                loss=f"{float(loss.item()):.4f}",
                det=f"{float(l_det.item()):.4f}",
                heat=f"{float(l_heat.item()):.4f}",
                mask=f"{float(l_mask.item()):.4f}",
                step=step,
            )

            if step % int(cfg["train"]["log_every"]) == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": step,
                            "loss": float(loss.item()),
                            "l_det": float(l_det.item()),
                            "l_heat": float(l_heat.item()),
                            "l_mask": float(l_mask.item()),
                        }
                    )
                )
            if step % int(cfg["train"]["vis_every"]) == 0:
                save_triplet_vis(
                    image=image.detach().cpu(),
                    gt_mask=gt_mask.detach().cpu(),
                    heatmap=out["heatmap"].detach().cpu(),
                    pred_mask=out["mask0"].detach().cpu(),
                    path=str(out_dir / "vis" / f"train_step{step}.png"),
                )
            if (step + 1) % val_every_steps == 0:
                val_metrics = run_eval(model, val_loader, device=device)
                print(json.dumps({"epoch": epoch, "step": step + 1, "val": val_metrics}, ensure_ascii=False))
                save_checkpoint(
                    str(out_dir / "last.pt"),
                    {
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "epoch": epoch,
                        "step": step + 1,
                        "best_iou": best_iou,
                        "best_step": best_step,
                    },
                )
                if val_metrics["iou"] > best_iou:
                    best_iou = val_metrics["iou"]
                    best_step = step + 1
                    save_checkpoint(
                        str(out_dir / "best_by_iou.pt"),
                        {
                            "model": model.state_dict(),
                            "optimizer": opt.state_dict(),
                            "epoch": epoch,
                            "step": step + 1,
                            "best_iou": best_iou,
                            "best_step": best_step,
                        },
                    )
                model.train()
                last_eval_step = step + 1
            step += 1

        if last_eval_step != step:
            val_metrics = run_eval(model, val_loader, device=device)
            print(json.dumps({"epoch": epoch, "step": step, "val": val_metrics}, ensure_ascii=False))
            if val_metrics["iou"] > best_iou:
                best_iou = val_metrics["iou"]
                best_step = step
                save_checkpoint(
                    str(out_dir / "best_by_iou.pt"),
                    {
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "epoch": epoch,
                        "step": step,
                        "best_iou": best_iou,
                        "best_step": best_step,
                    },
                )
        save_checkpoint(
            str(out_dir / "last.pt"),
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "epoch": epoch,
                "step": step,
                "best_iou": best_iou,
                "best_step": best_step,
            },
        )


if __name__ == "__main__":
    main()
