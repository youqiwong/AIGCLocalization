#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aigc_datasets.magicbrush_dataset import MagicBrushDataset, collate_magicbrush_batch
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


def build_loader(
    manifest: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    processor_name_or_path: str,
) -> DataLoader:
    ds = MagicBrushDataset(
        manifest_path=manifest,
        image_size=image_size,
        processor_name_or_path=processor_name_or_path,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_magicbrush_batch,
    )


def run_eval(model: torch.nn.Module, loader: DataLoader, accelerator: Accelerator) -> Dict[str, float]:
    model.eval()
    all_prob, all_label, all_pred_mask, all_gt_mask = [], [], [], []
    with torch.no_grad():
        vbar = tqdm(loader, desc="Val", leave=False, disable=not accelerator.is_local_main_process)
        for batch in vbar:
            image = batch["image"]
            pixel_values = batch["pixel_values"]
            image_grid_thw = batch["image_grid_thw"]
            label = batch["label"]
            mask = batch["mask"]
            out = model(pixel_values, image_grid_thw, out_hw=mask.shape[-2:])

            prob = accelerator.gather_for_metrics(out["p_edit"].detach())
            lab = accelerator.gather_for_metrics(label.detach())
            pred = accelerator.gather_for_metrics(out["mask0"].detach())
            gt = accelerator.gather_for_metrics(mask.detach())
            all_prob.append(prob.cpu())
            all_label.append(lab.cpu())
            all_pred_mask.append(pred.cpu())
            all_gt_mask.append(gt.cpu())

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

    train_cfg = cfg["train"]
    use_wandb = bool(train_cfg.get("use_wandb", True))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=bool(train_cfg.get("find_unused_parameters", True)))
    accelerator = Accelerator(
        mixed_precision=train_cfg.get("mixed_precision", "bf16"),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
        log_with=("wandb" if use_wandb else None),
        kwargs_handlers=[ddp_kwargs],
    )
    accelerator.print(
        f"[Accelerate] num_processes={accelerator.num_processes} "
        f"device={accelerator.device} mixed_precision={accelerator.mixed_precision}"
    )

    train_loader = build_loader(
        manifest=cfg["data"]["manifests"]["train"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        shuffle=True,
        processor_name_or_path=cfg["model"]["backbone"]["name_or_path"],
    )
    val_loader = build_loader(
        manifest=cfg["data"]["manifests"]["val"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        shuffle=False,
        processor_name_or_path=cfg["model"]["backbone"]["name_or_path"],
    )

    model = Stage1ForgeryModel(cfg["model"])
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    model, opt, train_loader, val_loader = accelerator.prepare(model, opt, train_loader, val_loader)

    start_epoch = 0
    step = 0
    best_iou = -1.0
    best_step = -1
    val_every_steps = int(train_cfg.get("val_every_steps", 500))
    resume = train_cfg.get("resume", "")
    if resume:
        ckpt = load_checkpoint(resume, map_location="cpu")
        accelerator.unwrap_model(model).load_state_dict(ckpt["model"], strict=False)
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        step = int(ckpt.get("step", 0))
        best_iou = float(ckpt.get("best_iou", -1.0))
        best_step = int(ckpt.get("best_step", -1))

    if use_wandb:
        wandb_project = os.environ.get("WANDB_PROJECT") or train_cfg.get("wandb_project") or "aigc_stage1_stage1"
        wandb_name = os.environ.get("WANDB_NAME") or train_cfg.get("wandb_name") or Path(cfg["output_dir"]).name
        accelerator.init_trackers(
            project_name=wandb_project,
            config=cfg,
            init_kwargs={"wandb": {"name": wandb_name}},
        )

    for epoch in range(start_epoch, int(train_cfg["epochs"])):
        model.train()
        last_eval_step = -1
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=True, disable=not accelerator.is_local_main_process)
        for batch in pbar:
            with accelerator.accumulate(model):
                image = batch["image"]
                pixel_values = batch["pixel_values"]
                image_grid_thw = batch["image_grid_thw"]
                label = batch["label"]
                gt_mask = batch["mask"]
                opt.zero_grad(set_to_none=True)
                out = model(pixel_values, image_grid_thw, out_hw=gt_mask.shape[-2:])
                heat_target = F.interpolate(gt_mask, size=out["heatmap"].shape[-2:], mode="nearest")
                l_det = detection_bce_loss(out["p_edit"], label)
                l_heat = focal_heatmap_loss(out["heatmap"], heat_target)
                l_mask = bce_dice_loss(out["mask0"], gt_mask)
                loss = (
                    float(train_cfg["lambda_det"]) * l_det
                    + float(train_cfg["lambda_heat"]) * l_heat
                    + float(train_cfg["lambda_mask"]) * l_mask
                )
                accelerator.backward(loss)
                opt.step()

            pbar.set_postfix(
                loss=f"{float(loss.item()):.4f}",
                det=f"{float(l_det.item()):.4f}",
                heat=f"{float(l_heat.item()):.4f}",
                mask=f"{float(l_mask.item()):.4f}",
                step=step,
            )

            if step % int(train_cfg["log_every"]) == 0:
                accelerator.print(
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
                if use_wandb:
                    accelerator.log(
                        {
                            "train/loss": float(loss.item()),
                            "train/l_det": float(l_det.item()),
                            "train/l_heat": float(l_heat.item()),
                            "train/l_mask": float(l_mask.item()),
                            "epoch": epoch,
                        },
                        step=step,
                    )

            if step % int(train_cfg["vis_every"]) == 0 and accelerator.is_main_process:
                save_triplet_vis(
                    image=accelerator.gather_for_metrics(image.detach()).cpu(),
                    gt_mask=accelerator.gather_for_metrics(gt_mask.detach()).cpu(),
                    heatmap=accelerator.gather_for_metrics(out["heatmap"].detach()).cpu(),
                    pred_mask=accelerator.gather_for_metrics(out["mask0"].detach()).cpu(),
                    path=str(out_dir / "vis" / f"train_step{step}.png"),
                )

            if (step + 1) % val_every_steps == 0:
                accelerator.wait_for_everyone()
                val_metrics = run_eval(model, val_loader, accelerator=accelerator)
                accelerator.print(json.dumps({"epoch": epoch, "step": step + 1, "val": val_metrics}, ensure_ascii=False))
                if use_wandb:
                    accelerator.log({f"val/{k}": float(v) for k, v in val_metrics.items()}, step=step + 1)
                if accelerator.is_main_process:
                    save_checkpoint(
                        str(out_dir / "last.pt"),
                        {
                            "model": accelerator.unwrap_model(model).state_dict(),
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
                    if accelerator.is_main_process:
                        save_checkpoint(
                            str(out_dir / "best_by_iou.pt"),
                            {
                                "model": accelerator.unwrap_model(model).state_dict(),
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
            accelerator.wait_for_everyone()
            val_metrics = run_eval(model, val_loader, accelerator=accelerator)
            accelerator.print(json.dumps({"epoch": epoch, "step": step, "val": val_metrics}, ensure_ascii=False))
            if use_wandb:
                accelerator.log({f"val/{k}": float(v) for k, v in val_metrics.items()}, step=step)
            if val_metrics["iou"] > best_iou:
                best_iou = val_metrics["iou"]
                best_step = step
                if accelerator.is_main_process:
                    save_checkpoint(
                        str(out_dir / "best_by_iou.pt"),
                        {
                            "model": accelerator.unwrap_model(model).state_dict(),
                            "optimizer": opt.state_dict(),
                            "epoch": epoch,
                            "step": step,
                            "best_iou": best_iou,
                            "best_step": best_step,
                        },
                    )
        if accelerator.is_main_process:
            save_checkpoint(
                str(out_dir / "last.pt"),
                {
                    "model": accelerator.unwrap_model(model).state_dict(),
                    "optimizer": opt.state_dict(),
                    "epoch": epoch,
                    "step": step,
                    "best_iou": best_iou,
                    "best_step": best_step,
                },
            )

    if use_wandb:
        accelerator.end_training()


if __name__ == "__main__":
    main()
