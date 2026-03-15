#!/usr/bin/env python3
import argparse
import datetime as dt
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
from accelerate.utils import DistributedDataParallelKwargs, broadcast_object_list
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aigc_datasets.magicbrush_dataset import MagicBrushDataset, collate_magicbrush_batch
from losses import bce_dice_loss, detection_bce_loss, focal_heatmap_loss
from models.stage1_model import Stage1ForgeryModel
from utils.checkpoint import (
    build_full_checkpoint_payload,
    build_slim_checkpoint_payload,
    load_checkpoint,
    load_stage1_checkpoint_into_model,
    save_checkpoint,
)
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
    prefetch_factor: int = 1,
    persistent_workers: bool = True,
) -> DataLoader:
    ds = MagicBrushDataset(
        manifest_path=manifest,
        image_size=image_size,
        processor_name_or_path=processor_name_or_path,
    )
    loader_kwargs = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_magicbrush_batch,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers
    return DataLoader(**loader_kwargs)


def run_eval(model: torch.nn.Module, loader: DataLoader, accelerator: Accelerator) -> Dict[str, float]:
    model.eval()
    all_prob, all_label, all_pred_mask, all_gt_mask = [], [], [], []
    with torch.no_grad():
        vbar = tqdm(loader, desc="Val", leave=False, disable=not accelerator.is_local_main_process, dynamic_ncols=True, mininterval=0.0)
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


def run_validation(
    *,
    epoch: int,
    step: int,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    val_loader: DataLoader,
    accelerator: Accelerator,
    use_wandb: bool,
    out_dir: Path,
    best_iou: float,
    best_step: int,
    cfg: Dict,
):
    accelerator.wait_for_everyone()
    val_metrics = run_eval(model, val_loader, accelerator=accelerator)
    accelerator.print(json.dumps({"epoch": epoch, "step": step, "val": val_metrics}, ensure_ascii=False))
    if use_wandb:
        accelerator.log({f"val/{k}": float(v) for k, v in val_metrics.items()}, step=step)
    if val_metrics["iou"] > best_iou:
        best_iou = val_metrics["iou"]
        best_step = step
        save_stage1_checkpoints(
            tag="best_by_iou",
            out_dir=out_dir,
            accelerator=accelerator,
            model=model,
            opt=opt,
            epoch=epoch,
            step=step,
            best_iou=best_iou,
            best_step=best_step,
            cfg=cfg,
        )
    model.train()
    return best_iou, best_step


def save_stage1_checkpoints(
    *,
    tag: str,
    out_dir: Path,
    accelerator: Accelerator,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    epoch: int,
    step: int,
    best_iou: float,
    best_step: int,
    cfg: Dict,
) -> None:
    if not accelerator.is_main_process:
        return
    unwrapped = accelerator.unwrap_model(model)
    save_checkpoint(
        str(out_dir / f"{tag}.pt"),
        build_slim_checkpoint_payload(
            model=unwrapped,
            epoch=epoch,
            step=step,
            best_iou=best_iou,
            best_step=best_step,
            cfg=cfg,
        ),
    )
    save_checkpoint(
        str(out_dir / f"{tag}_full.pt"),
        build_full_checkpoint_payload(
            model=unwrapped,
            optimizer=opt,
            epoch=epoch,
            step=step,
            best_iou=best_iou,
            best_step=best_step,
            cfg=cfg,
        ),
    )


def make_timestamped_output_dir(base_output_dir: str, accelerator: Accelerator, resume: str = "") -> Path:
    if resume:
        return Path(resume).resolve().parent
    out_dir_obj = None
    if accelerator.is_main_process:
        root = Path(base_output_dir)
        root.mkdir(parents=True, exist_ok=True)
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_dir_obj = root / ts
        out_dir_obj.mkdir(parents=True, exist_ok=False)
    out_dir_list = [str(out_dir_obj) if out_dir_obj is not None else ""]
    if accelerator.num_processes > 1:
        broadcast_object_list(out_dir_list)
    accelerator.wait_for_everyone()
    return Path(out_dir_list[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(int(cfg.get("seed", 42)))
    train_cfg = cfg["train"]
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=bool(train_cfg.get("find_unused_parameters", True)))
    accelerator = Accelerator(
        mixed_precision=train_cfg.get("mixed_precision", "bf16"),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
        log_with=("wandb" if bool(train_cfg.get("use_wandb", True)) else None),
        kwargs_handlers=[ddp_kwargs],
    )
    out_dir = make_timestamped_output_dir(cfg["output_dir"], accelerator=accelerator, resume=train_cfg.get("resume", ""))
    cfg["output_dir"] = str(out_dir)
    if accelerator.is_main_process:
        with open(out_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    accelerator.wait_for_everyone()
    use_wandb = bool(train_cfg.get("use_wandb", True))
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
        prefetch_factor=int(cfg["data"].get("prefetch_factor", 1)),
        persistent_workers=bool(cfg["data"].get("persistent_workers", True)),
    )
    val_loader = build_loader(
        manifest=cfg["data"]["manifests"]["val"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        shuffle=False,
        processor_name_or_path=cfg["model"]["backbone"]["name_or_path"],
        prefetch_factor=int(cfg["data"].get("prefetch_factor", 1)),
        persistent_workers=bool(cfg["data"].get("persistent_workers", True)),
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
    val_every_steps = int(train_cfg.get("val_every_steps", 0))
    val_every_epoch = int(train_cfg.get("val_every_epoch", 0))
    resume = train_cfg.get("resume", "")
    if resume:
        ckpt = load_checkpoint(resume, map_location="cpu")
        if "optimizer" not in ckpt:
            raise ValueError(f"resume checkpoint must be a full training checkpoint, got slim checkpoint: {resume}")
        load_stage1_checkpoint_into_model(accelerator.unwrap_model(model), ckpt)
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
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(
            total=len(train_loader),
            desc=f"Train Epoch {epoch}",
            leave=True,
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
            mininterval=0.1,
        )
        for batch in train_loader:
            with accelerator.accumulate(model):
                image = batch["image"]
                pixel_values = batch["pixel_values"]
                image_grid_thw = batch["image_grid_thw"]
                label = batch["label"]
                gt_mask = batch["mask"]
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
                opt.zero_grad(set_to_none=True)

            step += 1

            if accelerator.is_local_main_process:
                pbar.update(1)
                pbar.set_postfix(
                    loss=f"{float(loss.item()):.4f}",
                    det=f"{float(l_det.item()):.4f}",
                    heat=f"{float(l_heat.item()):.4f}",
                    mask=f"{float(l_mask.item()):.4f}",
                    step=step,
                    refresh=True,
                )

            if step % int(train_cfg["log_every"]) == 0:
                if bool(train_cfg.get("print_train_log", False)):
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

            vis_every = int(train_cfg.get("vis_every", 0))
            if vis_every > 0 and step % vis_every == 0 and accelerator.is_main_process:
                nvis = min(5, image.shape[0])
                save_triplet_vis(
                    image=image[:nvis].detach().float().cpu(),
                    gt_mask=gt_mask[:nvis].detach().float().cpu(),
                    heatmap=out["heatmap"][:nvis].detach().float().cpu(),
                    pred_mask=out["mask0"][:nvis].detach().float().cpu(),
                    path=str(out_dir / "vis" / f"train_step{step}.png"),
                )

            if val_every_epoch == 0 and val_every_steps > 0 and step % val_every_steps == 0:
                best_iou, best_step = run_validation(
                    epoch=epoch,
                    step=step,
                    model=model,
                    opt=opt,
                    val_loader=val_loader,
                    accelerator=accelerator,
                    use_wandb=use_wandb,
                    out_dir=out_dir,
                    best_iou=best_iou,
                    best_step=best_step,
                    cfg=cfg,
                )
                last_eval_step = step

        should_run_epoch_val = val_every_epoch > 0 and (epoch + 1) % val_every_epoch == 0
        should_run_step_fallback = val_every_epoch == 0 and (val_every_steps <= 0 or last_eval_step != step)
        if should_run_epoch_val or should_run_step_fallback:
            best_iou, best_step = run_validation(
                epoch=epoch,
                step=step,
                model=model,
                opt=opt,
                val_loader=val_loader,
                accelerator=accelerator,
                use_wandb=use_wandb,
                out_dir=out_dir,
                best_iou=best_iou,
                best_step=best_step,
                cfg=cfg,
            )
        pbar.close()
        save_stage1_checkpoints(
            tag="last",
            out_dir=out_dir,
            accelerator=accelerator,
            model=model,
            opt=opt,
            epoch=epoch,
            step=step,
            best_iou=best_iou,
            best_step=best_step,
            cfg=cfg,
        )

    if use_wandb:
        accelerator.end_training()


if __name__ == "__main__":
    main()
