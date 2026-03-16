#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
from losses import bce_dice_loss, detection_bce_loss, edge_bce_loss, focal_heatmap_loss
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
    edge_kernel_size: int,
    prefetch_factor: int = 1,
    persistent_workers: bool = True,
) -> DataLoader:
    ds = MagicBrushDataset(
        manifest_path=manifest,
        image_size=image_size,
        processor_name_or_path=processor_name_or_path,
        edge_kernel_size=edge_kernel_size,
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


def _lr_scale_for_step(
    optimizer_step: int,
    *,
    scheduler_name: str,
    num_training_steps: int,
    num_warmup_steps: int,
) -> float:
    if num_training_steps <= 0:
        return 1.0
    scheduler_name = scheduler_name.lower()
    if scheduler_name not in {"cosine", "constant", "none"}:
        raise ValueError(f"unsupported lr scheduler: {scheduler_name}")

    step_index = max(0, min(optimizer_step, num_training_steps - 1))
    if num_warmup_steps > 0 and step_index < num_warmup_steps:
        return float(step_index + 1) / float(max(1, num_warmup_steps))
    if scheduler_name == "cosine":
        decay_steps = max(1, num_training_steps - num_warmup_steps)
        progress = float(step_index - num_warmup_steps + 1) / float(decay_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return 1.0


def _set_optimizer_lrs(optimizer: torch.optim.Optimizer, base_lrs: List[float], scale: float) -> None:
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = float(base_lr) * float(scale)


def _get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _iter_named_tensors(obj: Any, prefix: str) -> Iterable[Tuple[str, torch.Tensor]]:
    if torch.is_tensor(obj):
        yield prefix, obj
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_named_tensors(value, child_prefix)
        return
    if isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            child_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            yield from _iter_named_tensors(value, child_prefix)


def _tensor_stats(tensor: torch.Tensor) -> Dict[str, Any]:
    flat = tensor.detach().float()
    finite = torch.isfinite(flat)
    stats = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "nan_count": int(torch.isnan(flat).sum().item()),
        "inf_count": int(torch.isinf(flat).sum().item()),
    }
    if finite.any():
        valid = flat[finite]
        stats["min"] = float(valid.min().item())
        stats["max"] = float(valid.max().item())
        stats["mean"] = float(valid.mean().item())
    else:
        stats["min"] = None
        stats["max"] = None
        stats["mean"] = None
    return stats


def _format_tensor_stats(name: str, tensor: torch.Tensor) -> str:
    stats = _tensor_stats(tensor)
    return (
        f"{name}: shape={stats['shape']} dtype={stats['dtype']} device={stats['device']} "
        f"min={stats['min']} max={stats['max']} mean={stats['mean']} "
        f"nan={stats['nan_count']} inf={stats['inf_count']}"
    )


def _rank_prefix(accelerator: Accelerator) -> str:
    return f"[rank{accelerator.process_index}]"


def _append_debug_line(out_dir: Path, accelerator: Accelerator, message: str) -> None:
    debug_path = out_dir / f"numeric_debug_rank{accelerator.process_index}.log"
    with open(debug_path, "a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def _emit_debug_message(out_dir: Path, accelerator: Accelerator, message: str) -> None:
    line = f"{_rank_prefix(accelerator)} {message}"
    print(line, flush=True)
    _append_debug_line(out_dir, accelerator, line)


def _raise_numeric_error(
    *,
    name: str,
    tensor: torch.Tensor,
    accelerator: Accelerator,
    epoch: int,
    step: int,
    out_dir: Path,
    reason: str,
) -> None:
    message = f"{reason} at epoch={epoch} step={step} {name}; {_format_tensor_stats(name, tensor)}"
    _emit_debug_message(out_dir, accelerator, message)
    raise FloatingPointError(message)


def _ensure_finite(
    *,
    name: str,
    tensor: torch.Tensor,
    accelerator: Accelerator,
    epoch: int,
    step: int,
    out_dir: Path,
) -> None:
    if not torch.isfinite(tensor.detach().float()).all():
        _raise_numeric_error(
            name=name,
            tensor=tensor,
            accelerator=accelerator,
            epoch=epoch,
            step=step,
            out_dir=out_dir,
            reason="Non-finite tensor detected",
        )


def _ensure_probability_tensor(
    *,
    name: str,
    tensor: torch.Tensor,
    accelerator: Accelerator,
    epoch: int,
    step: int,
    out_dir: Path,
    atol: float = 1e-4,
) -> None:
    _ensure_finite(name=name, tensor=tensor, accelerator=accelerator, epoch=epoch, step=step, out_dir=out_dir)
    view = tensor.detach().float()
    min_val = float(view.min().item())
    max_val = float(view.max().item())
    if min_val < -atol or max_val > 1.0 + atol:
        _raise_numeric_error(
            name=name,
            tensor=tensor,
            accelerator=accelerator,
            epoch=epoch,
            step=step,
            out_dir=out_dir,
            reason="BCE probability tensor out of [0, 1] range",
        )


def _collect_nonfinite_named_values(named_tensors: Iterable[Tuple[str, torch.Tensor]], limit: int = 8) -> List[str]:
    bad = []
    for name, tensor in named_tensors:
        if not torch.isfinite(tensor.detach().float()).all():
            bad.append(_format_tensor_stats(name, tensor))
            if len(bad) >= limit:
                break
    return bad


def _collect_nonfinite_gradients(model: torch.nn.Module, limit: int = 8) -> List[str]:
    return _collect_nonfinite_named_values(
        ((f"grad:{name}", param.grad) for name, param in model.named_parameters() if param.grad is not None),
        limit=limit,
    )


def _collect_nonfinite_trainable_params(model: torch.nn.Module, limit: int = 8) -> List[str]:
    return _collect_nonfinite_named_values(
        ((f"param:{name}", param) for name, param in model.named_parameters() if param.requires_grad),
        limit=limit,
    )


def _register_numeric_forward_hooks(model: torch.nn.Module):
    handles = []
    for module_name, module in model.named_modules():
        has_direct_trainable_params = any(param.requires_grad for param in module.parameters(recurse=False))
        is_stage1_top = module_name in {"backbone", "adapter", "proposer", "decoder"}
        if not has_direct_trainable_params and not is_stage1_top:
            continue

        def _hook(_module, _inputs, output, name=module_name):
            for tensor_name, tensor in _iter_named_tensors(output, prefix="output"):
                if not torch.isfinite(tensor.detach().float()).all():
                    raise FloatingPointError(
                        f"Non-finite forward output in module '{name or '<root>'}' "
                        f"({type(_module).__name__}) at {tensor_name}; {_format_tensor_stats(tensor_name, tensor)}"
                    )

        handles.append(module.register_forward_hook(_hook))
    return handles


def _emit_step_debug_snapshot(
    *,
    accelerator: Accelerator,
    out_dir: Path,
    epoch: int,
    step: int,
    values: Dict[str, Optional[torch.Tensor]],
    extra_lines: Optional[List[str]] = None,
) -> None:
    lines = [f"numeric debug snapshot epoch={epoch} step={step}"]
    for name, tensor in values.items():
        if tensor is None:
            continue
        lines.append(_format_tensor_stats(name, tensor))
    if extra_lines:
        lines.extend(extra_lines)
    for line in lines:
        _emit_debug_message(out_dir, accelerator, line)


def run_eval(
    model: torch.nn.Module,
    loader: DataLoader,
    accelerator: Accelerator,
    train_cfg: Dict[str, Any],
    vis_path: Optional[str] = None,
) -> Dict[str, float]:
    model.eval()
    all_prob, all_label, all_pred_mask, all_gt_mask = [], [], [], []
    use_edge_loss = bool(train_cfg.get("use_edge_loss", True))
    loss_sums = {"loss": 0.0, "l_det": 0.0, "l_heat": 0.0, "l_mask": 0.0, "l_edge": 0.0}
    total_items = 0
    vis_saved = False
    with torch.no_grad():
        vbar = tqdm(loader, desc="Val", leave=False, disable=not accelerator.is_local_main_process, dynamic_ncols=True, mininterval=0.0)
        for batch in vbar:
            image = batch["image"]
            pixel_values = batch["pixel_values"]
            image_grid_thw = batch["image_grid_thw"]
            label = batch["label"]
            mask = batch["mask"]
            edge_gt = batch["edge_gt"]
            out = model(pixel_values, image_grid_thw, out_hw=mask.shape[-2:])
            heat_target = F.interpolate(mask, size=out["heatmap"].shape[-2:], mode="nearest")

            l_det = detection_bce_loss(out["p_edit"], label)
            l_heat = focal_heatmap_loss(out["heatmap"], heat_target)
            l_mask = bce_dice_loss(out["mask0"], mask)
            if use_edge_loss and "edge0" in out:
                l_edge = edge_bce_loss(out["edge0"], edge_gt)
            else:
                l_edge = torch.zeros_like(l_det)
            loss = (
                float(train_cfg["lambda_det"]) * l_det
                + float(train_cfg["lambda_heat"]) * l_heat
                + float(train_cfg["lambda_mask"]) * l_mask
                + float(train_cfg.get("lambda_edge", 0.1)) * l_edge
            )

            prob = accelerator.gather_for_metrics(out["p_edit"].detach())
            lab = accelerator.gather_for_metrics(label.detach())
            pred = accelerator.gather_for_metrics(out["mask0"].detach())
            gt = accelerator.gather_for_metrics(mask.detach())
            batch_items = torch.tensor([label.shape[0]], device=label.device, dtype=torch.float32)
            gathered_items = accelerator.gather_for_metrics(batch_items)
            all_prob.append(prob.cpu())
            all_label.append(lab.cpu())
            all_pred_mask.append(pred.cpu())
            all_gt_mask.append(gt.cpu())
            total_items += int(gathered_items.sum().item())
            for name, tensor in {
                "loss": loss,
                "l_det": l_det,
                "l_heat": l_heat,
                "l_mask": l_mask,
                "l_edge": l_edge,
            }.items():
                gathered = accelerator.gather_for_metrics((tensor.detach() * batch_items).view(1))
                loss_sums[name] += float(gathered.sum().item())
            if vis_path and not vis_saved and accelerator.is_main_process:
                save_triplet_vis(
                    image=image.detach().float().cpu(),
                    gt_mask=mask.detach().float().cpu(),
                    heatmap=out["heatmap"].detach().float().cpu(),
                    pred_mask=out["mask0"].detach().float().cpu(),
                    gt_edge=edge_gt.detach().float().cpu(),
                    pred_edge=(torch.sigmoid(out["edge0"]).detach().float().cpu() if "edge0" in out else None),
                    path=vis_path,
                )
                vis_saved = True

    prob = torch.cat(all_prob, dim=0)
    label = torch.cat(all_label, dim=0)
    pred_mask = torch.cat(all_pred_mask, dim=0)
    gt_mask = torch.cat(all_gt_mask, dim=0)
    forged = label >= 0.5
    pixel_all = pixel_metrics(pred_mask, gt_mask)
    pixel_forged = pixel_metrics(pred_mask[forged], gt_mask[forged]) if forged.any() else {"f1": 0.0, "iou": 0.0}
    m = {}
    m.update(binary_auc_ap(prob, label))
    m.update(cls_metrics(prob, label))
    m["f1_all"] = pixel_all["f1"]
    m["iou_all"] = pixel_all["iou"]
    m["f1"] = pixel_forged["f1"]
    m["iou"] = pixel_forged["iou"]
    m["num_forged"] = float(forged.sum().item())
    if total_items > 0:
        m.update({name: loss_sums[name] / float(total_items) for name in loss_sums})
    return m


def run_validation(
    *,
    epoch: int,
    step: int,
    optimizer_step: int,
    train_cfg: Dict[str, Any],
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
    val_metrics = run_eval(
        model,
        val_loader,
        accelerator=accelerator,
        train_cfg=train_cfg,
        vis_path=str(out_dir / "vis" / f"val_step{step}.png"),
    )
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
            optimizer_step=optimizer_step,
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
    optimizer_step: int,
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
            optimizer_step=optimizer_step,
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
            optimizer_step=optimizer_step,
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
        edge_kernel_size=int(train_cfg.get("edge_kernel_size", 5)),
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
        edge_kernel_size=int(train_cfg.get("edge_kernel_size", 5)),
        prefetch_factor=int(cfg["data"].get("prefetch_factor", 1)),
        persistent_workers=bool(cfg["data"].get("persistent_workers", True)),
    )

    model = Stage1ForgeryModel(cfg["model"])
    resume = train_cfg.get("resume", "")
    init_from_checkpoint = train_cfg.get("init_from_checkpoint", "")
    if resume and init_from_checkpoint:
        raise ValueError("train.resume and train.init_from_checkpoint are mutually exclusive")
    if init_from_checkpoint:
        init_ckpt = load_checkpoint(init_from_checkpoint, map_location="cpu")
        load_stage1_checkpoint_into_model(model, init_ckpt)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    model, opt, train_loader, val_loader = accelerator.prepare(model, opt, train_loader, val_loader)

    grad_accum_steps = int(train_cfg.get("gradient_accumulation_steps", 1))
    num_update_steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum_steps))
    total_optimizer_steps = int(train_cfg["epochs"]) * num_update_steps_per_epoch
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.0))
    num_warmup_steps = int(math.ceil(total_optimizer_steps * warmup_ratio))
    scheduler_name = str(train_cfg.get("lr_scheduler", "constant")).lower()
    base_lrs = [float(train_cfg["lr"]) for _ in opt.param_groups]

    start_epoch = 0
    step = 0
    optimizer_step = 0
    best_iou = -1.0
    best_step = -1
    val_every_steps = int(train_cfg.get("val_every_steps", 0))
    val_every_epoch = int(train_cfg.get("val_every_epoch", 0))
    if resume:
        ckpt = load_checkpoint(resume, map_location="cpu")
        if "optimizer" not in ckpt:
            raise ValueError(f"resume checkpoint must be a full training checkpoint, got slim checkpoint: {resume}")
        load_stage1_checkpoint_into_model(accelerator.unwrap_model(model), ckpt)
        try:
            opt.load_state_dict(ckpt["optimizer"])
        except ValueError as exc:
            raise ValueError(
                "resume checkpoint optimizer state is incompatible with the current model. "
                "For adding a new edge head, use train.init_from_checkpoint to load only model weights."
            ) from exc
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        step = int(ckpt.get("step", 0))
        optimizer_step = int(ckpt.get("optimizer_step", step // max(1, grad_accum_steps)))
        best_iou = float(ckpt.get("best_iou", -1.0))
        best_step = int(ckpt.get("best_step", -1))

    current_lr_scale = _lr_scale_for_step(
        optimizer_step=optimizer_step,
        scheduler_name=scheduler_name,
        num_training_steps=total_optimizer_steps,
        num_warmup_steps=num_warmup_steps,
    )
    _set_optimizer_lrs(opt, base_lrs, current_lr_scale)
    accelerator.print(
        f"[Scheduler] type={scheduler_name} warmup_ratio={warmup_ratio} warmup_steps={num_warmup_steps} "
        f"total_optimizer_steps={total_optimizer_steps} grad_accumulation_steps={grad_accum_steps} "
        f"start_lr={_get_current_lr(opt):.6g}"
    )
    if init_from_checkpoint:
        accelerator.print(f"[Init] loaded model weights with strict=False from {init_from_checkpoint}")

    if use_wandb:
        wandb_project = os.environ.get("WANDB_PROJECT") or train_cfg.get("wandb_project") or "aigc_stage1_stage1"
        wandb_name = os.environ.get("WANDB_NAME") or train_cfg.get("wandb_name") or Path(cfg["output_dir"]).name
        accelerator.init_trackers(
            project_name=wandb_project,
            config=cfg,
            init_kwargs={"wandb": {"name": wandb_name}},
        )

    debug_numeric_checks = bool(train_cfg.get("debug_numeric_checks", True))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    use_edge_loss = bool(train_cfg.get("use_edge_loss", True))
    numeric_hook_handles = []
    if debug_numeric_checks:
        numeric_hook_handles = _register_numeric_forward_hooks(accelerator.unwrap_model(model))

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
            image = None
            label = None
            gt_mask = None
            edge_gt = None
            out = None
            heat_target = None
            l_det = None
            l_heat = None
            l_mask = None
            l_edge = None
            loss = None
            grad_norm = None
            applied_lr = _get_current_lr(opt)
            with accelerator.accumulate(model):
                try:
                    image = batch["image"]
                    pixel_values = batch["pixel_values"]
                    image_grid_thw = batch["image_grid_thw"]
                    label = batch["label"]
                    gt_mask = batch["mask"]
                    edge_gt = batch["edge_gt"]
                    out = model(pixel_values, image_grid_thw, out_hw=gt_mask.shape[-2:])
                    heat_target = F.interpolate(gt_mask, size=out["heatmap"].shape[-2:], mode="nearest")

                    if debug_numeric_checks:
                        _ensure_probability_tensor(
                            name="label",
                            tensor=label,
                            accelerator=accelerator,
                            epoch=epoch,
                            step=step,
                            out_dir=out_dir,
                        )
                        _ensure_probability_tensor(
                            name="gt_mask",
                            tensor=gt_mask,
                            accelerator=accelerator,
                            epoch=epoch,
                            step=step,
                            out_dir=out_dir,
                        )
                        _ensure_probability_tensor(
                            name="heat_target",
                            tensor=heat_target,
                            accelerator=accelerator,
                            epoch=epoch,
                            step=step,
                            out_dir=out_dir,
                        )
                        _ensure_finite(
                            name="edge_gt",
                            tensor=edge_gt,
                            accelerator=accelerator,
                            epoch=epoch,
                            step=step,
                            out_dir=out_dir,
                        )
                        _ensure_probability_tensor(
                            name="out.p_edit",
                            tensor=out["p_edit"],
                            accelerator=accelerator,
                            epoch=epoch,
                            step=step,
                            out_dir=out_dir,
                        )
                        _ensure_probability_tensor(
                            name="out.heatmap",
                            tensor=out["heatmap"],
                            accelerator=accelerator,
                            epoch=epoch,
                            step=step,
                            out_dir=out_dir,
                        )
                        _ensure_probability_tensor(
                            name="out.mask0",
                            tensor=out["mask0"],
                            accelerator=accelerator,
                            epoch=epoch,
                            step=step,
                            out_dir=out_dir,
                        )
                        if use_edge_loss and "edge0" in out:
                            _ensure_finite(
                                name="out.edge0",
                                tensor=out["edge0"],
                                accelerator=accelerator,
                                epoch=epoch,
                                step=step,
                                out_dir=out_dir,
                            )

                    l_det = detection_bce_loss(out["p_edit"], label)
                    l_heat = focal_heatmap_loss(out["heatmap"], heat_target)
                    l_mask = bce_dice_loss(out["mask0"], gt_mask)
                    if use_edge_loss and "edge0" in out:
                        l_edge = edge_bce_loss(out["edge0"], edge_gt)
                    else:
                        l_edge = torch.zeros_like(l_det)
                    loss = (
                        float(train_cfg["lambda_det"]) * l_det
                        + float(train_cfg["lambda_heat"]) * l_heat
                        + float(train_cfg["lambda_mask"]) * l_mask
                        + float(train_cfg.get("lambda_edge", 0.1)) * l_edge
                    )

                    if debug_numeric_checks:
                        _ensure_finite(
                            name="loss/det",
                            tensor=l_det,
                            accelerator=accelerator,
                            epoch=epoch,
                            step=step,
                            out_dir=out_dir,
                        )
                        _ensure_finite(
                            name="loss/heat",
                            tensor=l_heat,
                            accelerator=accelerator,
                            epoch=epoch,
                            step=step,
                            out_dir=out_dir,
                        )
                        _ensure_finite(
                            name="loss/mask",
                            tensor=l_mask,
                            accelerator=accelerator,
                            epoch=epoch,
                            step=step,
                            out_dir=out_dir,
                        )
                        _ensure_finite(
                            name="loss/edge",
                            tensor=l_edge,
                            accelerator=accelerator,
                            epoch=epoch,
                            step=step,
                            out_dir=out_dir,
                        )
                        _ensure_finite(
                            name="loss/total",
                            tensor=loss,
                            accelerator=accelerator,
                            epoch=epoch,
                            step=step,
                            out_dir=out_dir,
                        )

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        if max_grad_norm > 0:
                            grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                            grad_norm_tensor = torch.as_tensor(grad_norm, device=accelerator.device).float()
                            if debug_numeric_checks and not torch.isfinite(grad_norm_tensor).all():
                                raise FloatingPointError(
                                    f"Non-finite grad norm detected at epoch={epoch} step={step}: {float(grad_norm_tensor.item())}"
                                )

                        if debug_numeric_checks:
                            bad_grads = _collect_nonfinite_gradients(accelerator.unwrap_model(model))
                            if bad_grads:
                                raise FloatingPointError(
                                    "Non-finite gradients detected after backward: " + " | ".join(bad_grads)
                                )

                        lr_scale = _lr_scale_for_step(
                            optimizer_step=optimizer_step,
                            scheduler_name=scheduler_name,
                            num_training_steps=total_optimizer_steps,
                            num_warmup_steps=num_warmup_steps,
                        )
                        _set_optimizer_lrs(opt, base_lrs, lr_scale)
                        applied_lr = _get_current_lr(opt)
                        opt.step()
                        optimizer_step += 1

                        if debug_numeric_checks:
                            bad_params = _collect_nonfinite_trainable_params(accelerator.unwrap_model(model))
                            if bad_params:
                                raise FloatingPointError(
                                    "Non-finite trainable parameters detected after optimizer step: "
                                    + " | ".join(bad_params)
                                )

                        next_lr_scale = _lr_scale_for_step(
                            optimizer_step=optimizer_step,
                            scheduler_name=scheduler_name,
                            num_training_steps=total_optimizer_steps,
                            num_warmup_steps=num_warmup_steps,
                        )
                        _set_optimizer_lrs(opt, base_lrs, next_lr_scale)
                        opt.zero_grad(set_to_none=True)
                except Exception as exc:
                    extra_lines = []
                    if grad_norm is not None:
                        extra_lines.append(f"grad_norm={float(torch.as_tensor(grad_norm).float().item())}")
                    if debug_numeric_checks:
                        bad_grads = _collect_nonfinite_gradients(accelerator.unwrap_model(model))
                        if bad_grads:
                            extra_lines.append("bad_grads=" + " | ".join(bad_grads))
                        bad_params = _collect_nonfinite_trainable_params(accelerator.unwrap_model(model))
                        if bad_params:
                            extra_lines.append("bad_params=" + " | ".join(bad_params))
                    _emit_step_debug_snapshot(
                        accelerator=accelerator,
                        out_dir=out_dir,
                        epoch=epoch,
                        step=step,
                        values={
                            "label": label,
                            "gt_mask": gt_mask,
                            "edge_gt": edge_gt,
                            "heat_target": heat_target,
                            "out.p_edit": None if out is None else out.get("p_edit"),
                            "out.heatmap": None if out is None else out.get("heatmap"),
                            "out.mask0": None if out is None else out.get("mask0"),
                            "out.edge0": None if out is None else out.get("edge0"),
                            "loss/det": l_det,
                            "loss/heat": l_heat,
                            "loss/mask": l_mask,
                            "loss/edge": l_edge,
                            "loss/total": loss,
                        },
                        extra_lines=extra_lines + [f"exception={type(exc).__name__}: {exc}"],
                    )
                    raise

            step += 1
            current_lr = applied_lr

            if accelerator.is_local_main_process:
                pbar.update(1)
                pbar.set_postfix(
                    loss=f"{float(loss.item()):.4f}",
                    det=f"{float(l_det.item()):.4f}",
                    heat=f"{float(l_heat.item()):.4f}",
                    mask=f"{float(l_mask.item()):.4f}",
                    edge=f"{float(l_edge.item()):.4f}",
                    lr=f"{current_lr:.2e}",
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
                                "l_edge": float(l_edge.item()),
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
                            "train/l_edge": float(l_edge.item()),
                            "train/lr": current_lr,
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
                    gt_edge=edge_gt[:nvis].detach().float().cpu(),
                    pred_edge=(torch.sigmoid(out["edge0"][:nvis]).detach().float().cpu() if "edge0" in out else None),
                    path=str(out_dir / "vis" / f"train_step{step}.png"),
                )

            if val_every_epoch == 0 and val_every_steps > 0 and step % val_every_steps == 0:
                best_iou, best_step = run_validation(
                    epoch=epoch,
                    step=step,
                    optimizer_step=optimizer_step,
                    train_cfg=train_cfg,
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
                optimizer_step=optimizer_step,
                train_cfg=train_cfg,
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
            optimizer_step=optimizer_step,
            epoch=epoch,
            step=step,
            best_iou=best_iou,
            best_step=best_step,
            cfg=cfg,
        )

    if use_wandb:
        accelerator.end_training()
    for handle in numeric_hook_handles:
        handle.remove()


if __name__ == "__main__":
    main()
