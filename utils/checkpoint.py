from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, p)


def load_checkpoint(path: str, map_location: Optional[str] = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


def build_full_checkpoint_payload(
    *,
    model,
    optimizer,
    scheduler=None,
    epoch: int,
    step: int,
    optimizer_step: Optional[int] = None,
    best_iou: float,
    best_step: int,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    slim_payload = build_slim_checkpoint_payload(
        model=model,
        epoch=epoch,
        step=step,
        optimizer_step=optimizer_step,
        best_iou=best_iou,
        best_step=best_step,
        cfg=cfg,
    )
    payload = {
        "format": "stage1_resume_v2",
        **slim_payload,
        "optimizer": optimizer.state_dict(),
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    return payload


def build_slim_checkpoint_payload(
    *,
    model,
    epoch: int,
    step: int,
    optimizer_step: Optional[int] = None,
    best_iou: float,
    best_step: int,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "format": "stage1_slim_v1",
        "adapter": model.adapter.state_dict(),
        "proposer": model.proposer.state_dict(),
        "decoder": model.decoder.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_iou": best_iou,
        "best_step": best_step,
        "cfg": cfg,
    }
    if optimizer_step is not None:
        payload["optimizer_step"] = optimizer_step
    lora_state = model.backbone.get_lora_state_dict()
    if lora_state:
        payload["backbone_lora"] = lora_state
        trainable_backbone = {}
    else:
        trainable_backbone = model.backbone.get_trainable_state_dict()
    if trainable_backbone:
        payload["backbone_trainable"] = trainable_backbone
    return payload


def load_stage1_checkpoint_into_model(model, checkpoint: Dict[str, Any]) -> Dict[str, List[str]]:
    info: Dict[str, List[str]] = {"missing": [], "unexpected": []}
    fmt = checkpoint.get("format", "")
    if fmt == "stage1_full_v1" or "model" in checkpoint:
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        info["missing"].extend(list(missing))
        info["unexpected"].extend(list(unexpected))
        return info
    if fmt not in {"stage1_slim_v1", "stage1_resume_v2"}:
        raise ValueError(f"unsupported checkpoint format: {fmt or 'unknown'}")
    for module_name, module, state in [
        ("adapter", model.adapter, checkpoint["adapter"]),
        ("proposer", model.proposer, checkpoint["proposer"]),
        ("decoder", model.decoder, checkpoint["decoder"]),
    ]:
        missing, unexpected = module.load_state_dict(state, strict=False)
        info["missing"].extend([f"{module_name}.{name}" for name in missing])
        info["unexpected"].extend([f"{module_name}.{name}" for name in unexpected])
    model.backbone.load_lora_state_dict(checkpoint.get("backbone_lora", {}))
    model.backbone.load_trainable_state_dict(checkpoint.get("backbone_trainable", {}))
    return info
