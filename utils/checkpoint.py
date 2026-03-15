from pathlib import Path
from typing import Any, Dict, Optional

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
    epoch: int,
    step: int,
    best_iou: float,
    best_step: int,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    slim_payload = build_slim_checkpoint_payload(
        model=model,
        epoch=epoch,
        step=step,
        best_iou=best_iou,
        best_step=best_step,
        cfg=cfg,
    )
    return {
        "format": "stage1_resume_v2",
        **slim_payload,
        "optimizer": optimizer.state_dict(),
    }


def build_slim_checkpoint_payload(
    *,
    model,
    epoch: int,
    step: int,
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
    lora_state = model.backbone.get_lora_state_dict()
    if lora_state:
        payload["backbone_lora"] = lora_state
        trainable_backbone = {}
    else:
        trainable_backbone = model.backbone.get_trainable_state_dict()
    if trainable_backbone:
        payload["backbone_trainable"] = trainable_backbone
    return payload


def load_stage1_checkpoint_into_model(model, checkpoint: Dict[str, Any]) -> None:
    fmt = checkpoint.get("format", "")
    if fmt == "stage1_full_v1" or "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
        return
    if fmt not in {"stage1_slim_v1", "stage1_resume_v2"}:
        raise ValueError(f"unsupported checkpoint format: {fmt or 'unknown'}")
    model.adapter.load_state_dict(checkpoint["adapter"], strict=False)
    model.proposer.load_state_dict(checkpoint["proposer"], strict=False)
    model.decoder.load_state_dict(checkpoint["decoder"], strict=False)
    model.backbone.load_lora_state_dict(checkpoint.get("backbone_lora", {}))
    model.backbone.load_trainable_state_dict(checkpoint.get("backbone_trainable", {}))
