import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MockBackbone(nn.Module):
    def __init__(self, in_ch: int = 3, base_ch: int = 64):
        super().__init__()
        self.s1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.s2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
        )
        self.s3 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(inplace=True),
        )
        self.s4 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 8),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [B,3,H,W]
        l1 = self.s1(x)  # [B,C1,H/2,W/2]
        l2 = self.s2(l1)  # [B,C2,H/4,W/4]
        l3 = self.s3(l2)  # [B,C3,H/8,W/8]
        l4 = self.s4(l3)  # [B,C4,H/16,W/16]
        return {"feat_l1": l1, "feat_l2": l2, "feat_l3": l3, "feat_l4": l4}


def _reshape_tokens_to_2d(tokens: torch.Tensor) -> torch.Tensor:
    # tokens: [B,N,C] -> [B,C,h,w]
    b, n, c = tokens.shape
    side = int(n**0.5)
    if side * side != n:
        side = int((n // 2) ** 0.5)
        if side * side == n - 1:
            tokens = tokens[:, 1:, :]
            n = n - 1
            side = int(n**0.5)
        else:
            w = side
            h = max(n // max(w, 1), 1)
            tokens = tokens[:, : h * w, :]
            return tokens.transpose(1, 2).reshape(b, c, h, w)
    return tokens.transpose(1, 2).reshape(b, c, side, side)


def _pick_hidden_states(hidden_states: Tuple[torch.Tensor, ...]) -> List[torch.Tensor]:
    n = len(hidden_states)
    if n <= 4:
        return list(hidden_states)
    idx = [max(0, n // 8), max(1, n // 4), max(2, n // 2), n - 1]
    return [hidden_states[i] for i in idx]


class Qwen3VLBackbone(nn.Module):
    def __init__(
        self,
        name_or_path: str,
        mode: str = "qwen3vl",
        trainable_vision_blocks: int = 2,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_last_n_blocks: int = 4,
        lora_target_regex: Optional[str] = None,
        lora_include_mlp: bool = False,
        allow_fallback_mock: bool = True,
    ):
        super().__init__()
        self.mode = mode
        self.allow_fallback_mock = allow_fallback_mock
        self.qwen_ok = False
        self.mock = _MockBackbone()
        self.out_channels = [64, 128, 256, 512]
        self.vision_path: Optional[str] = None

        if mode != "qwen3vl":
            return

        try:
            from transformers import AutoModel
        except Exception:
            return

        try:
            model = AutoModel.from_pretrained(name_or_path, trust_remote_code=True)
            self.model = model
            self.vision_path, self.vision = self._find_vision_module(model)
            self._freeze_modules(trainable_vision_blocks=trainable_vision_blocks, use_lora=use_lora)
            if use_lora:
                self._inject_vision_lora(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    lora_last_n_blocks=lora_last_n_blocks,
                    lora_target_regex=lora_target_regex,
                    lora_include_mlp=lora_include_mlp,
                )
            self.qwen_ok = self.vision is not None
        except Exception:
            self.qwen_ok = False

    def _find_vision_module(self, model: nn.Module) -> Tuple[Optional[str], Optional[nn.Module]]:
        for name in ["visual", "vision_tower", "vision_model", "model.vision_tower", "model.visual"]:
            cur = model
            ok = True
            for part in name.split("."):
                if not hasattr(cur, part):
                    ok = False
                    break
                cur = getattr(cur, part)
            if ok:
                return name, cur
        return None, None

    def _set_module_by_path(self, root: nn.Module, path: str, module: nn.Module) -> None:
        parts = path.split(".")
        cur = root
        for p in parts[:-1]:
            cur = getattr(cur, p)
        setattr(cur, parts[-1], module)

    def _find_vision_blocks(self) -> Tuple[Optional[str], List[nn.Module]]:
        if self.vision is None:
            return None, []
        for maybe in ["blocks", "layers", "encoder.layers"]:
            cur = self.vision
            ok = True
            for part in maybe.split("."):
                if not hasattr(cur, part):
                    ok = False
                    break
                cur = getattr(cur, part)
            if ok and hasattr(cur, "__len__"):
                return maybe, list(cur)
        return None, []

    def _freeze_modules(self, trainable_vision_blocks: int, use_lora: bool) -> None:
        for p in self.model.parameters():
            p.requires_grad = False
        if self.vision is None:
            return

        if use_lora:
            return

        _, modules = self._find_vision_blocks()
        if modules:
            for m in modules[-trainable_vision_blocks:]:
                for p in m.parameters():
                    p.requires_grad = True

    def _inject_vision_lora(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_last_n_blocks: int,
        lora_target_regex: Optional[str],
        lora_include_mlp: bool,
    ) -> None:
        if self.vision is None:
            raise RuntimeError("vision module not found; cannot apply LoRA")
        try:
            from peft import LoraConfig, get_peft_model
        except Exception as e:
            raise RuntimeError("peft is required when use_lora=true") from e

        block_path, blocks = self._find_vision_blocks()
        if not blocks or block_path is None:
            raise RuntimeError("cannot find vision block list; cannot apply LoRA safely")

        n = len(blocks)
        start = max(0, n - int(lora_last_n_blocks))
        selected_idx = list(range(start, n))

        if lora_target_regex:
            target_regex = lora_target_regex
        else:
            ids = "|".join(str(i) for i in selected_idx)
            if lora_include_mlp:
                target_regex = (
                    rf"^{block_path}\.({ids})\."
                    rf"(attn\.(qkv|proj)|mlp\.(linear_fc1|linear_fc2))$"
                )
            else:
                target_regex = rf"^{block_path}\.({ids})\.attn\.(qkv|proj)$"

        compiled = re.compile(target_regex)
        matched = []
        for name, module in self.vision.named_modules():
            if isinstance(module, nn.Linear) and compiled.match(name):
                matched.append(name)
        if not matched:
            raise RuntimeError(f"no LoRA target linear modules matched regex: {target_regex}")

        lora_cfg = LoraConfig(
            r=int(r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            target_modules=target_regex,
            bias="none",
        )
        vision_with_lora = get_peft_model(self.vision, lora_cfg)
        self.vision = vision_with_lora
        if self.vision_path is not None:
            self._set_module_by_path(self.model, self.vision_path, self.vision)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(
            f"[LoRA] regex={target_regex} matched={len(matched)} selected_blocks={selected_idx} "
            f"trainable={trainable}/{total} ({100.0 * trainable / max(total,1):.4f}%)"
        )

    def _forward_qwen_vision(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        vision_out = self.vision(pixel_values=image, output_hidden_states=True, return_dict=True)
        hs = getattr(vision_out, "hidden_states", None)
        if hs is None:
            last = vision_out.last_hidden_state
            hs = (last, last, last, last)
        feats = _pick_hidden_states(tuple(hs))
        feats2d = [_reshape_tokens_to_2d(f) if f.dim() == 3 else f for f in feats]
        while len(feats2d) < 4:
            feats2d.insert(0, feats2d[0])
        f1, f2, f3, f4 = feats2d[-4:]
        return {"feat_l1": f1, "feat_l2": f2, "feat_l3": f3, "feat_l4": f4}

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        # image: Tensor[B,3,H,W]
        if self.qwen_ok:
            try:
                return self._forward_qwen_vision(image)
            except Exception:
                if not self.allow_fallback_mock:
                    raise
        return self.mock(image)
