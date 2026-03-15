from typing import Dict

import torch
import torch.nn as nn

from models.decoder import HeatmapGuidedDecoder
from models.feature_adapter import build_adapter
from models.proposer import CoarseSuspicionProposer
from models.qwen3vl_backbone import Qwen3VLBackbone
from utils.regions import build_candidate_regions


class Stage1ForgeryModel(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        bcfg = cfg["backbone"]
        self.backbone = Qwen3VLBackbone(
            name_or_path=bcfg["name_or_path"],
            mode=bcfg.get("mode", "qwen3vl"),
            trainable_vision_blocks=bcfg.get("trainable_vision_blocks", 2),
            use_lora=bcfg.get("use_lora", False),
            lora_r=bcfg.get("lora_r", 16),
            lora_alpha=bcfg.get("lora_alpha", 32),
            lora_dropout=bcfg.get("lora_dropout", 0.05),
            lora_last_n_blocks=bcfg.get("lora_last_n_blocks", 4),
            lora_target_regex=bcfg.get("lora_target_regex", None),
            lora_include_mlp=bcfg.get("lora_include_mlp", False),
            allow_fallback_mock=bcfg.get("allow_fallback_mock", True),
        )
        in_channels = self.backbone.out_channels
        self.adapter = build_adapter(
            adapter_type=cfg["adapter"].get("type", "plain_fpn"),
            in_channels=in_channels,
            out_channels=cfg["adapter"].get("out_channels", 256),
        )
        c = cfg["adapter"].get("out_channels", 256)
        self.proposer = CoarseSuspicionProposer(channels=c)
        self.decoder = HeatmapGuidedDecoder(channels=c, edge_head=cfg["decoder"].get("edge_head", False))
        self.topk_regions = cfg["proposer"].get("topk_regions", 8)
        self.region_window = cfg["proposer"].get("region_window", 48)

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        # image: [B,3,H,W]
        feats = self.backbone(image)
        pyr = self.adapter(feats)
        prop = self.proposer(pyr)
        dec = self.decoder(pyr, prop["heatmap"], out_hw=image.shape[-2:])
        regions = build_candidate_regions(prop["heatmap"], topk=self.topk_regions, window=self.region_window)
        return {
            "p_edit": prop["p_edit"],  # [B]
            "heatmap": prop["heatmap"],  # [B,1,h,w]
            "mask0": dec["mask0"],  # [B,1,H,W]
            "candidate_regions": regions,  # List[[K,4]]
            "features": pyr,
        }
