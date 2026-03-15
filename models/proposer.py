from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoarseSuspicionProposer(nn.Module):
    def __init__(self, channels: int = 256):
        super().__init__()
        self.cls_head = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, 1),
        )
        self.heat_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, 1),
        )

    def forward(self, pyr: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        p2, p3, p4, p5 = pyr["p2"], pyr["p3"], pyr["p4"], pyr["p5"]
        # image-level head from highest-level feature
        pooled = F.adaptive_avg_pool2d(p5, 1).flatten(1)  # [B,C]
        p_edit = torch.sigmoid(self.cls_head(pooled)).squeeze(1)  # [B]

        # heatmap head on fused multi-scale feature
        p3u = F.interpolate(p3, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        p4u = F.interpolate(p4, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        p5u = F.interpolate(p5, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        fuse = (p2 + p3u + p4u + p5u) / 4.0
        heatmap = torch.sigmoid(self.heat_head(fuse))  # [B,1,h,w]
        return {"p_edit": p_edit, "heatmap": heatmap}
