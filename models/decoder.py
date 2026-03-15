from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatmapGuidedDecoder(nn.Module):
    def __init__(self, channels: int = 256, edge_head: bool = False):
        super().__init__()
        self.edge_head = edge_head
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 4 + 4, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mask_head = nn.Conv2d(channels // 2, 1, 1)
        self.edge_conv = nn.Conv2d(channels // 2, 1, 1) if edge_head else None

    def _guided_cat(self, feat: torch.Tensor, heatmap: torch.Tensor, tgt_size) -> torch.Tensor:
        f = F.interpolate(feat, size=tgt_size, mode="bilinear", align_corners=False)
        h = F.interpolate(heatmap, size=tgt_size, mode="bilinear", align_corners=False)
        return torch.cat([f, h], dim=1)

    def forward(self, pyr: Dict[str, torch.Tensor], heatmap: torch.Tensor, out_hw) -> Dict[str, torch.Tensor]:
        p2, p3, p4, p5 = pyr["p2"], pyr["p3"], pyr["p4"], pyr["p5"]
        size = p2.shape[-2:]
        x = torch.cat(
            [
                self._guided_cat(p2, heatmap, size),
                self._guided_cat(p3, heatmap, size),
                self._guided_cat(p4, heatmap, size),
                self._guided_cat(p5, heatmap, size),
            ],
            dim=1,
        )
        x = self.fuse(x)
        mask_low = torch.sigmoid(self.mask_head(x))  # [B,1,h,w]
        mask0 = F.interpolate(mask_low, size=out_hw, mode="bilinear", align_corners=False)  # [B,1,H,W]
        out = {"mask0": mask0}
        if self.edge_conv is not None:
            out["edge"] = torch.sigmoid(F.interpolate(self.edge_conv(x), size=out_hw, mode="bilinear", align_corners=False))
        return out
