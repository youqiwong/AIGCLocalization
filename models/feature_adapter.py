from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv1x1(in_ch: int, out_ch: int) -> nn.Module:
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))


class PlainFPNAdapter(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        self.lateral = nn.ModuleList([_conv1x1(c, out_channels) for c in in_channels])
        self.smooth = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True)) for _ in in_channels]
        )

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        l1, l2, l3, l4 = feats["feat_l1"], feats["feat_l2"], feats["feat_l3"], feats["feat_l4"]
        xs = [l1, l2, l3, l4]
        p = [self.lateral[i](xs[i]) for i in range(4)]
        for i in [2, 1, 0]:
            p[i] = p[i] + F.interpolate(p[i + 1], size=p[i].shape[-2:], mode="nearest")
        p = [self.smooth[i](p[i]) for i in range(4)]

        # output pyramid: p2/p3/p4/p5 expected [B,C,H/4...]
        p2 = p[0]
        p3 = F.avg_pool2d(p2, 2)
        p4 = F.avg_pool2d(p3, 2)
        p5 = F.avg_pool2d(p4, 2)
        return {"p2": p2, "p3": p3, "p4": p4, "p5": p5}


class SuspicionGatedAdapter(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        self.base = PlainFPNAdapter(in_channels=in_channels, out_channels=out_channels)
        self.gate_prior = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, 1, 1),
            nn.Sigmoid(),
        )
        self.low_enhance = nn.Sequential(
            nn.Conv2d(out_channels + 1, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pyr = self.base(feats)
        prior = self.gate_prior(pyr["p4"])  # [B,1,h/16,w/16]
        prior_p2 = F.interpolate(prior, size=pyr["p2"].shape[-2:], mode="bilinear", align_corners=False)
        low = torch.cat([pyr["p2"], prior_p2], dim=1)
        pyr["p2"] = self.low_enhance(low)
        return pyr


def build_adapter(adapter_type: str, in_channels: List[int], out_channels: int = 256) -> nn.Module:
    if adapter_type == "suspicion_gated":
        return SuspicionGatedAdapter(in_channels=in_channels, out_channels=out_channels)
    return PlainFPNAdapter(in_channels=in_channels, out_channels=out_channels)
