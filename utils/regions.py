from typing import List

import torch
import torch.nn.functional as F


def build_candidate_regions(heatmap: torch.Tensor, topk: int = 8, window: int = 48) -> List[torch.Tensor]:
    # heatmap: [B,1,h,w], returns list of [K,4] in xyxy on heatmap scale
    b, _, h, w = heatmap.shape
    pooled = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    peaks = (heatmap == pooled) * heatmap
    flat = peaks.view(b, -1)
    vals, idx = torch.topk(flat, k=min(topk, flat.shape[1]), dim=1)
    regions: List[torch.Tensor] = []
    half = window // 2
    for bi in range(b):
        xyxy = []
        for i in range(idx.shape[1]):
            if vals[bi, i] <= 0:
                continue
            y = int(idx[bi, i].item() // w)
            x = int(idx[bi, i].item() % w)
            x1, y1 = max(0, x - half), max(0, y - half)
            x2, y2 = min(w - 1, x + half), min(h - 1, y + half)
            xyxy.append([x1, y1, x2, y2])
        if xyxy:
            regions.append(torch.tensor(xyxy, dtype=torch.float32, device=heatmap.device))
        else:
            regions.append(torch.zeros((0, 4), dtype=torch.float32, device=heatmap.device))
    return regions
