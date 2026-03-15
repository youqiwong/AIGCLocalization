from pathlib import Path
from typing import Optional

import torch
from torchvision.utils import save_image


def save_triplet_vis(
    image: torch.Tensor,
    gt_mask: torch.Tensor,
    heatmap: torch.Tensor,
    pred_mask: torch.Tensor,
    path: str,
    max_items: int = 4,
) -> None:
    # image: [B,3,H,W], others: [B,1,H,W] (heatmap may be lower-res)
    b = min(max_items, image.shape[0])
    if heatmap.shape[-2:] != image.shape[-2:]:
        heatmap = torch.nn.functional.interpolate(heatmap, size=image.shape[-2:], mode="bilinear", align_corners=False)
    tiles = []
    for i in range(b):
        h3 = heatmap[i].repeat(3, 1, 1)
        g3 = gt_mask[i].repeat(3, 1, 1)
        p3 = pred_mask[i].repeat(3, 1, 1)
        tiles.extend([image[i], g3, h3, p3])
    grid = torch.stack(tiles, dim=0)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, str(p), nrow=4)
