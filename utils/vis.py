from pathlib import Path
from typing import Optional

import torch
from torchvision.utils import save_image


def _heatmap_to_rgb(heatmap: torch.Tensor) -> torch.Tensor:
    if heatmap.dim() == 3 and heatmap.shape[0] == 1:
        heatmap = heatmap[0]
    elif heatmap.dim() != 2:
        raise ValueError(f"expected heatmap shape [H,W] or [1,H,W], got {tuple(heatmap.shape)}")
    heatmap = heatmap.clamp(0.0, 1.0)
    low = heatmap <= 0.5
    high = ~low
    rgb = torch.empty((3, *heatmap.shape[-2:]), dtype=heatmap.dtype, device=heatmap.device)
    low_t = heatmap[low] / 0.5
    high_t = (heatmap[high] - 0.5) / 0.5

    rgb[0][low] = low_t
    rgb[1][low] = low_t
    rgb[2][low] = 1.0

    rgb[0][high] = 1.0
    rgb[1][high] = 1.0 - high_t
    rgb[2][high] = 1.0 - high_t
    return rgb


def save_triplet_vis(
    image: torch.Tensor,
    gt_mask: torch.Tensor,
    heatmap: torch.Tensor,
    pred_mask: torch.Tensor,
    path: str,
    max_items: int = 5,
) -> None:
    # image: [B,3,H,W], others: [B,1,H,W] (heatmap may be lower-res)
    b = min(max_items, image.shape[0])
    if heatmap.shape[-2:] != image.shape[-2:]:
        heatmap = torch.nn.functional.interpolate(heatmap, size=image.shape[-2:], mode="bilinear", align_corners=False)
    if pred_mask.shape[-2:] != image.shape[-2:]:
        pred_mask = torch.nn.functional.interpolate(pred_mask, size=image.shape[-2:], mode="bilinear", align_corners=False)
    pred_mask = (pred_mask > 0.5).float()
    tiles = []
    for i in range(b):
        h3 = _heatmap_to_rgb(heatmap[i])
        g3 = gt_mask[i].repeat(3, 1, 1)
        p3 = pred_mask[i].repeat(3, 1, 1)
        tiles.extend([image[i], g3, h3, p3])
    grid = torch.stack(tiles, dim=0)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, str(p), nrow=4)
