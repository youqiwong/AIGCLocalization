from pathlib import Path
from typing import Optional

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image
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


def _mono_to_rgb(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 3 and mask.shape[0] == 1:
        mask = mask[0]
    elif mask.dim() != 2:
        raise ValueError(f"expected mono tensor shape [H,W] or [1,H,W], got {tuple(mask.shape)}")
    return mask.unsqueeze(0).repeat(3, 1, 1)


def _sample_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    pred = (pred_mask >= 0.5).float().view(-1)
    gt = gt_mask.float().view(-1)
    inter = float((pred * gt).sum().item())
    union = float(((pred + gt) > 0).float().sum().item())
    if union <= 0:
        return 1.0
    return inter / union


def save_triplet_vis(
    image: torch.Tensor,
    gt_mask: torch.Tensor,
    heatmap: torch.Tensor,
    pred_mask: torch.Tensor,
    path: str,
    gt_edge: Optional[torch.Tensor] = None,
    pred_edge: Optional[torch.Tensor] = None,
    max_items: int = 5,
) -> None:
    # image: [B,3,H,W], others: [B,1,H,W] (heatmap may be lower-res)
    b = min(max_items, image.shape[0])
    if heatmap.shape[-2:] != image.shape[-2:]:
        heatmap = torch.nn.functional.interpolate(heatmap, size=image.shape[-2:], mode="bilinear", align_corners=False)
    if pred_mask.shape[-2:] != image.shape[-2:]:
        pred_mask = torch.nn.functional.interpolate(pred_mask, size=image.shape[-2:], mode="bilinear", align_corners=False)
    pred_mask = (pred_mask > 0.5).float()
    if gt_edge is not None and gt_edge.shape[-2:] != image.shape[-2:]:
        gt_edge = torch.nn.functional.interpolate(gt_edge, size=image.shape[-2:], mode="nearest")
    if pred_edge is not None and pred_edge.shape[-2:] != image.shape[-2:]:
        pred_edge = torch.nn.functional.interpolate(pred_edge, size=image.shape[-2:], mode="bilinear", align_corners=False)
    tiles = []
    for i in range(b):
        h3 = _heatmap_to_rgb(heatmap[i])
        g3 = _mono_to_rgb(gt_mask[i])
        p3 = _mono_to_rgb(pred_mask[i])
        tiles.extend([image[i], g3, h3, p3])
        if gt_edge is not None and pred_edge is not None:
            ge3 = _mono_to_rgb(gt_edge[i])
            pe3 = _mono_to_rgb(pred_edge[i].clamp(0.0, 1.0))
            tiles.extend([ge3, pe3])
    grid = torch.stack(tiles, dim=0)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    nrow = 6 if gt_edge is not None and pred_edge is not None else 4
    save_image(grid, str(p), nrow=nrow)


def save_eval_annotated_vis(
    image: torch.Tensor,
    gt_mask: torch.Tensor,
    heatmap: torch.Tensor,
    pred_mask: torch.Tensor,
    prob: torch.Tensor,
    label: torch.Tensor,
    path: str,
    max_items: int = 5,
) -> None:
    b = min(max_items, image.shape[0])
    if heatmap.shape[-2:] != image.shape[-2:]:
        heatmap = torch.nn.functional.interpolate(heatmap, size=image.shape[-2:], mode="bilinear", align_corners=False)
    if pred_mask.shape[-2:] != image.shape[-2:]:
        pred_mask = torch.nn.functional.interpolate(pred_mask, size=image.shape[-2:], mode="bilinear", align_corners=False)
    pred_mask = (pred_mask > 0.5).float()
    font = ImageFont.load_default()
    row_images = []
    for i in range(b):
        tiles = [
            to_pil_image(image[i].detach().float().cpu().clamp(0.0, 1.0)),
            to_pil_image(_mono_to_rgb(gt_mask[i].detach().float().cpu().clamp(0.0, 1.0))),
            to_pil_image(_heatmap_to_rgb(heatmap[i].detach().float().cpu())),
            to_pil_image(_mono_to_rgb(pred_mask[i].detach().float().cpu())),
        ]
        tile_w, tile_h = tiles[0].size
        header_h = 24
        row = Image.new("RGB", (tile_w * len(tiles), tile_h + header_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(row)
        pred_cls = int(float(prob[i].item()) >= 0.5)
        gt_cls = int(float(label[i].item()) >= 0.5)
        iou = _sample_iou(pred_mask[i], gt_mask[i])
        caption = f"iou={iou:.4f} pred_cls={pred_cls} gt_cls={gt_cls} prob={float(prob[i].item()):.4f}"
        draw.text((4, 4), caption, fill=(0, 0, 0), font=font)
        for col, tile in enumerate(tiles):
            row.paste(tile, (col * tile_w, header_h))
        row_images.append(row)

    canvas_w = row_images[0].size[0]
    canvas_h = sum(img.size[1] for img in row_images)
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    y = 0
    for row in row_images:
        canvas.paste(row, (0, y))
        y += row.size[1]

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(p)
