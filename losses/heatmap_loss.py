import torch
import torch.nn.functional as F


def focal_heatmap_loss(
    pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0
) -> torch.Tensor:
    pred = pred.float().clamp(1e-6, 1 - 1e-6)
    target = target.float()
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    p_t = pred * target + (1 - pred) * (1 - target)
    loss = bce * ((1 - p_t) ** gamma)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    return (alpha_t * loss).mean()
