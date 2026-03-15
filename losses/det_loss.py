import torch
import torch.nn.functional as F


def detection_bce_loss(p_edit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(p_edit, target.float())
