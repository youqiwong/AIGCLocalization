import torch
import torch.nn.functional as F


def edge_bce_loss(edge_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(edge_logits.float(), target.float())
