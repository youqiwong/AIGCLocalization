from typing import Dict

import numpy as np
import torch


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def cls_metrics(prob: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> Dict[str, float]:
    p = (prob >= thr).float()
    y = target.float()
    tp = float(((p == 1) & (y == 1)).sum().item())
    tn = float(((p == 0) & (y == 0)).sum().item())
    fp = float(((p == 1) & (y == 0)).sum().item())
    fn = float(((p == 0) & (y == 1)).sum().item())
    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {"acc": acc, "f1": f1, "precision": precision, "recall": recall}


def pixel_metrics(pred: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> Dict[str, float]:
    p = (pred >= thr).float().view(pred.shape[0], -1)
    y = target.float().view(target.shape[0], -1)
    inter = (p * y).sum(dim=1)
    union = ((p + y) > 0).float().sum(dim=1)
    iou = ((inter + 1e-6) / (union + 1e-6)).mean().item()
    tp = inter
    fp = (p * (1 - y)).sum(dim=1)
    fn = ((1 - p) * y).sum(dim=1)
    f1 = ((2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)).mean().item()
    return {"iou": float(iou), "f1": float(f1)}


def binary_auc_ap(prob: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    y = target.detach().cpu().numpy().astype(np.float32)
    s = prob.detach().cpu().numpy().astype(np.float32)
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        auc = float(roc_auc_score(y, s))
        ap = float(average_precision_score(y, s))
    except Exception:
        # fallback without sklearn
        order = np.argsort(-s)
        y_sorted = y[order]
        tp = np.cumsum(y_sorted == 1)
        fp = np.cumsum(y_sorted == 0)
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / max(np.sum(y_sorted == 1), 1)
        ap = float(np.trapz(prec, rec))
        pos = s[y == 1]
        neg = s[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            auc = 0.0
        else:
            auc = float(np.mean((pos[:, None] > neg[None, :]).astype(np.float32)))
    return {"auc": auc, "ap": ap}
