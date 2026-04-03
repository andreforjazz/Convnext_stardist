from __future__ import annotations

import torch
import torch.nn.functional as F


def multitask_loss(
    prob_logit: torch.Tensor,
    dist_pred: torch.Tensor,
    cls_logit: torch.Tensor,
    prob_tgt: torch.Tensor,
    dist_tgt: torch.Tensor,
    cls_tgt: torch.Tensor,
    fg_mask: torch.Tensor,
    *,
    w_prob: float = 1.0,
    w_dist: float = 0.05,
    w_cls: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    prob_* : (B,1,H,W)
    dist_* : (B,R,H,W)
    cls_*  : (B,C,H,W) logits / (B,H,W) int64 targets
    fg_mask: (B,1,H,W) bool or float {0,1}
    """
    bce = F.binary_cross_entropy_with_logits(prob_logit, prob_tgt.float(), reduction="mean")

    fg = fg_mask.squeeze(1).float() > 0.5
    if fg.any():
        d_err = (dist_pred - dist_tgt).abs().mean(dim=1)
        dist_loss = (d_err * fg.float()).sum() / (fg.float().sum() + 1e-6)
    else:
        dist_loss = dist_pred.sum() * 0.0

    if (cls_tgt >= 0).any():
        cls_loss = F.cross_entropy(cls_logit, cls_tgt, ignore_index=-100, reduction="mean")
    else:
        cls_loss = cls_logit.sum() * 0.0

    total = w_prob * bce + w_dist * dist_loss + w_cls * cls_loss
    parts = {
        "loss": float(total.detach()),
        "bce": float(bce.detach()),
        "dist": float(dist_loss.detach()),
        "cls": float(cls_loss.detach()),
    }
    return total, parts
