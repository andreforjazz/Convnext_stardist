"""
losses_v2.py  —  multitask_loss_v2

Changes vs losses.py (v1):
  Fix #2 — Instance-level classification loss.
            Pixel CE (v1) trains on every labeled pixel individually; pixels of
            the *same* instance can vote for different classes.  Instance-level CE
            pools cls logits inside each ground-truth instance mask → single
            prediction per instance → CE against its label.  This directly
            optimises the vote_class() inference aggregation.

  Fix #3 — Class-weighted CE.
            Both pixel CE and instance CE accept a class_weights tensor so rare
            tissue classes (collagen, ear, skull …) receive proportionally more
            gradient.  Pass class_weights=None to use uniform weighting.
            Recommended: inverse-sqrt frequency, computed in train_v2.py.

  New config keys used by train_v2.py:
      train.loss_w_cls:  2.0   (increased from 0.5)
      train.loss_w_inst: 0.5   (instance-level CE weight)
      train.class_weights: auto | null | [w0, w1, …, w18]
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ── Instance-level CE ─────────────────────────────────────────────────────────

def _instance_cls_loss(
    cls_logit: torch.Tensor,          # (B, C, H, W)
    inst_map: torch.Tensor,           # (B, H, W)  int64 instance IDs (0 = bg)
    cls_tgt: torch.Tensor,            # (B, H, W)  int64 pixel labels (−100 = ignore)
    class_weights: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    For every ground-truth instance in the batch:
      1. Average the cls logits over all pixels inside the instance mask.
      2. Compute CE against the instance's class label.

    Pixels belonging to an instance but marked ignore_index are skipped
    (they come from tiles without inst2class supervision).

    Uses vectorised scatter_add so the cost is O(B*H*W), not O(N_instances).
    """
    B, C, H, W = cls_logit.shape
    device = cls_logit.device
    N = B * H * W

    # Flatten everything
    logits_flat = cls_logit.permute(0, 2, 3, 1).reshape(N, C)   # (N, C)
    inst_flat   = inst_map.reshape(N)                             # (N,)
    lbl_flat    = cls_tgt.reshape(N)                              # (N,)

    # Keep only foreground pixels that have a valid (non-ignore) class label
    valid = (inst_flat > 0) & (lbl_flat != ignore_index)
    if not valid.any():
        return torch.zeros((), device=device, dtype=cls_logit.dtype)

    inst_v = inst_flat[valid]   # (M,)
    logits_v = logits_flat[valid]  # (M, C)
    lbl_v    = lbl_flat[valid].long()  # (M,)

    # Make instance IDs globally unique across the batch
    batch_idx = torch.arange(B, device=device).repeat_interleave(H * W)
    batch_v   = batch_idx[valid]
    max_inst  = inst_map.max().item() + 1
    global_id = batch_v * int(max_inst) + inst_v   # (M,) unique per (batch, inst)

    # Re-index to 0…K-1
    uniq_ids, remapped = torch.unique(global_id, return_inverse=True)   # K unique instances
    K = len(uniq_ids)

    # Scatter-sum logits per instance, then normalise by count
    logit_sum = torch.zeros(K, C, device=device, dtype=cls_logit.dtype)
    logit_sum.scatter_add_(0, remapped.unsqueeze(1).expand(-1, C), logits_v.to(cls_logit.dtype))

    count = torch.zeros(K, device=device, dtype=cls_logit.dtype)
    count.scatter_add_(0, remapped, torch.ones(int(valid.sum()), device=device, dtype=cls_logit.dtype))

    inst_logits = logit_sum / (count.unsqueeze(1) + 1e-6)   # (K, C)

    # One label per instance  (all valid pixels of the same instance share the same
    # label since build_class_target assigns exactly one class per instance ID)
    inst_labels = torch.zeros(K, dtype=torch.long, device=device)
    inst_labels.scatter_(0, remapped, lbl_v)   # last write — all identical per instance

    return F.cross_entropy(
        inst_logits, inst_labels,
        weight=class_weights,
        reduction="mean",
    )


# ── Main loss ─────────────────────────────────────────────────────────────────

def multitask_loss_v2(
    prob_logit:    torch.Tensor,       # (B,1,H,W)
    dist_pred:     torch.Tensor,       # (B,R,H,W)
    cls_logit:     torch.Tensor,       # (B,C,H,W)
    prob_tgt:      torch.Tensor,       # (B,1,H,W)
    dist_tgt:      torch.Tensor,       # (B,R,H,W)
    cls_tgt:       torch.Tensor,       # (B,H,W) int64
    fg_mask:       torch.Tensor,       # (B,1,H,W)
    inst_map:      torch.Tensor,       # (B,H,W) int64   ← NEW: raw instance IDs
    *,
    w_prob:        float = 1.0,
    w_dist:        float = 0.05,
    w_cls:         float = 2.0,        # increased from 0.5
    w_inst:        float = 0.5,        # instance-level CE weight
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    prob_* : (B,1,H,W)
    dist_* : (B,R,H,W)
    cls_*  : (B,C,H,W) logits / (B,H,W) int64 targets
    fg_mask: (B,1,H,W) bool or float {0,1}
    inst_map: (B,H,W) int64 instance segmentation ground truth (0 = background)
    """

    # ── Probability BCE (unchanged from v1) ──────────────────────────────────
    bce = F.binary_cross_entropy_with_logits(
        prob_logit, prob_tgt.float(), reduction="mean"
    )

    # ── Distance masked L1 (unchanged from v1) ───────────────────────────────
    fg = fg_mask.squeeze(1).float() > 0.5
    if fg.any():
        d_err     = (dist_pred - dist_tgt).abs().mean(dim=1)
        dist_loss = (d_err * fg.float()).sum() / (fg.float().sum() + 1e-6)
    else:
        dist_loss = dist_pred.sum() * 0.0

    # ── Pixel CE with class weights (fix #3) ─────────────────────────────────
    has_cls = (cls_tgt >= 0).any()
    if has_cls:
        pixel_cls_loss = F.cross_entropy(
            cls_logit, cls_tgt,
            weight=class_weights,
            ignore_index=-100,
            reduction="mean",
        )
    else:
        pixel_cls_loss = torch.zeros((), device=cls_logit.device, dtype=cls_logit.dtype)

    # ── Instance-level CE (fix #2) ───────────────────────────────────────────
    if has_cls and w_inst > 1e-8:
        inst_loss = _instance_cls_loss(cls_logit, inst_map, cls_tgt, class_weights)
    else:
        inst_loss = torch.zeros((), device=cls_logit.device, dtype=cls_logit.dtype)

    # Combined cls term: pixel CE + instance CE (both use class_weights)
    cls_combined = pixel_cls_loss + w_inst * inst_loss

    total = w_prob * bce + w_dist * dist_loss + w_cls * cls_combined
    parts = {
        "loss":      float(total.detach()),
        "bce":       float(bce.detach()),
        "dist":      float(dist_loss.detach()),
        "cls_pixel": float(pixel_cls_loss.detach()),
        "cls_inst":  float(inst_loss.detach()),
    }
    return total, parts
