"""Legacy v5.1 / v5.2 / v5.3 inference variants — extracted from
infer_wsi_v54_gpu_peaks.ipynb cell 8 for reference only.

These functions are NOT used by the v5.4 main inference loop. They are kept
here so older diagnostic / comparison cells can still be revived if needed.

The active v5.4 production helpers live in `inference_v54.py` at the repo root.
Function bodies are copied verbatim from cell 8.

Includes:
  - process_tile_batch_v51 — full-cls-logits CPU vote-class loop
  - batch_forward_v52, process_tile_batch_v52 — GPU vote-class (avg_pool + argmax)
  - batch_forward_v53, process_tile_batch_v53 — adds GPU ring decoding via grid_sample

Note: process_tile_batch_v51 here is the legacy CPU vote-class loop. The
diagnostic-only `batch_forward_v51` (which returns full cls logits and is still
used by cell 14 of the notebook) lives in `inference_v54.py`.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as _F

# Reuse the active module's ImageNet stats so behaviour is identical.
from inference_v54 import (
    _IMNET_MEAN,
    _IMNET_STD,
    _grid_sample_dist_at_peaks,
    batch_forward_v51,
)


# ── 4.3 process_tile_batch_v51 (routes through Patch A forward) ───────────────
def process_tile_batch_v51(
    meta,
    patches,
    model,
    device,
    *,
    fp16: bool = False,
    bf16: bool = False,
    channels_last: bool = False,
    cls_perm_gpu=None,
    nms_dist: int = 8,
    prob_thresh: float = 0.45,
    refine_local_com: bool = True,
    refine_radius_px: int = 8,
    valid_margin: int = 0,
    w_slide: int,
    h_slide: int,
    idx2label,
    color_for_name=None,
    use_fast_vote_class: bool = True,
    vote_window_px: int = 9,
    coord_int: bool = True,
    include_class_probs: bool = False,
):
    """v5.1: integer-pixel rings + colorRGB inline; routed through batch_forward_v51."""
    from geometry import (
        local_peaks,
        dists_and_coords_from_peaks,
        polygon_ring_rowcol,
        vote_class as poly_vote_class,
    )

    _t = time.perf_counter()
    results = batch_forward_v51(
        model, patches, device,
        fp16=fp16, bf16=bf16, channels_last=channels_last,
        cls_perm_gpu=cls_perm_gpu,
    )
    gpu_time = time.perf_counter() - _t

    vm = int(valid_margin)
    hw = max(1, vote_window_px // 2)
    features = []

    for (x0, y0, tw, th), (prob, dist_m, cls_l) in zip(meta, results):
        pks = local_peaks(prob, min_distance=int(nms_dist), thresh=float(prob_thresh))
        if not len(pks):
            continue
        _, coords = dists_and_coords_from_peaks(
            dist_m, pks, prob,
            refine_local_com=refine_local_com,
            refine_radius_px=int(refine_radius_px),
        )
        _left, _top    = (x0 == 0), (y0 == 0)
        _right, _bottom = (x0 + tw >= w_slide), (y0 + th >= h_slide)
        H_t, W_t = prob.shape
        for k in range(coords.shape[0]):
            pr, pc = int(pks[k, 0]), int(pks[k, 1])
            if vm:
                if not _left   and pc < vm:        continue
                if not _right  and pc >= tw - vm:  continue
                if not _top    and pr < vm:         continue
                if not _bottom and pr >= th - vm:   continue
            if use_fast_vote_class:
                r0 = max(0, pr - hw); r1 = min(H_t, pr + hw)
                c0 = max(0, pc - hw); c1 = min(W_t, pc + hw)
                lw = cls_l[:, r0:r1, c0:c1].mean(axis=(1, 2))
                lw = lw - lw.max()
                e = np.exp(lw)
                probs_k = (e / (e.sum() + 1e-10)).astype(np.float32)
                cls_id = int(probs_k.argmax())
            else:
                cls_id, probs_k = poly_vote_class(cls_l, coords[k], (th, tw))
            ring = polygon_ring_rowcol(coords[k]) + np.array([x0, y0], dtype=np.float32)
            if coord_int:
                ring = np.rint(ring).astype(np.int32)
            name = idx2label.get(cls_id, f'class_{cls_id}')
            r_, g_, b_ = color_for_name(name) if color_for_name else (128, 128, 128)
            color_rgb = (int(r_) << 16) | (int(g_) << 8) | int(b_)
            props = {
                'classification': {
                    'name': name,
                    'index': int(cls_id),
                    'colorRGB': int(color_rgb),
                },
                'prob_peak': float(prob[pr, pc]),
            }
            if include_class_probs:
                props['class_probs'] = {
                    idx2label.get(i, str(i)): float(p)
                    for i, p in enumerate(probs_k)
                }
            features.append({
                'type': 'Feature',
                'geometry': {'type': 'Polygon', 'coordinates': [ring]},
                'properties': props,
            })
    return features, gpu_time


# -- 4.6 GPU vote-class forward (v5.2) ----------------------------------------
@torch.inference_mode()
def batch_forward_v52(
    model,
    patches_hwc,
    device,
    *,
    fp16: bool = False,
    bf16: bool = False,
    channels_last: bool = False,
    cls_perm_gpu=None,
    vote_window_px: int = 9,
    include_class_probs: bool = False,
):
    """v5.2: avg_pool2d + argmax on GPU before D2H.

    Default fast path ships a per-pixel int16 class map instead of fp32 (N, C, H, W).
    Falls back to full softmax probs if include_class_probs=True.

    Returns: list of (prob, dist_map, cls_out) per tile, where cls_out is either:
      - (H, W) int16 argmax map           (include_class_probs=False, default)
      - (C, H, W) fp32 softmax probs      (include_class_probs=True)
    """
    if bf16:
        infer_dtype = torch.bfloat16
    elif fp16 and device.type == "cuda":
        infer_dtype = torch.float16
    else:
        infer_dtype = torch.float32

    sizes = [(p.shape[0], p.shape[1]) for p in patches_hwc]
    max_h = int(math.ceil(max(s[0] for s in sizes) / 32) * 32)
    max_w = int(math.ceil(max(s[1] for s in sizes) / 32) * 32)

    arrs = []
    for p in patches_hwc:
        ph, pw = p.shape[:2]
        if ph != max_h or pw != max_w:
            pad = np.zeros((max_h, max_w, 3), dtype=np.uint8)
            pad[:ph, :pw] = p
            p = pad
        arrs.append(p)

    batch_u8 = torch.from_numpy(np.stack(arrs))
    batch = batch_u8.to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous()
    batch = batch.to(infer_dtype).div_(255.0)
    mean = torch.tensor(_IMNET_MEAN, device=device, dtype=infer_dtype).view(1, 3, 1, 1)
    std = torch.tensor(_IMNET_STD, device=device, dtype=infer_dtype).view(1, 3, 1, 1)
    batch = (batch - mean) / std
    if channels_last:
        batch = batch.contiguous(memory_format=torch.channels_last)

    prob_logit, dist_b, cls_b = model(batch)
    if cls_perm_gpu is not None:
        cls_b = cls_b.index_select(1, cls_perm_gpu)

    # GPU vote-class: window-mean of class logits then argmax over class dim.
    # avg_pool2d works in fp16/bf16 directly; argmax is invariant to monotonic
    # scale so we do not need a softmax for the fast path.
    pad = vote_window_px // 2
    cls_avg = _F.avg_pool2d(cls_b, kernel_size=vote_window_px, stride=1, padding=pad)

    if include_class_probs:
        cls_probs_gpu = torch.softmax(cls_avg.float(), dim=1)        # (N, C, H, W) fp32
        cls_out_all = cls_probs_gpu.cpu().numpy()
    else:
        cls_argmax_gpu = cls_avg.argmax(dim=1).to(torch.int16)       # (N, H, W) int16
        cls_out_all = cls_argmax_gpu.cpu().numpy()

    prob_all = torch.sigmoid(prob_logit).cpu().float().numpy()       # (N, 1, H, W)
    dist_all = dist_b.contiguous().cpu().float().numpy()             # (N, 32, H, W)

    out = []
    for i, (ph, pw) in enumerate(sizes):
        cls_i = (
            cls_out_all[i, :, :ph, :pw] if include_class_probs
            else cls_out_all[i, :ph, :pw]
        )
        out.append((prob_all[i, 0, :ph, :pw], dist_all[i, :, :ph, :pw], cls_i))
    return out


# -- 4.7 process_tile_batch_v52 — uses GPU-side vote-class --------------------
def process_tile_batch_v52(
    meta,
    patches,
    model,
    device,
    *,
    fp16: bool = False,
    bf16: bool = False,
    channels_last: bool = False,
    cls_perm_gpu=None,
    nms_dist: int = 8,
    prob_thresh: float = 0.45,
    refine_local_com: bool = True,
    refine_radius_px: int = 8,
    valid_margin: int = 0,
    w_slide: int,
    h_slide: int,
    idx2label,
    color_for_name=None,
    vote_window_px: int = 9,
    coord_int: bool = True,
    include_class_probs: bool = False,
):
    """v5.2: vote-class is precomputed on the GPU. The CPU loop here only does
    peak detection, ring decoding, and feature dict construction."""
    from geometry import (
        local_peaks,
        dists_and_coords_from_peaks,
        polygon_ring_rowcol,
    )

    _t = time.perf_counter()
    results = batch_forward_v52(
        model, patches, device,
        fp16=fp16, bf16=bf16, channels_last=channels_last,
        cls_perm_gpu=cls_perm_gpu,
        vote_window_px=vote_window_px,
        include_class_probs=include_class_probs,
    )
    gpu_time = time.perf_counter() - _t

    vm = int(valid_margin)
    features = []

    for (x0, y0, tw, th), (prob, dist_m, cls_out) in zip(meta, results):
        pks = local_peaks(prob, min_distance=int(nms_dist), thresh=float(prob_thresh))
        if not len(pks):
            continue
        _, coords = dists_and_coords_from_peaks(
            dist_m, pks, prob,
            refine_local_com=refine_local_com,
            refine_radius_px=int(refine_radius_px),
        )
        _left, _top = (x0 == 0), (y0 == 0)
        _right, _bottom = (x0 + tw >= w_slide), (y0 + th >= h_slide)
        for k in range(coords.shape[0]):
            pr, pc = int(pks[k, 0]), int(pks[k, 1])
            if vm:
                if not _left and pc < vm:           continue
                if not _right and pc >= tw - vm:    continue
                if not _top and pr < vm:            continue
                if not _bottom and pr >= th - vm:   continue
            if include_class_probs:
                # cls_out shape: (C, H, W) fp32 softmax
                probs_k = cls_out[:, pr, pc].astype(np.float32)
                cls_id = int(probs_k.argmax())
            else:
                # cls_out shape: (H, W) int16 argmax precomputed on GPU
                cls_id = int(cls_out[pr, pc])
            ring = polygon_ring_rowcol(coords[k]) + np.array([x0, y0], dtype=np.float32)
            if coord_int:
                ring = np.rint(ring).astype(np.int32)
            name = idx2label.get(cls_id, f"class_{cls_id}")
            r_, g_, b_ = color_for_name(name) if color_for_name else (128, 128, 128)
            color_rgb = (int(r_) << 16) | (int(g_) << 8) | int(b_)
            props = {
                "classification": {
                    "name": name,
                    "index": int(cls_id),
                    "colorRGB": int(color_rgb),
                },
                "prob_peak": float(prob[pr, pc]),
            }
            if include_class_probs:
                props["class_probs"] = {
                    idx2label.get(i, str(i)): float(p)
                    for i, p in enumerate(probs_k)
                }
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": props,
            })
    return features, gpu_time


# -- 4.8 GPU ring decoding (v5.3) --------------------------------------------

@torch.inference_mode()
def batch_forward_v53(
    model,
    patches_hwc,
    device,
    *,
    fp16: bool = False,
    bf16: bool = False,
    channels_last: bool = False,
    cls_perm_gpu=None,
    vote_window_px: int = 9,
    include_class_probs: bool = False,
):
    """v5.3: like v5.2 but keeps the dist tensor on the GPU for grid_sample.

    Returns: (per_tile_results, dist_b_gpu, sizes) where:
      per_tile_results: list of (prob_np, cls_out_np) per tile
      dist_b_gpu: (N, R, max_h, max_w) tensor still on the device
      sizes: list of (ph, pw) per tile (un-padded sizes)
    """
    if bf16:
        infer_dtype = torch.bfloat16
    elif fp16 and device.type == "cuda":
        infer_dtype = torch.float16
    else:
        infer_dtype = torch.float32

    sizes = [(p.shape[0], p.shape[1]) for p in patches_hwc]
    max_h = int(math.ceil(max(s[0] for s in sizes) / 32) * 32)
    max_w = int(math.ceil(max(s[1] for s in sizes) / 32) * 32)

    arrs = []
    for p in patches_hwc:
        ph, pw = p.shape[:2]
        if ph != max_h or pw != max_w:
            pad = np.zeros((max_h, max_w, 3), dtype=np.uint8)
            pad[:ph, :pw] = p
            p = pad
        arrs.append(p)

    batch_u8 = torch.from_numpy(np.stack(arrs))
    batch = batch_u8.to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous()
    batch = batch.to(infer_dtype).div_(255.0)
    mean = torch.tensor(_IMNET_MEAN, device=device, dtype=infer_dtype).view(1, 3, 1, 1)
    std = torch.tensor(_IMNET_STD, device=device, dtype=infer_dtype).view(1, 3, 1, 1)
    batch = (batch - mean) / std
    if channels_last:
        batch = batch.contiguous(memory_format=torch.channels_last)

    prob_logit, dist_b, cls_b = model(batch)
    if cls_perm_gpu is not None:
        cls_b = cls_b.index_select(1, cls_perm_gpu)

    # GPU vote-class (same as v5.2)
    pad = vote_window_px // 2
    cls_avg = _F.avg_pool2d(cls_b, kernel_size=vote_window_px, stride=1, padding=pad)
    if include_class_probs:
        cls_probs_gpu = torch.softmax(cls_avg.float(), dim=1)
        cls_out_all = cls_probs_gpu.cpu().numpy()
    else:
        cls_argmax_gpu = cls_avg.argmax(dim=1).to(torch.int16)
        cls_out_all = cls_argmax_gpu.cpu().numpy()

    prob_all = torch.sigmoid(prob_logit).cpu().float().numpy()  # (N, 1, H, W)
    # NOTE: dist_b stays on the GPU. We pass the contiguous fp32 view to grid_sample
    # later. Keeping it in the model dtype is fine for bilinear sampling.
    dist_b_gpu = dist_b.contiguous()

    per_tile = []
    for i, (ph, pw) in enumerate(sizes):
        cls_i = (
            cls_out_all[i, :, :ph, :pw] if include_class_probs
            else cls_out_all[i, :ph, :pw]
        )
        per_tile.append((prob_all[i, 0, :ph, :pw], cls_i))
    return per_tile, dist_b_gpu, sizes


def process_tile_batch_v53(
    meta,
    patches,
    model,
    device,
    *,
    fp16: bool = False,
    bf16: bool = False,
    channels_last: bool = False,
    cls_perm_gpu=None,
    nms_dist: int = 8,
    prob_thresh: float = 0.45,
    refine_local_com: bool = True,
    refine_radius_px: int = 8,
    valid_margin: int = 0,
    w_slide: int,
    h_slide: int,
    idx2label,
    color_for_name=None,
    vote_window_px: int = 9,
    coord_int: bool = True,
    include_class_probs: bool = False,
):
    """v5.3: GPU vote-class + GPU ring decoding via grid_sample.
    The CPU loop only does peak detection, peak refinement, ring assembly,
    and feature dict construction."""
    from geometry import (
        local_peaks,
        refine_peaks_local_com,
        dist_to_coord,
        polygon_ring_rowcol,
    )

    _t = time.perf_counter()
    per_tile, dist_b_gpu, sizes = batch_forward_v53(
        model, patches, device,
        fp16=fp16, bf16=bf16, channels_last=channels_last,
        cls_perm_gpu=cls_perm_gpu,
        vote_window_px=vote_window_px,
        include_class_probs=include_class_probs,
    )
    gpu_time = time.perf_counter() - _t

    vm = int(valid_margin)
    features = []

    try:
        for i, ((x0, y0, tw, th), (prob, cls_out)) in enumerate(zip(meta, per_tile)):
            ph, pw = sizes[i]
            pks = local_peaks(prob, min_distance=int(nms_dist), thresh=float(prob_thresh))
            if not len(pks):
                continue
            # Refine on CPU (cheap relative to ring decode).
            if refine_local_com:
                pts = refine_peaks_local_com(pks, prob, int(refine_radius_px))
            else:
                pts = pks.astype(np.float64)

            # Ring decoding on the GPU (was the slow CPU step).
            dists = _grid_sample_dist_at_peaks(
                dist_b_gpu, i, ph, pw, pts, device,
            )  # (K, R) fp32
            coords = dist_to_coord(dists, pts.astype(np.float32))  # (K, 2, R)

            _left, _top = (x0 == 0), (y0 == 0)
            _right, _bottom = (x0 + tw >= w_slide), (y0 + th >= h_slide)
            for k in range(coords.shape[0]):
                pr, pc = int(pks[k, 0]), int(pks[k, 1])
                if vm:
                    if not _left and pc < vm:           continue
                    if not _right and pc >= tw - vm:    continue
                    if not _top and pr < vm:            continue
                    if not _bottom and pr >= th - vm:   continue
                if include_class_probs:
                    probs_k = cls_out[:, pr, pc].astype(np.float32)
                    cls_id = int(probs_k.argmax())
                else:
                    cls_id = int(cls_out[pr, pc])
                ring = polygon_ring_rowcol(coords[k]) + np.array([x0, y0], dtype=np.float32)
                if coord_int:
                    ring = np.rint(ring).astype(np.int32)
                name = idx2label.get(cls_id, f"class_{cls_id}")
                r_, g_, b_ = color_for_name(name) if color_for_name else (128, 128, 128)
                color_rgb = (int(r_) << 16) | (int(g_) << 8) | int(b_)
                props = {
                    "classification": {
                        "name": name,
                        "index": int(cls_id),
                        "colorRGB": int(color_rgb),
                    },
                    "prob_peak": float(prob[pr, pc]),
                }
                if include_class_probs:
                    props["class_probs"] = {
                        idx2label.get(j, str(j)): float(p)
                        for j, p in enumerate(probs_k)
                    }
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": props,
                })
    finally:
        # Free the GPU dist tensor explicitly so the next batch has headroom.
        del dist_b_gpu

    return features, gpu_time
