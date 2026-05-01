"""v5.4 WSI inference helpers — extracted from infer_wsi_v54_gpu_peaks.ipynb cell 8.

Active production helpers used by the v5.4 main inference loop:
  - Tissue masking (Patch B): compute_tissue_mask_thumbnail, filter_tile_coords_by_mask,
    tissue_bbox_level0, load_or_build_tissue_mask
  - GPU forward + post-processing (v5.4): batch_forward_v54, process_tile_batch_v54,
    _grid_sample_dist_at_peaks
  - Streaming GeoJSON writer: write_geojson_streaming
  - Auto LOAD_TO_RAM picker (Patch E): auto_load_to_ram

Also keeps batch_forward_v51 because the single-tile diagnostic cell needs full
fp32 cls logits for visualization.

Function bodies are copied verbatim from cell 8; only the imports were relocated
to module level.
"""
from __future__ import annotations

import math
import time
import gzip
import json
import concurrent.futures as _cf
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as _F

# Optional / feature-flagged imports — same try/except pattern as cell 8.
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
try:
    import orjson
    _HAS_ORJSON = True
except ImportError:
    _HAS_ORJSON = False
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ── 4.1 Tissue mask (Patch B: Otsu + bbox clip + safety net) ─────────────────
def compute_tissue_mask_thumbnail(
    slide,
    *,
    target_thumbnail_w: int = 2048,
    use_otsu: bool = True,
    val_thresh: int = 240,
    sat_thresh: int = 15,
    min_component_frac: float = 5e-4,
    dilate_tiles: int = 1,
    tile_size_l0: int = 256,
):
    """Thumbnail tissue mask. Returns (mask_bool[H_t, W_t], scale=level0_px/thumb_px).
    Patch B: Otsu on V by default (auto-calibrates per slide). Saturation guard is
    additive only — never subtracts tissue. Morphological close + small-component
    removal + tile-sized safety dilation."""
    W0, H0 = slide.level_dimensions[0]
    scale = max(1.0, W0 / float(target_thumbnail_w))
    tw, th = int(round(W0 / scale)), int(round(H0 / scale))
    thumb = np.asarray(slide.get_thumbnail((tw, th)).convert('RGB'), dtype=np.uint8)

    if _HAS_CV2:
        hsv = cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)
        S, V = hsv[..., 1], hsv[..., 2]
        if use_otsu:
            # Otsu on V channel — auto-calibrates per slide
            _, m_v = cv2.threshold(V, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        else:
            m_v = (V < val_thresh).astype(np.uint8)
        # Saturation guard catches faint eosin / pink stained regions Otsu may miss.
        # OR (additive only — extends tissue, never subtracts)
        m_s = (S > sat_thresh).astype(np.uint8)
        mask = (m_v.astype(np.uint8) | m_s).astype(np.uint8)
        # Morphological close (fill small holes inside tissue)
        k_close = max(3, int(round(tile_size_l0 / scale / 4)) | 1)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close)),
        )
        # Drop dust/specks below min_component_frac of slide area
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= max(4, int(mask.size * min_component_frac))
        keep[0] = False
        mask = keep[lab].astype(np.uint8)
        # Safety dilation by ~1 tile so edge tissue is never dropped
        if dilate_tiles > 0:
            k_dil = max(3, int(round(dilate_tiles * tile_size_l0 / scale)) | 1)
            mask = cv2.dilate(
                mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dil, k_dil)),
            )
    else:
        # skimage fallback
        from skimage.color import rgb2hsv
        from skimage.filters import threshold_otsu
        from skimage.morphology import (
            binary_closing, disk, remove_small_objects, binary_dilation,
        )
        hsv = rgb2hsv(thumb)
        V = (hsv[..., 2] * 255).astype(np.uint8)
        S = (hsv[..., 1] * 255).astype(np.uint8)
        if use_otsu:
            t_v = int(threshold_otsu(V))
            m_v = V < t_v
        else:
            m_v = V < val_thresh
        m_s = S > sat_thresh
        mask = m_v | m_s
        k_close = max(3, int(round(tile_size_l0 / scale / 4)))
        mask = binary_closing(mask, disk(k_close // 2))
        mask = remove_small_objects(mask, max(4, int(mask.size * min_component_frac)))
        if dilate_tiles > 0:
            k_dil = max(3, int(round(dilate_tiles * tile_size_l0 / scale)))
            mask = binary_dilation(mask, disk(k_dil // 2))
    return mask.astype(bool), float(scale)


def filter_tile_coords_by_mask(tile_coords, mask, mask_scale, *, frac: float = 0.01):
    Hm, Wm = mask.shape
    kept = []
    for (x0, y0, tw, th) in tile_coords:
        r0 = max(0, int(np.floor(y0 / mask_scale)))
        r1 = min(Hm, int(np.ceil((y0 + th) / mask_scale)) + 1)
        c0 = max(0, int(np.floor(x0 / mask_scale)))
        c1 = min(Wm, int(np.ceil((x0 + tw) / mask_scale)) + 1)
        if r1 > r0 and c1 > c0 and float(mask[r0:r1, c0:c1].mean()) >= frac:
            kept.append((x0, y0, tw, th))
    return kept, len(tile_coords) - len(kept)


def tissue_bbox_level0(mask, mask_scale, slide_dims, pad_px: int = 256):
    """Patch B: clip bbox to slide dimensions so we never read beyond the slide."""
    W0, H0 = slide_dims
    ys, xs = np.where(mask)
    if not len(xs):
        return 0, 0, 0, 0
    x0 = max(0, int(np.floor(xs.min() * mask_scale)) - pad_px)
    y0 = max(0, int(np.floor(ys.min() * mask_scale)) - pad_px)
    x1 = min(W0, int(np.ceil((xs.max() + 1) * mask_scale)) + pad_px)
    y1 = min(H0, int(np.ceil((ys.max() + 1) * mask_scale)) + pad_px)
    return x0, y0, x1 - x0, y1 - y0


def load_or_build_tissue_mask(slide, slide_path, out_dir, *, cache: bool = True, **kw):
    cache_path = Path(out_dir) / f'{Path(slide_path).stem}_tissue_mask.npz'
    if cache and cache_path.exists():
        d = np.load(cache_path)
        return d['mask'].astype(bool), float(d['scale'])
    mask, scale = compute_tissue_mask_thumbnail(slide, **kw)
    if cache:
        np.savez_compressed(cache_path, mask=mask.astype(np.uint8), scale=np.float32(scale))
    return mask, scale


# ── 4.2 GPU-side normalize forward (Patch A: bulk D2H, no autocast, no pin) ──
_IMNET_MEAN = (0.485, 0.456, 0.406)
_IMNET_STD  = (0.229, 0.224, 0.225)

@torch.inference_mode()
def batch_forward_v51(
    model,
    patches_hwc,
    device,
    *,
    fp16: bool = False,
    bf16: bool = False,
    channels_last: bool = False,
    cls_perm_gpu=None,
):
    """Diagnostic-only — returns full cls logits.

    Patch A: bulk D2H (3 syncs/batch), no autocast on pre-halved model,
    no per-batch pin_memory(), apply cls_perm once on GPU.
    Inputs: list of uint8 HWC numpy arrays.
    Returns: list of (prob, dist_map, cls_log) numpy arrays per tile.

    Kept in the active module so the single-tile diagnostic cell
    (cell 14 of infer_wsi_v54_gpu_peaks.ipynb) — which needs full fp32
    cls_logits for argmax visualization — keeps working unchanged."""
    if bf16:
        infer_dtype = torch.bfloat16
    elif fp16 and device.type == 'cuda':
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

    # Patch A: NO pin_memory() per batch — single-shot pinning is slower than skipping
    batch_u8 = torch.from_numpy(np.stack(arrs))                 # (N, H, W, 3) uint8
    batch = batch_u8.to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous()
    batch = batch.to(infer_dtype).div_(255.0)
    mean = torch.tensor(_IMNET_MEAN, device=device, dtype=infer_dtype).view(1, 3, 1, 1)
    std  = torch.tensor(_IMNET_STD,  device=device, dtype=infer_dtype).view(1, 3, 1, 1)
    batch = (batch - mean) / std
    if channels_last:
        batch = batch.contiguous(memory_format=torch.channels_last)

    # Patch A: NO autocast — model is already pre-halved; autocast would fight it
    prob_logit, dist_b, cls_b = model(batch)

    # Patch A: apply cls_perm ONCE on GPU (was per-tile numpy fancy index in v5)
    if cls_perm_gpu is not None:
        cls_b = cls_b.index_select(1, cls_perm_gpu)

    # Patch A: 3 BULK D2H transfers (3 syncs total, not 384 like v5)
    prob_all = torch.sigmoid(prob_logit).cpu().float().numpy()   # (N, 1, H, W)
    dist_all = dist_b.contiguous().cpu().float().numpy()         # (N, 32, H, W)
    cls_all  = cls_b.contiguous().cpu().float().numpy()          # (N, 19, H, W)

    return [
        (prob_all[i, 0, :ph, :pw],
         dist_all[i, :, :ph, :pw],
         cls_all[i, :, :ph, :pw])
        for i, (ph, pw) in enumerate(sizes)
    ]


# ── 4.4 Streaming GeoJSON writer (unchanged from v5) ─────────────────────────
def write_geojson_streaming(
    features_iter,
    path,
    *,
    gzip_compress: bool = False,
    flush_bytes: int = 1 << 20,
):
    opener = gzip.open if gzip_compress else open
    use_orjson = _HAS_ORJSON
    if use_orjson:
        opt = orjson.OPT_SERIALIZE_NUMPY
    n_written = 0
    with opener(str(path), 'wb') as f:
        f.write(b'{"type":"FeatureCollection","features":[')
        first = True
        buf = bytearray()
        for feat in features_iter:
            if first:
                first = False
            else:
                buf.append(0x2C)
            if use_orjson:
                buf += orjson.dumps(feat, option=opt)
            else:
                geom = feat.get('geometry', {})
                if geom.get('type') == 'Polygon':
                    coords = geom.get('coordinates', [])
                    coords_py = [
                        ring.tolist() if isinstance(ring, np.ndarray) else ring
                        for ring in coords
                    ]
                    feat = {**feat, 'geometry': {'type': 'Polygon', 'coordinates': coords_py}}
                buf += json.dumps(feat, separators=(',', ':')).encode('utf-8')
            n_written += 1
            if len(buf) > flush_bytes:
                f.write(buf); buf.clear()
        if buf:
            f.write(buf)
        f.write(b']}')
    return n_written


# ── 4.5 Auto LOAD_TO_RAM picker (Patch E) ────────────────────────────────────
def auto_load_to_ram(
    slide_path,
    slide,
    level: int,
    *,
    max_gb: float = 1.5,
    ram_frac: float = 0.40,
) -> tuple[bool, str]:
    """Decide LOAD_TO_RAM per-slide. Returns (decision, reason).
    Conservative: defaults to False (DataLoader mode) unless tiny + local + fits."""
    W, H = slide.level_dimensions[level]
    raw_bytes = W * H * 3
    raw_gb = raw_bytes / 1e9
    is_network = (
        str(slide_path).startswith(r'\\') or '://' in str(slide_path)
    )
    avail_gb = (psutil.virtual_memory().available / 1e9) if _HAS_PSUTIL else 8.0
    fits = raw_bytes < ram_frac * (avail_gb * 1e9)
    small = raw_gb < max_gb
    if is_network:
        return False, f'network share (raw={raw_gb:.1f} GB)'
    if not small:
        return False, f'too big (raw={raw_gb:.1f} GB > {max_gb} GB)'
    if not fits:
        return False, f'tight RAM (raw={raw_gb:.1f} GB, avail={avail_gb:.1f} GB)'
    return True, f'small + local + fits (raw={raw_gb:.1f} GB, avail={avail_gb:.1f} GB)'


# -- 4.8 GPU ring decoding helper (v5.3) --------------------------------------
def _grid_sample_dist_at_peaks(dist_b_gpu, tile_idx, ph, pw, peaks_rc, device):
    """Bilinear-sample dist_b_gpu[tile_idx, :, :ph, :pw] at sub-pixel peak rows/cols.

    Returns (K, R) numpy fp32 array — same shape and meaning as
    geometry.dist_at_points_bilinear, just executed on the GPU.

    align_corners=True + mode='bilinear' + padding_mode='border' matches
    scipy.ndimage.map_coordinates(order=1, mode='nearest') for integer/sub-pixel
    coordinates inside [0, H-1] x [0, W-1].
    """
    K = len(peaks_rc)
    if K == 0:
        R = dist_b_gpu.shape[1]
        return np.zeros((0, R), dtype=np.float32)

    peaks_t = torch.as_tensor(peaks_rc, device=device, dtype=torch.float32)
    rows = peaks_t[:, 0]
    cols = peaks_t[:, 1]
    # align_corners=True: pixel (i, j) center maps to (j/(W-1)*2-1, i/(H-1)*2-1).
    # For W==1 or H==1, max(W-1, 1) avoids div-by-zero (the resulting coord is
    # outside [-1, 1] but border padding clamps it to the single valid pixel).
    x = (cols / float(max(pw - 1, 1))) * 2.0 - 1.0
    y = (rows / float(max(ph - 1, 1))) * 2.0 - 1.0
    grid = torch.stack([x, y], dim=-1).view(1, K, 1, 2)  # (1, K, 1, 2)

    # Slice the un-padded region and add a fake batch axis.
    # grid_sample requires input and grid to share dtype; cast to fp32 (grid is always fp32).
    sample_in = dist_b_gpu[tile_idx:tile_idx + 1, :, :ph, :pw].float()  # (1, R, ph, pw)
    sampled = _F.grid_sample(
        sample_in,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )  # (1, R, K, 1)

    # (1, R, K, 1) -> (K, R). float() in case the model is in fp16/bf16.
    out = sampled.squeeze(-1).squeeze(0).transpose(0, 1).contiguous().float()
    return out.cpu().numpy()


# -- 4.9 GPU local_peaks (v5.4) ----------------------------------------------

@torch.inference_mode()
def batch_forward_v54(
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
    nms_dist: int = 8,
    prob_thresh: float = 0.45,
):
    """v5.4: like v5.3 plus GPU local_peaks via F.max_pool2d.

    Returns: (per_tile_results, dist_b_gpu, sizes) where:
      per_tile_results: list of (prob_np, cls_out_np, peaks_np) per tile
        peaks_np is (K_i, 2) int32 of (row, col) — pre-computed on the GPU.
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
    pad_v = vote_window_px // 2
    cls_avg = _F.avg_pool2d(cls_b, kernel_size=vote_window_px, stride=1, padding=pad_v)
    if include_class_probs:
        cls_probs_gpu = torch.softmax(cls_avg.float(), dim=1)
        cls_out_all = cls_probs_gpu.cpu().numpy()
    else:
        cls_argmax_gpu = cls_avg.argmax(dim=1).to(torch.int16)
        cls_out_all = cls_argmax_gpu.cpu().numpy()

    # prob_b stays in fp32 for the peak comparison and CPU lookups
    prob_b = torch.sigmoid(prob_logit).float()                # (N, 1, H, W)

    # GPU local_peaks via max_pool2d.
    # F.max_pool2d with kernel=k, padding=k//2 produces output of shape (H+1, W+1)
    # for even k. Crop to (H, W) so indexing matches scipy's same-shape output.
    nms = int(nms_dist)
    pad_p = nms // 2
    H_pad, W_pad = prob_b.shape[-2:]
    mx = _F.max_pool2d(prob_b, kernel_size=nms, stride=1, padding=pad_p)
    mx = mx[..., :H_pad, :W_pad]
    peak_mask = (prob_b >= mx) & (prob_b > float(prob_thresh))   # (N, 1, H, W) bool

    # One nonzero on the whole batch, then split by batch index on the host.
    nz = peak_mask.nonzero(as_tuple=False)                       # (total, 4): [b, c, r, x]
    nz_cpu = nz.cpu().numpy()                                    # small int array

    # D2H prob — still needed on CPU for prob_peak[pr, pc] and refine_local_com
    prob_all = prob_b.cpu().numpy()                              # (N, 1, H, W) fp32

    dist_b_gpu = dist_b.contiguous()

    per_tile = []
    for i, (ph, pw) in enumerate(sizes):
        cls_i = (
            cls_out_all[i, :, :ph, :pw] if include_class_probs
            else cls_out_all[i, :ph, :pw]
        )
        # Filter peaks for tile i to within the un-padded (ph, pw) region.
        mask_i = nz_cpu[:, 0] == i
        pks_i = nz_cpu[mask_i, 2:4]                              # (K_raw, 2): [row, col]
        if pks_i.size:
            keep = (pks_i[:, 0] < ph) & (pks_i[:, 1] < pw)
            pks_i = pks_i[keep]
        pks_i = pks_i.astype(np.int32, copy=False)
        per_tile.append((prob_all[i, 0, :ph, :pw], cls_i, pks_i))

    return per_tile, dist_b_gpu, sizes


def process_tile_batch_v54(
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
    """v5.4: GPU vote-class + GPU ring decoding + GPU local_peaks.
    The CPU loop only does peak refinement, ring assembly, and feature dict construction."""
    from geometry import (
        refine_peaks_local_com,
        dist_to_coord,
        polygon_ring_rowcol,
    )

    _t = time.perf_counter()
    per_tile, dist_b_gpu, sizes = batch_forward_v54(
        model, patches, device,
        fp16=fp16, bf16=bf16, channels_last=channels_last,
        cls_perm_gpu=cls_perm_gpu,
        vote_window_px=vote_window_px,
        include_class_probs=include_class_probs,
        nms_dist=nms_dist,
        prob_thresh=prob_thresh,
    )
    gpu_time = time.perf_counter() - _t

    vm = int(valid_margin)
    features = []

    try:
        for i, ((x0, y0, tw, th), (prob, cls_out, pks)) in enumerate(zip(meta, per_tile)):
            ph, pw = sizes[i]
            if not len(pks):
                continue
            # Refine peaks (CPU, cheap relative to ring decode).
            if refine_local_com:
                pts = refine_peaks_local_com(pks, prob, int(refine_radius_px))
            else:
                pts = pks.astype(np.float64)

            # Ring decoding on the GPU (from v5.3).
            dists = _grid_sample_dist_at_peaks(
                dist_b_gpu, i, ph, pw, pts, device,
            )                                                   # (K, R) fp32
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
        del dist_b_gpu

    return features, gpu_time
