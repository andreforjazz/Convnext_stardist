"""
inference_utils.py — all inference helpers for WSI nuclear segmentation.

Every function that would otherwise be defined inline in the notebook lives here
so the notebook cells contain only configuration + high-level orchestration.

Model / forward pass
--------------------
    load_model_and_classes(weights_path, config_path, device) -> (model, idx2label)
    run_tile(model, rgb_hwc, device)                          -> (prob, dist_map, cls_log)
    batch_forward_fast(model, patches, device, fp16)          -> list[(prob, dist_map, cls_log)]
    forward_batch_with_perm(model, patches, device, ...)      -> list[(prob, dist_map, cls_log)]
    process_tile_batch(meta, patches, model, device, ...)     -> (features, gpu_time)

Class ordering / colour palette
--------------------------------
    build_class_permutation(config_path, legacy_class_order) -> (cls_perm, idx2label)
    build_class_colormap(num_classes, idx2label, labels_viz, colors_viz) -> dict

Tile geometry
-------------
    build_tile_coords(w, h, tile_size, tile_overlap) -> list[(x0, y0, tw, th)]
    pick_diagnostic_tile(w, h, tile_size, mode, x, y) -> (sx, sy, rw, rh)

Classification / geometry
--------------------------
    vote_class(cls_log, coords_rc, tile_shape)    -> (cls_id, probs)
    polygon_ring_rowcol(coords_rc)                 -> np.ndarray (n_rays+1, 2)
    label_color(name, labels_viz, colors_viz)       -> (r, g, b) 0-255
    label_color_float(name, labels_viz, colors_viz) -> float32 [0,1] array

GeoJSON export
--------------
    feat_classified(feat)            -> GeoJSON Feature (class + colour)
    feat_segmentation_only(feat)     -> GeoJSON Feature (geometry only)
    write_geojson_feature_collection(path, features, ...) -> compact / gzip export

Slide reading / tile I/O
------------------------
    read_slide_region(slide, x0, y0, level, tw, th) -> (H,W,3) uint8
    get_tile_from_ram(ram_array, x0, y0, tw, th)    -> (H,W,3) uint8
    make_wsi_tile_dataloader(slide_path, ...)        -> torch DataLoader
"""
from __future__ import annotations

import gzip
import json
import math
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml


# ── ImageNet normalisation (same constants as training) ──────────────────────
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _import_build_model_v2():
    """Import build_model_v2; ensure repo root is on sys.path (Jupyter-friendly)."""
    try:
        from shared_convnext_stardist_decoder.model_v2 import build_model_v2
    except ModuleNotFoundError:
        # Notebook kernels often cwd != repo root and the package is not installed.
        repo_root = Path(__file__).resolve().parent.parent
        r = str(repo_root)
        if r not in sys.path:
            sys.path.insert(0, r)
        from shared_convnext_stardist_decoder.model_v2 import build_model_v2
    return build_model_v2


def _normalize_chw(rgb_hwc: np.ndarray) -> torch.Tensor:
    """uint8 HWC → float32 CHW normalised to ImageNet stats."""
    x = torch.from_numpy(rgb_hwc.astype(np.float32) / 255.0).permute(2, 0, 1)
    return (x - _IMAGENET_MEAN) / _IMAGENET_STD


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_and_classes(
    weights_path: str | Path,
    config_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict[int, str]]:
    """
    Load model from weights + config.  Returns (model, idx2label).

    idx2label maps integer class index → tissue name string.
    Class ordering follows model.class_names from the config file exactly —
    no alphabetical sorting or permutation is applied here.  If training was
    done with the legacy alphabetical order you still need PERMUTE logic in
    the notebook; this function just loads the model cleanly.
    """
    build_model_v2 = _import_build_model_v2()

    weights_path = Path(weights_path)
    config_path  = Path(config_path)

    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sd = torch.load(weights_path, map_location=device, weights_only=True)

    # Auto-detect checkpoint format: old checkpoints have a bare "head_cls.weight"
    # (single Conv2d); new checkpoints have "head_cls.0.weight" (Sequential).
    # Patch the config so the model is built to match the checkpoint exactly.
    if "head_cls.weight" in sd:
        cfg.setdefault("model", {})["head_cls_layers"] = 1
        # Also recover cls_semantic_dim from the checkpoint weight shape:
        # head_cls.weight shape = (num_classes, dc + S) → S = shape[1] - dc
        dc = cfg.get("model", {}).get("decoder_channels", 128)
        S  = int(sd["head_cls.weight"].shape[1]) - dc
        cfg["model"]["cls_semantic_dim"] = S
        print(f"[load_model] Detected legacy single-layer cls head "
              f"(cls_semantic_dim={S}). Loading in compatibility mode.")
    else:
        cfg.setdefault("model", {})["head_cls_layers"] = 2

    model = build_model_v2(cfg).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    class_names: list[str] = cfg.get("model", {}).get("class_names", [])
    idx2label = {i: str(n) for i, n in enumerate(class_names)}
    return model, idx2label


# ── Single-tile forward pass ──────────────────────────────────────────────────

@torch.no_grad()
def run_tile(
    model: nn.Module,
    rgb_hwc: np.ndarray,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model on one tile (uint8 HWC array).

    Returns
    -------
    prob     : (H, W)        float32  nucleus probability [0, 1]
    dist_map : (n_rays, H, W) float32  predicted ray distances
    cls_log  : (C, H, W)     float32  raw classifier logits
    """
    ph, pw = rgb_hwc.shape[:2]
    nh = int(math.ceil(ph / 32) * 32)
    nw = int(math.ceil(pw / 32) * 32)
    if nh != ph or nw != pw:
        pad = np.zeros((nh, nw, 3), dtype=np.uint8)
        pad[:ph, :pw] = rgb_hwc
        rgb_hwc = pad

    x = _normalize_chw(rgb_hwc).unsqueeze(0).to(device)
    prob_logit, dist_t, cls_logit = model(x)

    prob    = torch.sigmoid(prob_logit)[0, 0].float().cpu().numpy()[:ph, :pw]
    dist_map = dist_t[0].float().cpu().numpy()[:, :ph, :pw]
    cls_log  = cls_logit[0].float().cpu().numpy()[:, :ph, :pw]
    return prob, dist_map, cls_log


@torch.no_grad()
def batch_forward_fast(
    model: nn.Module,
    patches_hwc: list[np.ndarray],
    device: torch.device | str = "cpu",
    fp16: bool = False,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Forward pass on a batch of tiles (possibly different sizes).

    Each input tile is uint8 HWC.  Tiles are padded to the batch maximum
    multiple-of-32 size, stacked, run through the model, then unpadded.

    Returns a list of (prob, dist_map, cls_log) tuples, one per input tile.
    """
    infer_dtype = torch.float16 if (fp16 and str(device).startswith("cuda")) else torch.float32
    sizes = [(p.shape[0], p.shape[1]) for p in patches_hwc]
    max_h = int(math.ceil(max(s[0] for s in sizes) / 32) * 32)
    max_w = int(math.ceil(max(s[1] for s in sizes) / 32) * 32)

    tensors = []
    for p in patches_hwc:
        ph, pw = p.shape[:2]
        if ph != max_h or pw != max_w:
            pad = np.zeros((max_h, max_w, 3), dtype=np.uint8)
            pad[:ph, :pw] = p
            p = pad
        t = _normalize_chw(p)
        if infer_dtype == torch.float16:
            t = t.half()
        tensors.append(t)

    batch = torch.stack(tensors).to(device, non_blocking=True)
    with torch.amp.autocast("cuda", enabled=(infer_dtype == torch.float16)):
        prob_logit, dist_b, cls_b = model(batch)

    results = []
    for i, (ph, pw) in enumerate(sizes):
        prob    = torch.sigmoid(prob_logit[i, 0]).float().cpu().numpy()[:ph, :pw]
        dist_m  = dist_b[i].float().cpu().numpy()[:, :ph, :pw]
        cls_log = cls_b[i].float().cpu().numpy()[:, :ph, :pw]
        results.append((prob, dist_m, cls_log))
    return results


# ── Classification voting ─────────────────────────────────────────────────────

def vote_class(
    cls_log: np.ndarray,
    coords_rc: np.ndarray,
    tile_shape: tuple[int, int],
) -> tuple[int, np.ndarray]:
    """
    Assign a class to one nucleus by averaging logits over its polygon footprint.

    Parameters
    ----------
    cls_log   : (C, H, W) logit map
    coords_rc : (n_rays, 2) star-polygon vertex coordinates [row, col]
    tile_shape: (H, W) of the tile (for clipping)

    Returns
    -------
    cls_id : int  argmax class index (0-based)
    probs  : (C,) float32  softmax probabilities
    """
    H, W = tile_shape
    C = cls_log.shape[0]
    rows = np.clip(coords_rc[:, 0].astype(np.int64), 0, H - 1)
    cols = np.clip(coords_rc[:, 1].astype(np.int64), 0, W - 1)
    mean_logits = cls_log[:, rows, cols].mean(axis=1)        # (C,)
    e = np.exp(mean_logits - mean_logits.max())
    probs = (e / e.sum()).astype(np.float32)
    return int(probs.argmax()), probs


# ── Polygon geometry ──────────────────────────────────────────────────────────

def polygon_ring_rowcol(coords_rc: np.ndarray) -> np.ndarray:
    """
    Convert star-polygon ray-end coords → closed [col, row] ring for GeoJSON.

    Parameters
    ----------
    coords_rc : (n_rays, 2)  [row, col] vertex coordinates

    Returns
    -------
    ring : (n_rays+1, 2)  [col, row] with first == last (GeoJSON closed ring)
    """
    ring = np.empty((coords_rc.shape[0] + 1, 2), dtype=np.float32)
    ring[:-1, 0] = coords_rc[:, 1]   # col → x
    ring[:-1, 1] = coords_rc[:, 0]   # row → y
    ring[-1]     = ring[0]
    return ring


# ── Colour helpers ────────────────────────────────────────────────────────────

def label_color(
    name: str,
    labels_viz: Sequence[str],
    colors_viz: Sequence[Sequence[int]],
) -> tuple[int, int, int]:
    """
    Return (R, G, B) 0-255 for a tissue name from the LABELS_VIZ palette.
    Falls back to mid-grey if the name is not found.
    """
    key = str(name).strip().lower()
    for lab, col in zip(labels_viz, colors_viz):
        if lab.strip().lower() == key:
            return (int(col[0]), int(col[1]), int(col[2]))
    return (128, 128, 128)


def label_color_float(
    name: str,
    labels_viz: Sequence[str],
    colors_viz: Sequence[Sequence[int]],
) -> np.ndarray:
    """Same as label_color but returns float32 array [0, 1] for matplotlib."""
    r, g, b = label_color(name, labels_viz, colors_viz)
    return np.array([r / 255.0, g / 255.0, b / 255.0], dtype=np.float32)


# ── GeoJSON feature helpers ───────────────────────────────────────────────────

def feat_classified(feat: dict) -> dict:
    """
    Return a copy of a GeoJSON Feature keeping only classification properties
    (name, index) and prob_peak.  Used for the classified GeoJSON export.
    """
    p = feat["properties"]
    return {
        "type":     "Feature",
        "id":       feat["id"],
        "geometry": feat["geometry"],
        "properties": {
            "classification": p.get("classification", {}),
            "prob_peak":      p.get("prob_peak", 0.0),
        },
    }


def feat_segmentation_only(feat: dict) -> dict:
    """
    Return a GeoJSON Feature stripped of all classification info.
    Used for the plain segmentation-only export (compatible with QuPath/Fiji).
    """
    return {
        "type":       "Feature",
        "id":         feat["id"],
        "geometry":   feat["geometry"],
        "properties": {"prob_peak": feat["properties"].get("prob_peak", 0.0)},
    }


def round_geometry_coords(geom: dict, ndigits: int) -> dict:
    """
    Return a shallow copy of *geom* with all coordinate pairs rounded to *ndigits*.
    Supports Polygon and MultiPolygon.  Sub-pixel digits are wasted in GeoJSON text.
    """
    gtype = geom.get("type")
    if gtype == "Polygon":
        return {
            "type": "Polygon",
            "coordinates": [
                [[round(float(x), ndigits), round(float(y), ndigits)] for x, y in ring]
                for ring in geom["coordinates"]
            ],
        }
    if gtype == "MultiPolygon":
        return {
            "type": "MultiPolygon",
            "coordinates": [
                [
                    [[round(float(x), ndigits), round(float(y), ndigits)] for x, y in ring]
                    for ring in poly
                ]
                for poly in geom["coordinates"]
            ],
        }
    return geom


def feature_with_rounded_geometry(feat: dict, ndigits: int | None) -> dict:
    """Copy *feat*; optionally replace geometry with coordinate-rounded version."""
    if ndigits is None:
        return feat
    out = {**feat, "geometry": round_geometry_coords(feat["geometry"], ndigits)}
    return out


def write_geojson_feature_collection(
    path: str | Path,
    features: list[dict],
    *,
    coord_decimals: int | None = 2,
    indent: int | None = None,
    gzip_compress: bool = False,
) -> None:
    """
    Write a FeatureCollection to *path* with compact JSON (fast, small on disk).

    Parameters
    ----------
    coord_decimals
        If set, round every polygon coordinate to this many decimal places.
        ``2`` is usually visually identical at full-resolution WSI; ``None`` keeps
        full float strings (larger files).
    indent
        ``None`` → smallest output (recommended for huge exports).  Use ``2`` only
        for small debug files.
    gzip_compress
        If True, writes gzip-compressed JSON (use a ``.geojson.gz`` path; QuPath
        may need you to decompress first, or open via a tool that supports gzip).
    """
    path = Path(path)
    if coord_decimals is not None:
        feats_out = [feature_with_rounded_geometry(f, coord_decimals) for f in features]
    else:
        feats_out = features

    payload = {"type": "FeatureCollection", "features": feats_out}
    kwargs: dict = {"ensure_ascii": False}
    if indent is None:
        kwargs["separators"] = (",", ":")
    else:
        kwargs["indent"] = indent

    text = json.dumps(payload, **kwargs)

    if gzip_compress:
        with gzip.open(path, "wt", encoding="utf-8", newline="\n") as gz:
            gz.write(text)
    else:
        path.write_text(text, encoding="utf-8")


# ── Slide reading ─────────────────────────────────────────────────────────────

def read_slide_region(
    slide,
    x0: int,
    y0: int,
    level: int,
    tile_w: int,
    tile_h: int,
) -> np.ndarray:
    """Read a region from an OpenSlide object; returns (H, W, 3) uint8."""
    ds = slide.level_downsamples[level]
    region = slide.read_region(
        (int(x0 * ds), int(y0 * ds)), level, (tile_w, tile_h)
    )
    return np.asarray(region.convert("RGB"), dtype=np.uint8)


def get_tile_from_ram(
    ram_array: np.ndarray,
    x0: int,
    y0: int,
    tw: int,
    th: int,
) -> np.ndarray:
    """Slice a tile from a slide already loaded into a numpy (H, W, 3) array."""
    return ram_array[y0 : y0 + th, x0 : x0 + tw].copy()


def build_tile_coords(
    w_slide: int,
    h_slide: int,
    tile_size: int,
    tile_overlap: int,
) -> list[tuple[int, int, int, int]]:
    """
    Build the full list of ``(x0, y0, tile_width, tile_height)`` for a WSI tile grid.

    Edge tiles may be smaller than ``tile_size``.
    Tiles with either dimension < 8 px are dropped.
    ``step = tile_size - tile_overlap``
    """
    step = max(1, tile_size - int(tile_overlap))
    coords: list[tuple[int, int, int, int]] = []
    for y0 in range(0, h_slide, step):
        for x0 in range(0, w_slide, step):
            tw = min(tile_size, w_slide - x0)
            th = min(tile_size, h_slide - y0)
            if tw >= 8 and th >= 8:
                coords.append((x0, y0, tw, th))
    return coords


def pick_diagnostic_tile(
    w_slide: int,
    h_slide: int,
    tile_size: int,
    mode: str = "centre",
    sample_x: int = 0,
    sample_y: int = 0,
) -> tuple[int, int, int, int]:
    """
    Pick a representative tile position for visual diagnostics.

    Parameters
    ----------
    mode : ``"centre"`` | ``"random"`` | ``"fixed"``
    sample_x, sample_y : top-left corner in pixels, only used when ``mode="fixed"``

    Returns
    -------
    sx, sy : top-left corner of the tile (pixels at SLIDE_LEVEL)
    rw, rh : actual tile size (may be < tile_size at the slide edges)
    """
    import random as _random

    if mode == "fixed":
        sx, sy = int(sample_x), int(sample_y)
    elif mode == "random":
        sx = _random.randint(0, max(0, w_slide - tile_size))
        sy = _random.randint(0, max(0, h_slide - tile_size))
    else:  # "centre"
        sx = max(0, w_slide // 2 - tile_size // 2)
        sy = max(0, h_slide // 2 - tile_size // 2)
    rw = min(tile_size, w_slide - sx)
    rh = min(tile_size, h_slide - sy)
    return sx, sy, rw, rh


def build_class_permutation(
    config_path: str | Path,
    legacy_class_order: list[str],
) -> tuple[np.ndarray, dict[int, str]]:
    """
    Build a class-index permutation array for checkpoints trained with legacy
    alphabetical class ordering (pre-2025 runs).

    Reads ``model.class_names`` from *config_path* (the *target* display order),
    then maps each target name back to its position in *legacy_class_order*.

    Applying ``cls_log[cls_perm]`` to a raw (C, H, W) logit map reorders the
    channels so that index ``i`` corresponds to ``class_names[i]``.

    Returns
    -------
    cls_perm  : int64 ndarray, shape (num_classes,)
    idx2label : dict mapping display index → tissue name
    """
    config_path = Path(config_path)
    with config_path.open(encoding="utf-8") as f:
        cn_target: list[str] = yaml.safe_load(f).get("model", {}).get("class_names", [])

    norm = lambda s: str(s).strip().lower()
    leg_to_idx = {norm(n): i for i, n in enumerate(legacy_class_order)}

    missing = {norm(n) for n in cn_target} - set(leg_to_idx)
    extra   = set(leg_to_idx) - {norm(n) for n in cn_target}
    if missing or extra:
        raise ValueError(
            "build_class_permutation: config class_names and legacy_class_order must "
            f"contain the same tissue names.\n  missing from legacy : {missing}\n"
            f"  extra in legacy    : {extra}"
        )

    cls_perm  = np.array([leg_to_idx[norm(n)] for n in cn_target], dtype=np.int64)
    idx2label = {i: n for i, n in enumerate(cn_target)}
    return cls_perm, idx2label


def build_class_colormap(
    num_classes: int,
    idx2label: dict[int, str],
    labels_viz: Sequence[str],
    colors_viz: Sequence[Sequence[int]],
) -> dict:
    """
    Build colour helpers from the display palette.

    Returns a dict with:

    ``"colors_float"``
        ``(num_classes, 3)`` float32 array with RGB values in ``[0, 1]``
    ``"idx_to_rgb_int"``
        ``dict[int, tuple[int,int,int]]`` for GeoJSON ``colorRGB`` packing
    ``"color_for_idx"``
        ``callable(cls_idx) -> np.ndarray`` float RGB ``[0,1]`` for matplotlib
    ``"color_for_name"``
        ``callable(name) -> tuple[int,int,int]`` uint8 RGB for GeoJSON
    ``"cmap"``
        ``matplotlib.colors.ListedColormap`` aligned to class indices
        (``None`` if matplotlib is unavailable)
    ``"norm"``
        ``matplotlib.colors.BoundaryNorm`` for the above cmap
        (``None`` if matplotlib is unavailable)
    """
    colors_float = np.clip(
        np.array(
            [label_color_float(idx2label.get(i, str(i)), labels_viz, colors_viz)
             for i in range(num_classes)],
            dtype=np.float32,
        ),
        0.0, 1.0,
    )
    idx_to_rgb_int = {
        i: label_color(idx2label.get(i, str(i)), labels_viz, colors_viz)
        for i in range(num_classes)
    }

    def color_for_idx(cls_idx: int) -> np.ndarray:
        return colors_float[int(cls_idx) % num_classes]

    def color_for_name(name: str) -> tuple[int, int, int]:
        return label_color(name, labels_viz, colors_viz)

    try:
        from matplotlib.colors import BoundaryNorm, ListedColormap
        cmap = ListedColormap(colors_float)
        norm = BoundaryNorm(np.arange(-0.5, num_classes + 0.5, 1), cmap.N)
    except ImportError:
        cmap = norm = None

    return {
        "colors_float":  colors_float,
        "idx_to_rgb_int": idx_to_rgb_int,
        "color_for_idx":  color_for_idx,
        "color_for_name": color_for_name,
        "cmap":           cmap,
        "norm":           norm,
    }


# ── Per-batch inference helpers ───────────────────────────────────────────────

def forward_batch_with_perm(
    model: nn.Module,
    patches: list[np.ndarray],
    device: torch.device | str,
    *,
    fp16: bool = False,
    cls_perm: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Batch forward pass with optional class-channel permutation.

    Calls ``batch_forward_fast``, then reorders logit channels by ``cls_perm``
    if provided (needed for checkpoints trained with alphabetical class order).

    Returns a list of ``(prob, dist_map, cls_log)`` tuples, one per input tile.
    """
    results = batch_forward_fast(model, patches, device, fp16=fp16)
    if cls_perm is not None:
        results = [(p, d, c[cls_perm]) for p, d, c in results]
    return results


def process_tile_batch(
    meta: list[tuple[int, int, int, int]],
    patches: list[np.ndarray],
    model: nn.Module,
    device: torch.device | str,
    *,
    fp16: bool = False,
    cls_perm: np.ndarray | None = None,
    nms_dist: int = 8,
    prob_thresh: float = 0.45,
    refine_local_com: bool = True,
    refine_radius_px: int = 8,
    valid_margin: int = 0,
    w_slide: int,
    h_slide: int,
    idx2label: dict[int, str],
) -> tuple[list[dict], float]:
    """
    Run one GPU batch: forward pass → peak detection → polygon extraction.

    Parameters
    ----------
    meta     : ``(x0, y0, tile_w, tile_h)`` for each patch
    patches  : uint8 ``(H, W, 3)`` numpy arrays, same order as *meta*
    w_slide, h_slide : full-slide dimensions at inference level (for edge detection)

    Returns
    -------
    features : list of GeoJSON Feature dicts — **id is NOT set** (caller assigns)
    gpu_time : seconds spent in the forward pass
    """
    # lazy import: geometry requires scipy / skimage which may not be available at module load
    # Import vote_class and polygon_ring_rowcol from geometry (not the inference_utils
    # copies) because dists_and_coords_from_peaks returns (n_poly, 2, n_rays) coords —
    # geometry.py's functions expect (2, n_rays) per nucleus, while the inference_utils
    # copies expect (n_rays, 2), which would create degenerate 2-point polygons.
    from geometry import (  # type: ignore
        local_peaks,
        dists_and_coords_from_peaks,
        vote_class,
        polygon_ring_rowcol,
    )

    _t = time.perf_counter()
    results = forward_batch_with_perm(model, patches, device, fp16=fp16, cls_perm=cls_perm)
    gpu_time = time.perf_counter() - _t

    vm = int(valid_margin) if valid_margin else 0
    features: list[dict] = []

    for (x0, y0, tw, th), (prob, dist_m, cls_l) in zip(meta, results):
        pks = local_peaks(prob, min_distance=int(nms_dist), thresh=float(prob_thresh))
        if not len(pks):
            continue
        _, coords = dists_and_coords_from_peaks(
            dist_m, pks, prob,
            refine_local_com=refine_local_com,
            refine_radius_px=int(refine_radius_px),
        )
        _left   = x0 == 0
        _top    = y0 == 0
        _right  = x0 + tw >= w_slide
        _bottom = y0 + th >= h_slide
        for k in range(coords.shape[0]):
            pr, pc = int(pks[k, 0]), int(pks[k, 1])
            if vm:
                if not _left   and pc < vm:         continue
                if not _right  and pc >= tw - vm:   continue
                if not _top    and pr < vm:          continue
                if not _bottom and pr >= th - vm:    continue
            cls_id, probs_k = vote_class(cls_l, coords[k], (th, tw))
            ring = polygon_ring_rowcol(coords[k]) + np.array([x0, y0], dtype=np.float32)
            name = idx2label.get(cls_id, f"class_{cls_id}")
            features.append({
                "type": "Feature", "id": "",
                "geometry": {"type": "Polygon", "coordinates": [ring.tolist()]},
                "properties": {
                    "classification": {"name": name, "index": int(cls_id)},
                    "prob_peak": float(prob[pr, pc]),
                    "class_probs": {
                        idx2label.get(i, str(i)): float(p)
                        for i, p in enumerate(probs_k)
                    },
                },
            })
    return features, gpu_time


def postprocess_batch_v4(
    meta: list[tuple[int, int, int, int]],
    results: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    nms_dist: int = 8,
    prob_thresh: float = 0.45,
    refine_local_com: bool = True,
    refine_radius_px: int = 8,
    valid_margin: int = 0,
    w_slide: int,
    h_slide: int,
    idx2label: dict[int, str],
    cls_perm: np.ndarray | None = None,
    vote_window_px: int = 9,
    include_class_probs: bool = True,
) -> list[dict]:
    """CPU-only post-processing for a batch of forward-pass results.

    Separating this from the GPU forward pass allows the caller to pipeline:
    run this in a background thread while the next GPU batch is already in flight.

    ``local_peaks`` and ``dists_and_coords_from_peaks`` are dispatched in parallel
    across tiles via a thread pool (both functions release the GIL in C/Cython).
    """
    from geometry import local_peaks, dists_and_coords_from_peaks  # type: ignore
    from scipy.ndimage import uniform_filter as _uf
    from scipy.special import softmax as _softmax
    import concurrent.futures as _cf

    vm = int(valid_margin) if valid_margin else 0
    features: list[dict] = []

    def _tile_peaks(args: tuple) -> tuple:
        prob, dist_m = args
        pks = local_peaks(prob, min_distance=int(nms_dist), thresh=float(prob_thresh))
        if not len(pks):
            return None, None
        _, coords = dists_and_coords_from_peaks(
            dist_m, pks, prob,
            refine_local_com=refine_local_com,
            refine_radius_px=int(refine_radius_px),
        )
        return pks, coords

    # Parallel local_peaks across tiles — each call is independent and releases GIL
    n_workers = min(len(results), 8)
    with _cf.ThreadPoolExecutor(max_workers=n_workers) as _pool:
        peaks_coords = list(_pool.map(_tile_peaks, [(p, d) for p, d, _ in results]))

    for (x0, y0, tw, th), (prob, _, cls_l), (pks, coords) in zip(meta, results, peaks_coords):
        if pks is None or not len(pks):
            continue

        # ── Vectorized margin filter ──────────────────────────────────────────
        if vm:
            keep = np.ones(len(pks), dtype=bool)
            if x0 != 0:           keep &= pks[:, 1] >= vm
            if x0 + tw < w_slide: keep &= pks[:, 1] < tw - vm
            if y0 != 0:           keep &= pks[:, 0] >= vm
            if y0 + th < h_slide: keep &= pks[:, 0] < th - vm
            pks    = pks[keep]
            coords = coords[keep]
            if not len(pks):
                continue

        # ── Batch logit extraction + softmax ──────────────────────────────────
        cls_avg    = _uf(cls_l, size=(1, vote_window_px, vote_window_px), mode='nearest')
        logits_all = cls_avg[:, pks[:, 0], pks[:, 1]]   # (C, N)
        probs_all  = _softmax(logits_all, axis=0)        # (C, N)
        raw_ids    = probs_all.argmax(axis=0)            # (N,)
        cls_ids    = cls_perm[raw_ids] if cls_perm is not None else raw_ids
        if cls_perm is not None and include_class_probs:
            probs_all = probs_all[cls_perm]

        # ── Batch polygon rings ───────────────────────────────────────────────
        rings = np.stack([coords[:, 1, :], coords[:, 0, :]], axis=2)  # (N, n_rays, 2) [x,y]
        rings = np.concatenate([rings, rings[:, :1, :]], axis=1)       # close ring
        rings += np.array([x0, y0], dtype=np.float32)

        # ── Dict construction (unavoidably per-nucleus) ───────────────────────
        for k in range(len(pks)):
            pr, pc  = int(pks[k, 0]), int(pks[k, 1])
            cls_id  = int(cls_ids[k])
            name    = idx2label.get(cls_id, f"class_{cls_id}")
            feat: dict = {
                "type": "Feature", "id": "",
                "geometry": {"type": "Polygon", "coordinates": [rings[k].tolist()]},
                "properties": {
                    "classification": {"name": name, "index": cls_id},
                    "prob_peak": float(prob[pr, pc]),
                },
            }
            if include_class_probs:
                feat["properties"]["class_probs"] = {
                    idx2label.get(i, str(i)): float(probs_all[i, k])
                    for i in range(probs_all.shape[0])
                }
            features.append(feat)

    return features


def process_tile_batch_v4(
    meta: list[tuple[int, int, int, int]],
    patches: list[np.ndarray],
    model: nn.Module,
    device: torch.device | str,
    *,
    fp16: bool = False,
    cls_perm: np.ndarray | None = None,
    nms_dist: int = 8,
    prob_thresh: float = 0.45,
    refine_local_com: bool = True,
    refine_radius_px: int = 8,
    valid_margin: int = 0,
    w_slide: int,
    h_slide: int,
    idx2label: dict[int, str],
    # v4 extras
    use_fast_vote_class: bool = False,
    vote_window_px: int = 64,
    include_class_probs: bool = True,
) -> tuple[list[dict], float]:
    """
    v4 wrapper around :func:`process_tile_batch`.

    Extra v4 parameters
    -------------------
    use_fast_vote_class
        If True, average cls logits over a square window around each peak
        (``vote_window_px × vote_window_px``) rather than the full polygon
        interior.  Faster but slightly less accurate for large nuclei.
    vote_window_px
        Side length of the square averaging window when ``use_fast_vote_class``
        is True.  Ignored otherwise.
    include_class_probs
        If False, strip ``class_probs`` from each feature's properties before
        returning (smaller GeoJSON, faster export).
    """
    if use_fast_vote_class:
        _t = time.perf_counter()
        results = forward_batch_with_perm(model, patches, device, fp16=fp16, cls_perm=cls_perm)
        gpu_time = time.perf_counter() - _t
        features = postprocess_batch_v4(
            meta, results,
            nms_dist=nms_dist, prob_thresh=prob_thresh,
            refine_local_com=refine_local_com, refine_radius_px=refine_radius_px,
            valid_margin=valid_margin, w_slide=w_slide, h_slide=h_slide,
            idx2label=idx2label, cls_perm=cls_perm,
            vote_window_px=vote_window_px, include_class_probs=include_class_probs,
        )
        return features, gpu_time

    # Standard path — delegate to process_tile_batch
    features, gpu_time = process_tile_batch(
        meta, patches, model, device,
        fp16=fp16, cls_perm=cls_perm, nms_dist=nms_dist, prob_thresh=prob_thresh,
        refine_local_com=refine_local_com, refine_radius_px=refine_radius_px,
        valid_margin=valid_margin, w_slide=w_slide, h_slide=h_slide,
        idx2label=idx2label,
    )
    if not include_class_probs:
        for f in features:
            f["properties"].pop("class_probs", None)
    return features, gpu_time


def write_geojson_feature_collection_v4(
    path: str | Path,
    features: list[dict],
    *,
    coord_decimals: int | None = 2,
    gzip_compress: bool = False,
    use_orjson: bool = False,
) -> None:
    """
    v4 GeoJSON writer.  Identical to :func:`write_geojson_feature_collection`
    but adds ``use_orjson`` for faster serialisation on large exports.

    Parameters
    ----------
    use_orjson
        If True and ``orjson`` is installed, use it instead of stdlib ``json``
        (2-4× faster, lower peak memory).  Falls back silently to stdlib if
        ``orjson`` is not installed.
    """
    if use_orjson:
        try:
            import orjson  # type: ignore

            path = Path(path)
            if coord_decimals is not None:
                feats_out = [feature_with_rounded_geometry(f, coord_decimals) for f in features]
            else:
                feats_out = features
            payload = {"type": "FeatureCollection", "features": feats_out}
            raw = orjson.dumps(payload)
            if gzip_compress:
                with gzip.open(path, "wb") as gz:
                    gz.write(raw)
            else:
                Path(path).write_bytes(raw)
            return
        except ImportError:
            pass  # fall through to stdlib

    write_geojson_feature_collection(
        path, features,
        coord_decimals=coord_decimals,
        gzip_compress=gzip_compress,
    )


# ── DataLoader tile pipeline (disk-backed WSI; avoids starving the GPU) ───────


class WSITileDataset(Dataset):
    """
    Random-access RGB tiles from an on-disk WSI or from a full-slide RAM array.

    For ``ram_hwc`` set, use only with ``DataLoader(..., num_workers=0)`` — otherwise
    worker processes may each copy the whole slide (especially on Windows spawn).
    """

    def __init__(
        self,
        slide_path: str,
        slide_level: int,
        tile_coords: list[tuple[int, int, int, int]],
        ram_hwc: np.ndarray | None = None,
    ):
        self.slide_path = slide_path
        self.slide_level = int(slide_level)
        self.tile_coords = tile_coords
        self.ram_hwc = ram_hwc
        self._slide = None

    def __len__(self) -> int:
        return len(self.tile_coords)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        x0, y0, tw, th = self.tile_coords[idx]
        if self.ram_hwc is not None:
            tile = get_tile_from_ram(self.ram_hwc, x0, y0, tw, th)
        else:
            if self._slide is None:
                import openslide

                self._slide = openslide.OpenSlide(self.slide_path)
            tile = read_slide_region(self._slide, x0, y0, self.slide_level, tw, th)
        return tile, idx


def collate_wsi_tile_batch(
    batch: list[tuple[np.ndarray, int]],
) -> tuple[list[np.ndarray], list[int]]:
    tiles = [b[0] for b in batch]
    idxs = [int(b[1]) for b in batch]
    return tiles, idxs


def make_wsi_tile_dataloader(
    slide_path: str,
    slide_level: int,
    tile_coords: list[tuple[int, int, int, int]],
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> DataLoader:
    """
    Build a ``DataLoader`` over `WSITileDataset` (disk-backed tiles only).

    ``prefetch_factor`` and ``persistent_workers`` apply only when ``num_workers > 0``.
    """
    ds = WSITileDataset(slide_path, slide_level, tile_coords, ram_hwc=None)
    kw = {
        "dataset": ds,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "collate_fn": collate_wsi_tile_batch,
    }
    if int(num_workers) > 0:
        kw["prefetch_factor"] = int(prefetch_factor)
        kw["persistent_workers"] = bool(persistent_workers)
    return DataLoader(**kw)
