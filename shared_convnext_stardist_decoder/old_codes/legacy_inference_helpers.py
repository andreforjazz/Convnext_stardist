"""Legacy / superseded helpers extracted from `inference_utils.py`.

These functions were either duplicates of canonical versions in `geometry.py`
(with a conflicting array-shape convention) or vestigial v4 helpers superseded
by the v5.4 pipeline.  They are kept here for archival reference only — none
of the active code paths call into this module.

What lives here, and what supersedes each
-----------------------------------------
- `vote_class`
    Duplicate of `geometry.vote_class`.  The active code (notebooks via
    `from geometry import *` and `inference_v54.py` via explicit import) uses
    the geometry version.  The two implementations agree on the integer-class
    output but disagree on the (row, col) vs (col, row) array-shape convention,
    so they are NOT interchangeable.  Use `geometry.vote_class`.

- `polygon_ring_rowcol`
    Duplicate of `geometry.polygon_ring_rowcol`, same shape-convention issue
    as `vote_class`.  Use `geometry.polygon_ring_rowcol`.

- `process_tile_batch_v4`
    v4 wrapper around `inference_utils.process_tile_batch` adding
    `use_fast_vote_class` / `vote_window_px` / `include_class_probs` shims.
    Superseded by `inference_v54.process_tile_batch_v54` (GPU vote-class +
    GPU ring decoding + GPU local_peaks).

- `write_geojson_feature_collection_v4`
    v4 GeoJSON writer with optional `orjson` fast path.  Superseded by
    `inference_v54.write_geojson_streaming` which streams features incrementally
    and avoids holding the whole FeatureCollection in memory.

Note
----
`postprocess_batch_v4` is NOT archived here.  It is still imported by
`eval_classification.ipynb` (cell 1) and called from cell 5 of that notebook,
so it remains in `inference_utils.py`.

Imports below mirror `inference_utils.py`'s top-of-file imports plus the
cross-module references the v4 wrappers need (they delegate back into the
active `inference_utils` module for `process_tile_batch`, `forward_batch_with_perm`,
`postprocess_batch_v4`, `feature_with_rounded_geometry`, and
`write_geojson_feature_collection`).
"""
from __future__ import annotations

import gzip
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Active-module dependencies still referenced by v4 wrapper bodies.
from inference_utils import (
    forward_batch_with_perm,
    process_tile_batch,
    postprocess_batch_v4,
    feature_with_rounded_geometry,
    write_geojson_feature_collection,
)


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
