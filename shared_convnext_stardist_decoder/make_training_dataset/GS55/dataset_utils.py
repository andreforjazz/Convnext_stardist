"""
dataset_utils.py — shared helpers for the GS55 training dataset pipeline.

All reusable logic lives here; the notebooks contain only configuration and
high-level orchestration.

Public API
----------
Notebook 0 — CODA GeoJSON building
    assign_coda_to_geojson(geojson_path, mask_path, out_path, ...)
    polygon_centroid(ring)

Notebook 1 — Cell-type analysis
    extract_cell_types_from_geojson(geojson_path)
    normalize_slide_stem(geojson_path)

Notebook 2 — Tile dataset building
    calculate_hybrid_weights(class_counts)
    get_slide_mpp(slide)
    assign_cells_to_tiles(cells_xy, tile_size, stride)
    augment_image(img, aug_id)
    augment_coords(x, y, aug_id, tile_size)
    augment_polygon_ring(ring, aug_id, tile_size)
    clip_features_to_tile(features, tile_x, tile_y, tile_size, cell_indices)
    rasterize_tile_features(features, tile_size)
    write_tile_geojson(features, out_path)
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd


# ── Notebook 0: CODA GeoJSON building ─────────────────────────────────────────

def polygon_centroid(
    ring: List[List[float]] | np.ndarray,
) -> Tuple[float | None, float | None]:
    """
    Return the (x, y) centroid of a closed polygon ring.

    Handles both lists of [x, y] and numpy arrays.  Returns (None, None) if
    the ring has fewer than 3 points.
    """
    pts = np.asarray(ring, dtype=np.float64)
    if pts.shape[0] < 3:
        return None, None
    # close the ring if it isn't already
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    cx = pts[:-1, 0].mean()
    cy = pts[:-1, 1].mean()
    return float(cx), float(cy)


def assign_coda_to_geojson(
    geojson_path: str | Path,
    mask_path: str | Path,
    out_path: str | Path,
    labels: List[str],
    colors: List[List[int]],
    mpp_20x: float = 0.4416,
    mpp_mask: float = 2.0,
) -> int:
    """
    Assign CODA organ-class labels to a StarDist nuclear GeoJSON.

    Unlike the GS40 version this function requires **no bounding-box file**.
    It assumes the CODA mask covers the entire slide at ``mpp_mask`` resolution,
    so centroids are converted from 20x coordinates to mask coordinates by a
    single scale factor:

        mask_x = geojson_x * (mpp_20x / mpp_mask)
        mask_y = geojson_y * (mpp_20x / mpp_mask)

    Parameters
    ----------
    geojson_path : path to input StarDist GeoJSON (polygons in 20x coordinates)
    mask_path    : path to CODA classification mask (.tif or .png, uint8 pixel
                   values are 1-based class indices)
    out_path     : path for the labelled output GeoJSON
    labels       : list of class names, 0-indexed  (mask value k → labels[k-1])
    colors       : matching list of [R, G, B] display colours
    mpp_20x      : microns-per-pixel at 20x (GeoJSON coordinate space)
    mpp_mask     : microns-per-pixel of the CODA mask

    Returns
    -------
    n_assigned : number of nuclei that received a valid class label
    """
    geojson_path = Path(geojson_path)
    out_path = Path(out_path)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    scale = mpp_20x / mpp_mask  # 20x → mask pixel scale factor

    try:
        data = json.loads(geojson_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[SKIP] Corrupted GeoJSON {geojson_path.name}: {exc}")
        return 0

    feats = data if isinstance(data, list) else data.get("features", [])

    # 1-indexed class map (mask pixel value k → name/colour)
    label_map = {
        i + 1: {"name": labels[i], "color": colors[i]}
        for i in range(len(labels))
    }

    n_assigned = 0
    for feat in feats:
        geom = feat.get("geometry", {})
        if geom.get("type") != "Polygon":
            continue
        ring = geom.get("coordinates", [[]])[0]
        cx, cy = polygon_centroid(ring)
        if cx is None:
            continue

        # Map centroid to mask pixel coordinates
        mx = int(round(cx * scale))
        my = int(round(cy * scale))

        if 0 <= mx < mask.shape[1] and 0 <= my < mask.shape[0]:
            val = int(mask[my, mx])
            cls = label_map.get(val, {"name": "Unassigned", "color": [128, 128, 128]})
        else:
            cls = {"name": "OutsideMask", "color": [0, 0, 0]}

        feat.setdefault("properties", {})["classification"] = cls
        if cls["name"] not in ("Unassigned", "OutsideMask"):
            n_assigned += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )
    return n_assigned


# ── Notebook 1: cell-type analysis ────────────────────────────────────────────

def extract_cell_types_from_geojson(
    geojson_path: Path,
) -> Dict[str, int]:
    """
    Count occurrences of each classification name in a GeoJSON file.

    Returns an empty dict on any error (file not found, invalid JSON, …).
    """
    try:
        data = json.loads(geojson_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, FileNotFoundError) as exc:
        print(f"[ERROR] {geojson_path.name}: {exc}")
        return {}

    feats = data if isinstance(data, list) else data.get("features", [])
    counts: Counter = Counter()
    for feat in feats:
        name = feat.get("properties", {}).get("classification", {}).get("name")
        if name:
            counts[name] += 1
    return dict(counts)


def normalize_slide_stem(geojson_path: Path) -> str:
    """
    Strip common CODA suffixes from a GeoJSON stem to recover the original slide ID.

    Examples
    --------
    ``"slide_001__CODAclass.geojson"`` → ``"slide_001"``
    """
    stem = geojson_path.stem
    for suffix in ("__CODAclass", "_CODAclass", "-CODAclass", "_annotated", "__annotated"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


# ── Notebook 2: tile dataset building ─────────────────────────────────────────

def calculate_hybrid_weights(
    class_counts: pd.Series,
    rare_pct: float = 0.01,
    medium_pct: float = 0.10,
    very_common_pct: float = 0.20,
) -> pd.Series:
    """
    Compute per-class tile-sampling weights using a 4-tier strategy.

    Tier thresholds are expressed as fractions of total cells.

    ==============================  ============================================
    Tier                            Strategy
    ==============================  ============================================
    RARE       (< ``rare_pct``)     Oversample to reference level (cap 10×)
    MEDIUM     (< ``medium_pct``)   Moderate adjustment to reference (min 50%)
    COMMON     (< ``very_common_pct``)  Downsample to 30% of reference
    VERY COMMON (≥ ``very_common_pct``) Aggressive downsample to 10% of ref
    ==============================  ============================================

    The *reference* is 70% of the median count across MEDIUM-tier classes.

    Returns a ``pd.Series`` of float weights, same index as ``class_counts``.
    """
    total = class_counts.sum()
    rare_thr        = total * rare_pct
    medium_thr      = total * medium_pct
    very_common_thr = total * very_common_pct

    medium_mask = (class_counts >= rare_thr) & (class_counts < medium_thr)
    median_medium = (
        class_counts[medium_mask].median()
        if medium_mask.sum() > 0
        else class_counts.median()
    )
    reference = median_medium * 0.7

    weights = pd.Series(index=class_counts.index, dtype=float)
    for cls, count in class_counts.items():
        frac = count / total
        if frac < rare_pct:
            weights[cls] = min(reference / count, 10.0)
        elif frac < medium_pct:
            weights[cls] = max(reference / count, 0.5)
        elif frac < very_common_pct:
            weights[cls] = max(reference * 0.3 / count, 0.10)
        else:
            weights[cls] = max(reference * 0.1 / count, 0.05)
    return weights


def get_slide_mpp(slide) -> float:
    """
    Extract the MPP_X property from an OpenSlide handle.

    Falls back to ``0.4416`` (20× GS40/GS55 scan parameter) if the property
    is missing or cannot be parsed.
    """
    try:
        import openslide
        return float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.4416))
    except Exception:
        return 0.4416


def assign_cells_to_tiles(
    cells_xy: np.ndarray,
    tile_size: int,
    stride: int,
) -> Dict[Tuple[int, int], List[int]]:
    """
    Assign each cell centroid to the non-overlapping tile it falls inside.

    Parameters
    ----------
    cells_xy  : (N, 2) float array of (x, y) coordinates in slide pixels
    tile_size : tile width/height in pixels
    stride    : step between tile origins (non-overlapping → stride == tile_size)

    Returns
    -------
    dict mapping ``(tile_x, tile_y)`` → list of cell indices in that tile
    """
    tiles: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for idx, (x, y) in enumerate(cells_xy):
        tx = int(x // stride) * stride
        ty = int(y // stride) * stride
        if tx <= x < tx + tile_size and ty <= y < ty + tile_size:
            tiles[(tx, ty)].append(idx)
    return dict(tiles)


# 8 deterministic geometric augmentation variants (0 = identity)
_AUG_NAMES = [
    "identity", "hflip", "vflip", "rot90_ccw",
    "rot180",   "rot270_ccw", "hvflip", "hflip+rot90_ccw",
]


def augment_image(img: np.ndarray, aug_id: int) -> np.ndarray:
    """
    Apply one of 8 deterministic geometric augmentations to an H×W×3 array.

    ``aug_id`` must be in ``[0, 7]``; values outside this range return the
    original unchanged.
    """
    if aug_id == 0: return img
    if aug_id == 1: return img[:, ::-1, :].copy()                    # hflip
    if aug_id == 2: return img[::-1, :, :].copy()                    # vflip
    if aug_id == 3: return np.rot90(img, k=1).copy()                 # 90° CCW
    if aug_id == 4: return np.rot90(img, k=2).copy()                 # 180°
    if aug_id == 5: return np.rot90(img, k=3).copy()                 # 270° CCW
    if aug_id == 6: return img[::-1, ::-1, :].copy()                 # hvflip
    if aug_id == 7: return np.rot90(img[:, ::-1, :], k=1).copy()     # hflip + 90° CCW
    return img


def augment_coords(
    x: int,
    y: int,
    aug_id: int,
    tile_size: int,
) -> Tuple[int, int]:
    """
    Transform a cell's local (x, y) coordinate to match ``augment_image(aug_id)``.

    ``tile_size`` is the side length of the (square) tile.
    """
    S = tile_size - 1
    if aug_id == 0: return x, y
    if aug_id == 1: return S - x, y
    if aug_id == 2: return x,     S - y
    if aug_id == 3: return y,     S - x
    if aug_id == 4: return S - x, S - y
    if aug_id == 5: return S - y, x
    if aug_id == 6: return S - x, S - y
    if aug_id == 7: return y,     x
    return x, y


def augment_polygon_ring(
    ring: List[List[float]],
    aug_id: int,
    tile_size: int,
) -> List[List[float]]:
    """
    Apply the same geometric transform as ``augment_image(aug_id)`` to every
    vertex of a polygon ring (float coordinates in tile space).

    The ring is expected to be a list of ``[x, y]`` pairs.  The closing
    duplicate point (if present) is preserved after transformation.

    Matches ``augment_coords`` exactly, but operates on float coordinates so
    that sub-pixel polygon precision is retained.
    """
    S = float(tile_size - 1)
    transforms = {
        0: lambda x, y: (x, y),
        1: lambda x, y: (S - x, y),
        2: lambda x, y: (x, S - y),
        3: lambda x, y: (y, S - x),
        4: lambda x, y: (S - x, S - y),
        5: lambda x, y: (S - y, x),
        6: lambda x, y: (S - x, S - y),
        7: lambda x, y: (y, x),
    }
    fn = transforms.get(aug_id, lambda x, y: (x, y))
    return [[fn(pt[0], pt[1])[0], fn(pt[0], pt[1])[1]] for pt in ring]


def clip_features_to_tile(
    features: list,
    tile_x: int,
    tile_y: int,
    tile_size: int,
    cell_indices: List[int],
) -> list:
    """
    Extract and localize GeoJSON features for a single tile.

    Given the full list of valid slide features (aligned with ``cells_xy``)
    and the indices of cells that fall inside this tile, returns a new list
    of deep-copied feature dicts with polygon coordinates localized to the
    tile origin (i.e. each vertex has ``tile_x`` / ``tile_y`` subtracted so
    coordinates run from 0 to ``tile_size``).

    Parameters
    ----------
    features    : list of GeoJSON feature dicts for the slide (same ordering
                  as ``cells_xy`` / ``cells_labels`` in ``slide_data``)
    tile_x      : left edge of the tile in slide pixel space
    tile_y      : top edge of the tile in slide pixel space
    tile_size   : tile width/height in pixels (used for clamping, not clipping)
    cell_indices: indices into ``features`` whose centroids fall in this tile

    Returns
    -------
    list of feature dicts with localized coordinates (deep-copied)
    """
    import copy
    clipped = []
    for ci in cell_indices:
        feat = copy.deepcopy(features[ci])
        geom = feat.get("geometry", {})
        if geom.get("type") != "Polygon":
            continue
        new_rings = []
        for ring in geom.get("coordinates", []):
            new_rings.append([[pt[0] - tile_x, pt[1] - tile_y] for pt in ring])
        feat["geometry"]["coordinates"] = new_rings
        clipped.append(feat)
    return clipped


def rasterize_tile_features(
    features: list,
    tile_size: int,
) -> Tuple[np.ndarray, Dict[str, str]]:
    """
    Rasterize a list of localized GeoJSON polygon features into a uint16
    instance mask and build the matching inst2class mapping.

    Each feature is drawn with a unique 1-based instance ID.  The class name
    is read from ``properties.classification.name``.

    Parameters
    ----------
    features  : list of GeoJSON feature dicts with coordinates already
                localized to tile space (output of ``clip_features_to_tile``)
    tile_size : side length of the square output mask (pixels)

    Returns
    -------
    mask       : (tile_size, tile_size) uint16 array — 0 = background,
                 1…N = nucleus instance IDs
    inst2class : dict mapping instance ID string → tissue class name string,
                 e.g. ``{"1": "kidney", "2": "bone"}``
    """
    mask = np.zeros((tile_size, tile_size), dtype=np.uint16)
    inst2class: Dict[str, str] = {}

    for inst_id, feat in enumerate(features, start=1):
        geom = feat.get("geometry", {})
        if geom.get("type") != "Polygon":
            continue
        rings = geom.get("coordinates", [])
        if not rings:
            continue
        ring = rings[0]
        pts = np.array([[pt[0], pt[1]] for pt in ring], dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], color=inst_id)

        cls_name = (
            feat.get("properties", {})
                .get("classification", {})
                .get("name", "")
        )
        if cls_name:
            inst2class[str(inst_id)] = cls_name

    return mask, inst2class


def write_tile_geojson(features: list, out_path: "Path") -> None:
    """
    Write a list of GeoJSON feature dicts as a FeatureCollection to ``out_path``.

    Creates parent directories as needed.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fc = {"type": "FeatureCollection", "features": features}
    out_path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
