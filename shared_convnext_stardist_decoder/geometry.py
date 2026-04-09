"""StarDist-style geometry: rays -> polygons, simple peak suppression."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter


def ray_angles(n_rays: int) -> np.ndarray:
    return np.linspace(0, 2 * np.pi, n_rays, endpoint=False)


def dist_to_coord(dist: np.ndarray, points: np.ndarray, scale_dist=(1.0, 1.0)) -> np.ndarray:
    """
    dist:   (n_poly, n_rays)
    points: (n_poly, 2)  row, col
    returns (n_poly, 2, n_rays) -> [row, col], vertex k
    """
    dist = np.asarray(dist, dtype=np.float32)
    points = np.asarray(points, dtype=np.float32)
    n_rays = dist.shape[1]
    phis = ray_angles(n_rays).astype(np.float32)
    coord = (dist[:, np.newaxis, :] * np.array([np.sin(phis), np.cos(phis)], dtype=np.float32))
    coord *= np.asarray(scale_dist, dtype=np.float32).reshape(1, 2, 1)
    coord += points[..., np.newaxis]
    return coord


def local_peaks(prob: np.ndarray, min_distance: int, thresh: float) -> np.ndarray:
    """Returns (n, 2) int array of (row, col) peaks."""
    prob = np.asarray(prob, dtype=np.float32)
    size = int(max(1, min_distance))
    mx = maximum_filter(prob, size=size, mode="nearest")
    mask = (prob >= mx) & (prob > thresh)
    yy, xx = np.nonzero(mask)
    return np.stack([yy, xx], axis=1)


def dist_at_points(dist_map: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Sample (H,W,R) dist_map at integer points -> (n, R)."""
    out = []
    for (y, x) in points:
        out.append(dist_map[int(y), int(x)])
    return np.asarray(out, dtype=np.float32)


def polygon_ring_rowcol(coord_rc: np.ndarray) -> np.ndarray:
    """
    coord_rc: (2, n_rays) row, col vertices
    returns closed ring (n_rays+1, 2) as [x, y] for GeoJSON (col=x, row=y).
    """
    row = coord_rc[0]
    col = coord_rc[1]
    ring = np.stack([col, row], axis=1)
    return np.vstack([ring, ring[0:1]])


def dedupe_nucleus_features_by_centroid(
    features: list[dict],
    *,
    min_dist_px: float,
) -> list[dict]:
    """
    Greedy non-maximum suppression on polygon centroids in slide (pixel) space.

    Use after multi-offset / tiled inference so the same nucleus is not counted many
    times. Keeps the higher ``prob_peak`` when two centroids fall within
    ``min_dist_px`` (Euclidean distance on GeoJSON x,y). Typical value is a few
    pixels below the expected nuclear diameter at the inference level.
    """
    if len(features) <= 1:
        return list(features)

    min_d = float(min_dist_px)
    if min_d <= 0:
        return list(features)
    min_d2 = min_d * min_d
    cell = max(min_d, 1e-3)

    n = len(features)
    centroids = np.empty((n, 2), dtype=np.float64)
    scores = np.empty(n, dtype=np.float64)
    for i, feat in enumerate(features):
        ring = np.asarray(feat["geometry"]["coordinates"][0], dtype=np.float64)
        centroids[i] = ring[:-1].mean(axis=0)
        scores[i] = float(feat["properties"].get("prob_peak", 0.0))

    order = np.argsort(-scores)
    buckets: dict[tuple[int, int], list[int]] = {}
    kept_orig_idx: list[int] = []

    for i in order:
        i = int(i)
        cx, cy = centroids[i, 0], centroids[i, 1]
        ci = int(np.floor(cx / cell))
        cj = int(np.floor(cy / cell))
        conflict = False
        for di in (-2, -1, 0, 1, 2):
            for dj in (-2, -1, 0, 1, 2):
                for j in buckets.get((ci + di, cj + dj), ()):
                    dx = centroids[i, 0] - centroids[j, 0]
                    dy = centroids[i, 1] - centroids[j, 1]
                    if dx * dx + dy * dy < min_d2:
                        conflict = True
                        break
                if conflict:
                    break
            if conflict:
                break
        if conflict:
            continue
        kept_orig_idx.append(i)
        key = (ci, cj)
        buckets.setdefault(key, []).append(i)

    kept_orig_idx.sort()
    return [features[i] for i in kept_orig_idx]


def dedupe_nucleus_features_by_polygon_overlap(
    features: list[dict],
    *,
    min_overlap_ratio: float = 0.5,
    grid_cell_px: float = 32.0,
) -> list[dict]:
    """
    Second-stage deduplication: drop polygons that overlap an already-kept polygon
    "too much" (same nucleus predicted twice with centroids too far apart for
    :func:`dedupe_nucleus_features_by_centroid`).

    Greedy by descending ``prob_peak``. For each candidate, compares to kept
    polygons whose bounding boxes may intersect using a spatial grid. Keeps the
    candidate only if for every kept polygon *j*,

        intersection_area / min(area_i, area_j) < min_overlap_ratio.

    Requires **shapely** (``pip install shapely``).
    """
    try:
        from shapely.geometry import Polygon
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "dedupe_nucleus_features_by_polygon_overlap needs shapely>=2. "
            "Install shapely or disable polygon overlap deduplication."
        ) from exc

    if len(features) <= 1:
        return list(features)

    thr = float(min_overlap_ratio)
    if not (0.0 < thr < 1.0):
        return list(features)
    cell = float(grid_cell_px)
    if cell <= 0:
        cell = 32.0

    def ring_to_poly(feat: dict) -> Polygon | None:
        ring = np.asarray(feat["geometry"]["coordinates"][0], dtype=np.float64)
        if ring.shape[0] >= 2 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        if ring.shape[0] < 3:
            return None
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 1e-12:
            return None
        return poly

    n = len(features)
    scores = np.array(
        [float(f["properties"].get("prob_peak", 0.0)) for f in features],
        dtype=np.float64,
    )
    order = np.argsort(-scores)

    buckets: dict[tuple[int, int], list[int]] = {}
    kept_poly: dict[int, Polygon] = {}
    kept_list: list[int] = []

    def cells_touching_bbox(b: tuple[float, float, float, float]) -> list[tuple[int, int]]:
        minx, miny, maxx, maxy = b
        ix0 = int(np.floor(minx / cell))
        iy0 = int(np.floor(miny / cell))
        ix1 = int(np.floor(maxx / cell))
        iy1 = int(np.floor(maxy / cell))
        out: list[tuple[int, int]] = []
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                out.append((ix, iy))
        return out

    for ii in order:
        i = int(ii)
        poly_i = ring_to_poly(features[i])
        if poly_i is None:
            continue
        a_i = poly_i.area
        bi = poly_i.bounds
        conflict = False
        seen_j: set[int] = set()
        for key in cells_touching_bbox(bi):
            for j in buckets.get(key, ()):
                if j in seen_j:
                    continue
                seen_j.add(j)
                poly_j = kept_poly[j]
                bj = poly_j.bounds
                if bi[2] < bj[0] or bj[2] < bi[0] or bi[3] < bj[1] or bj[3] < bi[1]:
                    continue
                inter = poly_i.intersection(poly_j).area
                denom = min(a_i, poly_j.area)
                if denom <= 0:
                    continue
                if inter / denom >= thr:
                    conflict = True
                    break
            if conflict:
                break
        if conflict:
            continue
        kept_list.append(i)
        kept_poly[i] = poly_i
        for key in cells_touching_bbox(bi):
            buckets.setdefault(key, []).append(i)

    kept_list.sort()
    return [features[i] for i in kept_list]


def vote_class(
    cls_logits: np.ndarray, coord_rc: np.ndarray, image_shape: tuple[int, int]
) -> tuple[int, np.ndarray]:
    """
    cls_logits: (C, H, W) logits; class vote = softmax(mean pooled logits inside polygon).
    Returns (class_id, probs) with probs shape (C,).
    """
    from skimage.draw import polygon

    h, w = image_shape
    rr, cc = polygon(coord_rc[0], coord_rc[1], (h, w))
    c = cls_logits.shape[0]
    if len(rr) == 0:
        return -1, np.zeros(c, dtype=np.float32)
    patches = cls_logits[:, rr, cc]
    v = np.mean(patches, axis=1).astype(np.float64)
    v = v - np.max(v)
    ex = np.exp(v)
    probs = (ex / (np.sum(ex) + 1e-10)).astype(np.float32)
    pred = int(np.argmax(probs))
    return pred, probs
