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


def dist_at_points_bilinear(dist_map: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Sample (H,W,R) dist_map with bilinear interpolation at (row, col); points may be sub-pixel."""
    from scipy.ndimage import map_coordinates

    dist_map = np.asarray(dist_map, dtype=np.float32)
    points = np.asarray(points, dtype=np.float64)
    if dist_map.ndim != 3:
        raise ValueError("dist_map must be (H, W, R)")
    H, W, R = dist_map.shape
    n = len(points)
    if n == 0:
        return np.zeros((0, R), dtype=np.float32)
    out = np.empty((n, R), dtype=np.float32)
    coords = np.stack([points[:, 0], points[:, 1]], axis=0)
    for k in range(R):
        out[:, k] = map_coordinates(
            dist_map[:, :, k],
            coords,
            order=1,
            mode="nearest",
            prefilter=False,
        ).astype(np.float32)
    return out


def refine_peaks_local_com(
    peaks_rc: np.ndarray,
    prob_hw: np.ndarray,
    radius_px: int,
) -> np.ndarray:
    """
    Nudge each peak toward the intensity-weighted centroid of ``prob`` in a square window.
    Improves star-polygon fit when the discrete peak sits off the true nuclear center
    (common on elongated or noisy blobs). Returns float (row, col) per peak.
    """
    peaks_rc = np.asarray(peaks_rc, dtype=np.float64)
    prob_hw = np.asarray(prob_hw, dtype=np.float32)
    if len(peaks_rc) == 0:
        return peaks_rc
    H, W = prob_hw.shape
    rad = int(max(0, radius_px))
    out = np.empty_like(peaks_rc, dtype=np.float64)
    for i, (r, c) in enumerate(peaks_rc):
        r0 = int(max(0, np.floor(r - rad)))
        r1 = int(min(H, np.ceil(r + rad + 1)))
        c0 = int(max(0, np.floor(c - rad)))
        c1 = int(min(W, np.ceil(c + rad + 1)))
        patch = prob_hw[r0:r1, c0:c1]
        s = float(patch.sum())
        if s < 1e-12:
            out[i, 0], out[i, 1] = float(r), float(c)
            continue
        yy, xx = np.mgrid[r0:r1, c0:c1]
        out[i, 0] = float((yy * patch).sum() / s)
        out[i, 1] = float((xx * patch).sum() / s)
    return out


def dists_and_coords_from_peaks(
    dist_map_nrays_hw: np.ndarray,
    peaks_rc: np.ndarray,
    prob_hw: np.ndarray,
    *,
    refine_local_com: bool = False,
    refine_radius_px: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build ray distances and vertex coords for all peaks in one tile.

    ``dist_map_nrays_hw``: (n_rays, H, W) as returned by the model.
    If ``refine_local_com``, shifts each peak to local prob COM then samples rays with bilinear
    interpolation (recommended together).
    """
    dist_hw_r = np.transpose(np.asarray(dist_map_nrays_hw, dtype=np.float32), (1, 2, 0))
    peaks_rc = np.asarray(peaks_rc)
    if len(peaks_rc) == 0:
        n_rays = dist_hw_r.shape[2]
        return np.zeros((0, n_rays), dtype=np.float32), np.zeros((0, 2, n_rays), dtype=np.float32)
    if refine_local_com:
        pts = refine_peaks_local_com(peaks_rc, prob_hw, refine_radius_px)
        dists = dist_at_points_bilinear(dist_hw_r, pts)
        coords = dist_to_coord(dists, pts.astype(np.float32))
    else:
        dists = dist_at_points(dist_hw_r, peaks_rc)
        coords = dist_to_coord(dists, peaks_rc.astype(np.float32))
    return dists, coords


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
    Greedy NMS on polygon centroids (slide pixel space), O(n log n) via cKDTree.

    For each nucleus processed in descending ``prob_peak`` order: keep it if no
    already-kept nucleus lies within ``min_dist_px`` (Euclidean on GeoJSON x,y);
    otherwise discard. Falls back to a spatial-hash implementation when scipy is
    not available so the function always works.
    """
    if len(features) <= 1:
        return list(features)

    min_d = float(min_dist_px)
    if min_d <= 0:
        return list(features)

    n = len(features)
    centroids = np.empty((n, 2), dtype=np.float64)
    scores = np.empty(n, dtype=np.float64)
    for i, feat in enumerate(features):
        ring = np.asarray(feat["geometry"]["coordinates"][0], dtype=np.float64)
        centroids[i] = ring[:-1].mean(axis=0)
        scores[i] = float(feat["properties"].get("prob_peak", 0.0))

    try:
        from scipy.spatial import cKDTree  # O(n log n) — fast for millions of nuclei

        order = np.argsort(-scores)
        # Pre-compute all within-radius neighbor lists in one C call.
        tree = cKDTree(centroids)
        neighbors_all = tree.query_ball_point(centroids, r=min_d, workers=-1)

        state = np.zeros(n, dtype=np.int8)  # 0=unseen  1=kept  -1=removed
        for idx in order:
            idx = int(idx)
            if state[idx] == -1:
                continue
            state[idx] = 1  # keep
            for nb in neighbors_all[idx]:
                if state[nb] == 0:
                    state[nb] = -1  # mark neighbors as removed

        kept_orig_idx = sorted(int(i) for i in np.where(state == 1)[0])

    except ImportError:
        # Fallback: spatial-hash grid (pure numpy, O(n) amortized)
        min_d2 = min_d * min_d
        cell = max(min_d, 1e-3)
        order = np.argsort(-scores)
        buckets: dict[tuple[int, int], list[int]] = {}
        kept_orig_idx = []
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
            buckets.setdefault((ci, cj), []).append(i)
        kept_orig_idx.sort()

    return [features[i] for i in kept_orig_idx]


def dedupe_nucleus_features_by_polygon_overlap(
    features: list[dict],
    *,
    min_overlap_ratio: float = 0.5,
    min_iou: float | None = None,
    grid_cell_px: float = 32.0,
) -> list[dict]:
    """
    Second-stage deduplication: drop polygons that overlap an already-kept polygon
    "too much" (same nucleus predicted twice with centroids too far apart for
    :func:`dedupe_nucleus_features_by_centroid`).

    Greedy by descending ``prob_peak``. For each candidate, compares to kept
    polygons whose bounding boxes may intersect using a spatial grid. Keeps the
    candidate only if for every kept polygon *j*,

        intersection_area / min(area_i, area_j) < min_overlap_ratio

    and, if ``min_iou`` is not None,

        intersection_area / union_area < min_iou.

    IoU catches elongated / dumbbell nuclei where two peaks sit far apart but the
    polygons still overlap substantially (centroid NMS alone misses those).

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
    iou_thr: float | None
    if min_iou is None:
        iou_thr = None
    else:
        iou_thr = float(min_iou)
        if not (0.0 < iou_thr < 1.0):
            iou_thr = None
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
                if iou_thr is not None:
                    union = a_i + poly_j.area - inter
                    if union > 0 and inter / union >= iou_thr:
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
