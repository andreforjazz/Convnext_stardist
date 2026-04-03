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
