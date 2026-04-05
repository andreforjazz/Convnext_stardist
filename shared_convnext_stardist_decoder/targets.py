# from __future__ import annotations

# """
# StarDist-compatible training targets (NumPy).

# For faster offline pipelines, generate the same arrays with StarDist's C++ star_dist:
#     from stardist.geometry import star_dist
#     d = star_dist(lbl.astype(np.uint16), n_rays, mode="cpp")

# Ray marching fallback is adapted from StarDist (BSD-3-Clause) geom2d._py_star_dist.
# """

# import numpy as np
# from scipy.ndimage import distance_transform_edt, find_objects


# def edt_prob(lbl_img: np.ndarray) -> np.ndarray:
#     """Normalized EDT probability map (per instance), StarDist-style."""
#     lbl_img = np.asarray(lbl_img)
#     prob = np.zeros(lbl_img.shape, np.float32)
#     objects = find_objects(lbl_img)
#     for i, sl in enumerate(objects, 1):
#         if sl is None:
#             continue
#         mask = lbl_img[sl] == i
#         if not np.any(mask):
#             continue
#         dist = distance_transform_edt(~mask)
#         vals = dist[mask]
#         prob[sl][mask] = vals / (float(np.max(vals)) + 1e-10)
#     return prob


# def star_dist_py(lbl: np.ndarray, n_rays: int = 32) -> np.ndarray:
#     """
#     Label image (uint-like) -> (H, W, n_rays) float32 ray distances.
#     Background (0) rays are 0.
#     """
#     n_rays = int(n_rays)
#     if n_rays < 3:
#         raise ValueError("n_rays must be >= 3")

#     a = np.ascontiguousarray(lbl.astype(np.uint16, copy=False))
#     dst = np.empty(a.shape + (n_rays,), np.float32)
#     st_rays = np.float32((2 * np.pi) / n_rays)

#     for i in range(a.shape[0]):
#         for j in range(a.shape[1]):
#             value = int(a[i, j])
#             if value == 0:
#                 dst[i, j] = 0
#                 continue
#             for k in range(n_rays):
#                 phi = np.float32(k * st_rays)
#                 dy = np.cos(phi)
#                 dx = np.sin(phi)
#                 x, y = np.float32(0), np.float32(0)
#                 while True:
#                     x += dx
#                     y += dy
#                     ii = int(round(i + x))
#                     jj = int(round(j + y))
#                     if (
#                         ii < 0
#                         or ii >= a.shape[0]
#                         or jj < 0
#                         or jj >= a.shape[1]
#                         or value != int(a[ii, jj])
#                     ):
#                         t_corr = 1 - 0.5 / max(abs(float(dx)), abs(float(dy)), 1e-6)
#                         x -= t_corr * dx
#                         y -= t_corr * dy
#                         dist = float(np.sqrt(x**2 + y**2))
#                         dst[i, j, k] = dist
#                         break
#     return dst


# def build_class_target(
#     inst: np.ndarray, inst_to_class: dict[int, int], ignore_index: int = -100
# ) -> np.ndarray:
#     """Per-pixel class index for CE; background / unknown -> ignore_index."""
#     inst = np.asarray(inst)
#     tgt = np.full(inst.shape, ignore_index, dtype=np.int64)
#     for k, cls_idx in inst_to_class.items():
#         mask = inst == int(k)
#         tgt[mask] = int(cls_idx)
#     return tgt


# def assemble_targets(
#     lbl_instance: np.ndarray,
#     inst_to_class: dict[int, int] | None,
#     n_rays: int,
#     ignore_index: int = -100,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Returns:
#         prob:   (H, W) float32
#         dist:   (H, W, n_rays) float32
#         cls:    (H, W) int64 (ignore_index outside assigned nuclei)
#         fg_mask:(H, W) bool  (instance > 0)
#     """
#     lbl_instance = np.asarray(lbl_instance)
#     prob = edt_prob(lbl_instance)
#     try:
#         from stardist.geometry import star_dist
#         dist = star_dist(lbl_instance.astype(np.uint16, copy=False), n_rays, mode="cpp")
#     except ImportError:
#         dist = star_dist_py(lbl_instance, n_rays=n_rays)
#     fg = lbl_instance > 0
#     if inst_to_class:
#         cls = build_class_target(lbl_instance, inst_to_class, ignore_index=ignore_index)
#         cls[~fg] = ignore_index
#     else:
#         cls = np.full(lbl_instance.shape, ignore_index, dtype=np.int64)
#     return prob, dist, cls, fg


from __future__ import annotations

"""
StarDist-compatible training targets (NumPy).

For faster offline pipelines, generate the same arrays with StarDist's C++ star_dist:
    from stardist.geometry import star_dist
    d = star_dist(lbl.astype(np.uint16), n_rays, mode="cpp")

Ray marching fallback is adapted from StarDist (BSD-3-Clause) geom2d._py_star_dist.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, find_objects


def edt_prob(lbl_img: np.ndarray) -> np.ndarray:
    """Normalized EDT probability map (per instance), StarDist-style."""
    lbl_img = np.asarray(lbl_img)
    prob = np.zeros(lbl_img.shape, np.float32)
    objects = find_objects(lbl_img)
    for i, sl in enumerate(objects, 1):
        if sl is None:
            continue
        mask = lbl_img[sl] == i
        if not np.any(mask):
            continue
        dist = distance_transform_edt(mask)  # FIXED: removed ~ operator
        vals = dist[mask]
        prob[sl][mask] = vals / (float(np.max(vals)) + 1e-10)
    return prob


def star_dist_py(lbl: np.ndarray, n_rays: int = 32) -> np.ndarray:
    """
    Label image (uint-like) -> (H, W, n_rays) float32 ray distances.
    Background (0) rays are 0.
    """
    n_rays = int(n_rays)
    if n_rays < 3:
        raise ValueError("n_rays must be >= 3")

    a = np.ascontiguousarray(lbl.astype(np.uint16, copy=False))
    dst = np.empty(a.shape + (n_rays,), np.float32)
    st_rays = np.float32((2 * np.pi) / n_rays)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            value = int(a[i, j])
            if value == 0:
                dst[i, j] = 0
                continue
            for k in range(n_rays):
                phi = np.float32(k * st_rays)
                dy = np.cos(phi)
                dx = np.sin(phi)
                x, y = np.float32(0), np.float32(0)
                while True:
                    x += dx
                    y += dy
                    ii = int(round(i + x))
                    jj = int(round(j + y))
                    if (
                        ii < 0
                        or ii >= a.shape[0]
                        or jj < 0
                        or jj >= a.shape[1]
                        or value != int(a[ii, jj])
                    ):
                        t_corr = 1 - 0.5 / max(abs(float(dx)), abs(float(dy)), 1e-6)
                        x -= t_corr * dx
                        y -= t_corr * dy
                        dist = float(np.sqrt(x**2 + y**2))
                        dst[i, j, k] = dist
                        break
    return dst


def build_class_target(
    inst: np.ndarray, inst_to_class: dict[int, int], ignore_index: int = -100
) -> np.ndarray:
    """Per-pixel class index for CE; background / unknown -> ignore_index."""
    inst = np.asarray(inst)
    tgt = np.full(inst.shape, ignore_index, dtype=np.int64)
    for k, cls_idx in inst_to_class.items():
        mask = inst == int(k)
        tgt[mask] = int(cls_idx)
    return tgt


def assemble_targets(
    lbl_instance: np.ndarray,
    inst_to_class: dict[int, int] | None,
    n_rays: int,
    ignore_index: int = -100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        prob:   (H, W) float32
        dist:   (H, W, n_rays) float32
        cls:    (H, W) int64 (ignore_index outside assigned nuclei)
        fg_mask:(H, W) bool  (instance > 0)
    """
    lbl_instance = np.asarray(lbl_instance)
    prob = edt_prob(lbl_instance)
    try:
        from stardist.geometry import star_dist
        dist = star_dist(lbl_instance.astype(np.uint16, copy=False), n_rays, mode="cpp")
    except ImportError:
        dist = star_dist_py(lbl_instance, n_rays=n_rays)
    fg = lbl_instance > 0
    if inst_to_class:
        cls = build_class_target(lbl_instance, inst_to_class, ignore_index=ignore_index)
        cls[~fg] = ignore_index
    else:
        cls = np.full(lbl_instance.shape, ignore_index, dtype=np.int64)
    return prob, dist, cls, fg