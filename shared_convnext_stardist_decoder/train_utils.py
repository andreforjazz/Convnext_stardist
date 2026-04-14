"""
train_utils.py
==============
Helper functions for the ConvNeXt + StarDist multitask training pipeline.

Called from train_multitask_GS40_paths.ipynb.  Every public function has a
clear docstring so you can inspect its behaviour without reading the notebook.

Sections
--------
1. Data I/O          — read_stems, find_mask, load_mask
2. Rasterize         — rasterize_masks
3. inst2class        — build_all_inst2class  (KD-tree centroid matching)
4. Resolve           — resolve_class         (int or string → tissue name)
5. Diagnostics       — diagnose_match_quality, diagnose_class_distribution
6. Audit & filter    — audit_stems, filter_stems
7. Visualisation     — visualise_samples, diagnostic_centroid_viz
8. QuPath export     — export_qupath_geojsons
9. Config & training — build_training_config, write_config, run_training
"""

from __future__ import annotations

import json
import shutil
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile
import yaml
from PIL import Image
from scipy.spatial import cKDTree
from tqdm import tqdm

# matplotlib imported lazily inside visualisation functions so this module
# can be imported in headless environments without GUI errors.


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Data I/O
# ═══════════════════════════════════════════════════════════════════════════════

_HEADER_NAMES = {
    "stem", "stems", "image", "images", "file", "filename",
    "name", "tile", "tiles", "0", "1", "index", "path",
}


def read_stems(csv_path: Path) -> list[str]:
    """Read a list of tile stems from a CSV file (one stem per row, no header).

    Strips path components and extension so both ``tile_001`` and
    ``/path/to/tile_001.png`` are handled identically.
    """
    df  = pd.read_csv(csv_path, header=None)
    raw = df.iloc[:, 0].dropna().astype(str).tolist()
    out = [Path(s).stem for s in raw]
    out = [s for s in out if s and s.lower() not in _HEADER_NAMES and len(s) > 5]
    seen: set[str] = set()
    return [s for s in out if not (s in seen or seen.add(s))]  # type: ignore[func-returns-value]


def find_mask(stem: str, inst_labels_dir: Path) -> Path | None:
    """Return the path of the instance mask PNG/TIF for *stem*, or None."""
    for ext in (".png", ".tif", ".tiff"):
        p = inst_labels_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def load_mask(path: Path) -> np.ndarray | None:
    """Load a uint16 instance mask; return None on any read error.

    Returns an int32 2-D array (H, W) — background = 0, nuclei = 1…N.
    """
    try:
        if path.suffix.lower() in (".tif", ".tiff"):
            arr = tifffile.imread(path)
        else:
            arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None or arr.size == 0:
            return None
        if arr.ndim != 2:
            arr = arr.squeeze()
        return arr.astype(np.int32)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Rasterize
# ═══════════════════════════════════════════════════════════════════════════════

def rasterize_masks(
    all_stems: list[str],
    train_images_dir: Path,
    stardist_json_dir: Path,
    inst_labels_dir: Path,
) -> tuple[int, int, int]:
    """Rasterize StarDist JSON polygons → uint16 instance mask PNGs.

    Skips stems that already have a mask.  Returns (created, skipped, failed).
    """
    missing = [
        s for s in all_stems
        if find_mask(s, inst_labels_dir) is None
        and (stardist_json_dir / f"{s}.json").exists()
    ]
    print(f"Total stems: {len(all_stems)}  |  Existing masks: {len(all_stems) - len(missing)}  |  To rasterize: {len(missing)}")

    created = failed = skipped = 0
    for stem in tqdm(missing, desc="Rasterize masks"):
        img_path = next(
            (train_images_dir / f"{stem}{ext}" for ext in (".png", ".tif", ".jpg")
             if (train_images_dir / f"{stem}{ext}").exists()),
            None,
        )
        if img_path is None:
            skipped += 1
            continue
        try:
            with Image.open(img_path) as im:
                w, h = im.size
            mask  = np.zeros((h, w), dtype=np.uint16)
            polys = json.loads((stardist_json_dir / f"{stem}.json").read_bytes())
            for inst_id, obj in enumerate(polys, start=1):
                c = obj.get("contour", [])
                if not c:
                    continue
                pts = (
                    np.column_stack((c[0][1], c[0][0]))
                    .astype(np.int32)
                    .reshape(-1, 1, 2)
                )
                cv2.fillPoly(mask, [pts], color=inst_id)
            cv2.imwrite(str(inst_labels_dir / f"{stem}.png"), mask)
            created += 1
        except Exception as e:
            print(f"  FAIL {stem}: {e}")
            failed += 1

    print(f"Done — created: {created}  skipped: {skipped}  failed: {failed}")
    return created, skipped, failed


# ═══════════════════════════════════════════════════════════════════════════════
# 3. inst2class building
# ═══════════════════════════════════════════════════════════════════════════════

def _stardist_centroids(mask: np.ndarray) -> dict[int, tuple[float, float]]:
    """Compute (x, y) centroid for each non-zero instance id in the mask."""
    h, w = mask.shape
    flat = mask.ravel()
    ys, xs = np.divmod(np.arange(h * w), w)
    return {
        int(iid): (float(xs[flat == iid].mean()), float(ys[flat == iid].mean()))
        for iid in np.unique(flat)
        if iid != 0
    }


def build_inst2class(
    stem: str,
    csv_dir: Path,
    inst_labels_dir: Path,
    class_names: list[str],
    class_to_idx: dict[str, int],
    max_dist_px: float,
) -> dict[str, str] | None:
    """Match CellViT CSV centroids to nearest StarDist instances via KD-tree.

    Returns ``{inst_id_str: tissue_name_str}`` or None if matching fails.
    Tissue names (not integers) are stored so the sidecar is independent of
    CLASS_NAMES ordering.
    """
    csv_path  = csv_dir / f"{stem}.csv"
    mask_path = find_mask(stem, inst_labels_dir)
    if not csv_path.exists() or mask_path is None:
        return None

    try:
        df = pd.read_csv(csv_path, header=None).iloc[:, :3]
        df.columns = ["x", "y", "class"]
    except Exception:
        return None

    mask = load_mask(mask_path)
    if mask is None:
        return None

    sd_c = _stardist_centroids(mask)
    if not sd_c:
        return None

    inst_ids = list(sd_c.keys())
    tree     = cKDTree(np.array([sd_c[i] for i in inst_ids], dtype=np.float64))

    out: dict[str, str] = {}
    for _, row in df.iterrows():
        try:
            cx, cy    = float(row["x"]), float(row["y"])
            class_raw = str(row["class"]).strip().lower()
        except (ValueError, TypeError):
            continue
        # Resolve integer (legacy alphabetical) or string name
        try:
            cls_idx = int(float(class_raw))
            if not (0 <= cls_idx < len(class_names)):
                continue
        except ValueError:
            cls_idx = class_to_idx.get(class_raw)
            if cls_idx is None:
                continue
        dist, idx = tree.query([cx, cy])
        if dist > max_dist_px:
            continue
        matched = str(inst_ids[idx])
        if matched not in out:
            out[matched] = class_names[cls_idx]   # store tissue NAME string
    return out if out else None


def build_all_inst2class(
    all_stems: list[str],
    train_csv_dir: Path,
    val_csv_dir: Path,
    inst_labels_dir: Path,
    class_names: list[str],
    class_to_idx: dict[str, int],
    max_dist_px: float,
    force_rebuild: bool,
) -> tuple[int, int, int]:
    """Build inst2class JSON sidecars for all stems.

    For each stem tries train_csv_dir first, then val_csv_dir.
    Returns (created, skipped, failed).
    """
    created = skipped = failed = 0
    for stem in tqdm(all_stems, desc="Build inst2class"):
        out_path = inst_labels_dir / f"{stem}_inst2class.json"
        if out_path.exists() and not force_rebuild:
            skipped += 1
            continue
        mapping = (
            build_inst2class(stem, train_csv_dir, inst_labels_dir, class_names, class_to_idx, max_dist_px)
            or build_inst2class(stem, val_csv_dir,   inst_labels_dir, class_names, class_to_idx, max_dist_px)
        )
        if mapping is None:
            failed += 1
            continue
        out_path.write_text(json.dumps(mapping), encoding="utf-8")
        created += 1

    # Self-documenting metadata sidecar
    (inst_labels_dir / "inst2class_metadata.json").write_text(
        json.dumps({
            "class_names":      class_names,
            "value_format":     "tissue_name_string",
            "max_match_dist_px": max_dist_px,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"inst2class — created: {created}  skipped: {skipped}  failed: {failed}")
    return created, skipped, failed


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Resolve class value
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_class(val: object, class_to_idx: dict[str, int]) -> str:
    """Convert an inst2class value to a tissue name string.

    Handles both the new format (``"bone"``) and legacy integer format (``"11"``).
    Legacy integers are treated as alphabetically-sorted class indices.
    Returns ``"unclassified"`` for any unresolvable value.
    """
    if val is None:
        return "unclassified"
    try:
        idx  = int(val)
        alpha = sorted(class_to_idx.keys())
        return alpha[idx] if 0 <= idx < len(alpha) else "unknown"
    except (ValueError, TypeError):
        name = str(val).strip().lower()
        return name if name in class_to_idx else "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def diagnose_match_quality(
    all_stems: list[str],
    inst_labels_dir: Path,
    n_sample: int = 500,
) -> dict:
    """Report what fraction of StarDist nuclei received a class label.

    Samples up to *n_sample* tiles.  Prints a summary and returns a dict with
    ``match_rate``, ``total_sd``, ``total_matched``, ``tiles_zero``.
    """
    sample    = random.sample(all_stems, min(n_sample, len(all_stems)))
    total_sd  = total_matched = tiles_zero = 0

    for stem in tqdm(sample, desc="Match quality", leave=False):
        mask_path = find_mask(stem, inst_labels_dir)
        if mask_path is None:
            continue
        mask = load_mask(mask_path)
        if mask is None:
            continue
        total_sd += len(np.unique(mask)) - 1   # exclude background
        jp = inst_labels_dir / f"{stem}_inst2class.json"
        if jp.exists():
            try:
                d = json.loads(jp.read_bytes().decode("utf-8-sig"))
                total_matched += len(d)
                if not d:
                    tiles_zero += 1
            except Exception:
                tiles_zero += 1
        else:
            tiles_zero += 1

    rate = 100.0 * total_matched / max(total_sd, 1)
    print(f"\nMatch quality ({len(sample)} tiles sampled):")
    print(f"  Nuclei matched : {total_matched:,} / {total_sd:,}  ({rate:.1f}%)")
    print(f"  Tiles zero match: {tiles_zero}")
    if rate < 50:
        print("  WARNING: <50% — consider increasing MAX_MATCH_DIST_PX in Cell 2.")
    elif rate >= 70:
        print("  Match rate looks healthy.")
    return dict(match_rate=rate, total_sd=total_sd,
                total_matched=total_matched, tiles_zero=tiles_zero)


def diagnose_class_distribution(
    inst_labels_dir: Path,
    class_names: list[str],
    class_to_idx: dict[str, int],
    n_sample: int = 2000,
) -> dict:
    """Report global class distribution and flag suspiciously single-class tiles.

    Samples up to *n_sample* inst2class JSON files.  Returns a dict with
    ``global_counts`` (Counter), ``single_tiles`` (Counter), ``missing_classes``.
    """
    all_json  = [f for f in inst_labels_dir.glob("*_inst2class.json")
                 if f.name != "inst2class_metadata.json"]
    samp      = random.sample(all_json, min(n_sample, len(all_json)))
    global_c  = Counter()
    single_t  = Counter()
    empty_t   = 0

    for jf in tqdm(samp, desc="Class distribution", leave=False):
        try:
            d = json.loads(jf.read_bytes().decode("utf-8-sig"))
        except Exception:
            continue
        if not d:
            empty_t += 1
            continue
        names = [resolve_class(v, class_to_idx) for v in d.values()]
        global_c.update(names)
        uniq = set(names)
        if len(uniq) == 1:
            single_t[list(uniq)[0]] += 1

    total_n = sum(global_c.values())
    print(f"\nClass distribution ({len(samp)} files sampled):")
    for name, cnt in sorted(global_c.items(), key=lambda x: -x[1]):
        bar = "|" * int(30 * cnt / max(total_n, 1))
        print(f"  {name:14s}  {cnt:7,}  {100*cnt/max(total_n,1):5.1f}%  {bar}")

    missing = [n for n in class_names if n not in global_c]
    if missing:
        print(f"\n  WARNING: zero labeled nuclei for: {missing}")
    else:
        print(f"\n  All {len(class_names)} classes present.")

    if single_t:
        print(f"\nSingle-class tiles (flag if unexpected):")
        for t, n in sorted(single_t.items(), key=lambda x: -x[1]):
            flag = "  *** CHECK ***" if t == "bone" and n / len(samp) > 0.05 else ""
            print(f"  {t:14s}  {n:5d}  ({100*n/len(samp):.1f}%){flag}")

    return dict(global_counts=global_c, single_tiles=single_t,
                missing_classes=missing)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Audit & filter
# ═══════════════════════════════════════════════════════════════════════════════

def audit_stems(
    stems: list[str],
    label: str,
    inst_labels_dir: Path,
    class_names: list[str],
) -> dict:
    """Scan masks and inst2class sidecars for corruption or unknown class names.

    Returns a dict with coverage stats and lists of bad stems.  Call after
    rasterize + build_all_inst2class to decide whether the data is ready.

    Keys: ``mask_cov``, ``json_cov``, ``bad_masks``, ``bad_jsons``, ``foreign``.
    """
    known = set(class_names) | {str(i) for i in range(len(class_names))}
    bad_masks: list[str] = []
    bad_jsons: list[str] = []
    foreign:   dict[str, set] = {}
    has_mask = has_json = 0

    for stem in tqdm(stems, desc=f"Audit {label}", leave=False):
        mp = find_mask(stem, inst_labels_dir)
        if mp is None or load_mask(mp) is None:
            bad_masks.append(stem)
        else:
            has_mask += 1
        jp = inst_labels_dir / f"{stem}_inst2class.json"
        if not jp.exists():
            continue
        try:
            d = json.loads(jp.read_bytes().decode("utf-8-sig"))
        except Exception:
            bad_jsons.append(stem)
            continue
        if not d:
            continue
        unk = {str(v).strip().lower() for v in d.values()} - known
        if unk:
            foreign[stem] = unk
        has_json += 1

    n = len(stems)
    mask_cov = 100.0 * has_mask / max(n, 1)
    json_cov = 100.0 * has_json / max(n, 1)

    print(f"\n  {label} ({n} stems)")
    print(f"  Mask coverage       : {has_mask}/{n} ({mask_cov:.1f}%)")
    print(f"  inst2class coverage : {has_json}/{n} ({json_cov:.1f}%)")
    print(f"  Bad masks           : {len(bad_masks)}")
    print(f"  Bad JSONs           : {len(bad_jsons)}")
    if foreign:
        all_unk = set().union(*foreign.values())
        print(f"  Unknown class names : {sorted(all_unk)}  ← add to CLASS_NAMES or fix JSON")
    return dict(mask_cov=mask_cov, json_cov=json_cov,
                bad_masks=bad_masks, bad_jsons=bad_jsons, foreign=foreign)


def filter_stems(
    stems: list[str],
    audit_result: dict,
    inst_labels_dir: Path,
    require_json: bool = False,
) -> list[str]:
    """Remove stems with bad masks / corrupt JSONs from *stems*.

    If *require_json* is True, also remove stems with no inst2class sidecar
    (stricter — fewer samples, but every stem has classification labels).
    """
    bad = set(audit_result["bad_masks"] + audit_result["bad_jsons"])

    def _has_json(stem: str) -> bool:
        j = inst_labels_dir / f"{stem}_inst2class.json"
        if not j.exists():
            return False
        try:
            return bool(json.loads(j.read_bytes().decode("utf-8-sig")))
        except Exception:
            return False

    return [
        s for s in stems
        if s not in bad
        and find_mask(s, inst_labels_dir) is not None
        and (not require_json or _has_json(s))
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def visualise_samples(
    train_stems: list[str],
    val_stems: list[str],
    n: int,
    train_images_dir: Path,
    inst_labels_dir: Path,
    class_names: list[str],
    class_to_idx: dict[str, int],
    colors_viz: list[list[int]],
) -> None:
    """Show random tile samples: H&E | instance mask | class map.

    Plots *n* random tiles from each split.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    cmap = plt.cm.get_cmap("tab20", len(class_names))

    def _one(stem: str, axes) -> None:
        img_path = next(
            (train_images_dir / f"{stem}{ext}" for ext in (".png", ".jpg", ".tif")
             if (train_images_dir / f"{stem}{ext}").exists()),
            None,
        )
        mp = find_mask(stem, inst_labels_dir)
        if img_path is None or mp is None:
            for ax in axes:
                ax.set_visible(False)
            return
        img  = np.asarray(Image.open(img_path).convert("RGB"))
        mask = load_mask(mp)
        if mask is None:
            for ax in axes:
                ax.set_visible(False)
            return

        class_map = np.full(mask.shape, -1, dtype=np.int32)
        jp = inst_labels_dir / f"{stem}_inst2class.json"
        if jp.exists():
            try:
                d = json.loads(jp.read_bytes().decode("utf-8-sig"))
                for iid_str, val in d.items():
                    idx = class_to_idx.get(resolve_class(val, class_to_idx), -1)
                    if idx >= 0:
                        class_map[mask == int(iid_str)] = idx
            except Exception:
                pass

        axes[0].imshow(img)
        axes[0].set_title(stem[:30], fontsize=7)
        mask_rgb = plt.cm.nipy_spectral((mask % 32) / 32.0)[..., :3]
        mask_rgb[mask == 0] = 0
        axes[1].imshow(mask_rgb)
        axes[1].set_title(f"Mask n={mask.max()}", fontsize=7)
        cls_rgb = np.zeros((*mask.shape, 3))
        n_cls = 0
        for ci in range(len(class_names)):
            px = class_map == ci
            if px.any():
                cls_rgb[px] = cmap(ci)[:3]
                n_cls += 1
        axes[2].imshow(cls_rgb)
        axes[2].set_title(f"Classes: {n_cls}", fontsize=7)
        for ax in axes:
            ax.axis("off")

    for split_name, stems in [("TRAIN", train_stems), ("VAL", val_stems)]:
        samp = random.sample(stems, min(n, len(stems)))
        fig, axes = plt.subplots(len(samp), 3, figsize=(10, 3.2 * len(samp)))
        if len(samp) == 1:
            axes = [axes]
        fig.suptitle(f"{split_name} — H&E | instance mask | class map",
                     y=1.01, fontsize=10)
        for row, stem in zip(axes, samp):
            _one(stem, row)
        plt.tight_layout()
        plt.show()

    # Legend
    patches = [mpatches.Patch(color=cmap(i), label=n)
               for i, n in enumerate(class_names)]
    plt.figure(figsize=(12, 1))
    plt.legend(handles=patches, ncol=10, loc="center", fontsize=8, frameon=False)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def diagnostic_centroid_viz(
    train_stems: list[str],
    n: int,
    train_images_dir: Path,
    stardist_json_dir: Path,
    inst_labels_dir: Path,
    class_names: list[str],
    class_to_idx: dict[str, int],
    colors_viz: list[list[int]],
    out_geojson_dir: Path | None = None,
) -> None:
    """Overlay inst2class centroids on H&E tiles using the canonical colour palette.

    Picks *n* random tiles from *train_stems* that have a StarDist JSON.
    Optionally writes a Point GeoJSON to *out_geojson_dir* for QuPath.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    def _color_f(idx: int):
        if 0 <= idx < len(colors_viz):
            return tuple(c / 255.0 for c in colors_viz[idx])
        return (0.55, 0.55, 0.55)

    pool = [s for s in train_stems if (stardist_json_dir / f"{s}.json").exists()]
    if not pool:
        print("No stems with StarDist JSON.")
        return

    for stem in random.sample(pool, min(n, len(pool))):
        img_path = next(
            (train_images_dir / f"{stem}{ext}" for ext in (".png", ".jpg", ".tif")
             if (train_images_dir / f"{stem}{ext}").exists()),
            None,
        )
        if img_path is None:
            continue

        img   = np.asarray(Image.open(img_path).convert("RGB"))
        polys = json.loads((stardist_json_dir / f"{stem}.json").read_bytes())
        inst2cls: dict = {}
        jp = inst_labels_dir / f"{stem}_inst2class.json"
        if jp.exists():
            try:
                inst2cls = json.loads(jp.read_bytes().decode("utf-8-sig"))
            except Exception:
                pass

        xs, ys, cols, geo_rows = [], [], [], []
        for iid, obj in enumerate(polys, start=1):
            c = obj.get("contour", [])
            if not c:
                continue
            cx   = float(np.mean(c[0][1]))
            cy   = float(np.mean(c[0][0]))
            name = resolve_class(inst2cls.get(str(iid)), class_to_idx)
            idx  = class_to_idx.get(name, -1)
            xs.append(cx)
            ys.append(cy)
            cols.append(_color_f(idx))
            rgb_i = list(colors_viz[idx]) if 0 <= idx < len(colors_viz) else [180, 180, 180]
            geo_rows.append((cx, cy, class_names[idx] if idx >= 0 else "unclassified", rgb_i))

        if out_geojson_dir is not None:
            out_geojson_dir.mkdir(parents=True, exist_ok=True)
            _write_point_geojson(geo_rows, out_geojson_dir / f"{stem}_centroids.geojson")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(img)
        axes[0].set_title("H&E")
        axes[0].axis("off")
        axes[1].imshow(img)
        axes[1].scatter(xs, ys, c=cols, s=8, linewidths=0)
        axes[1].set_title(f"Centroids n={len(xs)} — check colours match tissue")
        axes[1].axis("off")
        leg = [
            Line2D([0], [0], marker="o", color="w", label=n,
                   markerfacecolor=_color_f(i), markersize=7)
            for i, n in enumerate(class_names)
        ]
        fig.legend(handles=leg, loc="lower center", ncol=7, fontsize=7,
                   bbox_to_anchor=(0.5, -0.04))
        plt.suptitle(stem, fontsize=9)
        plt.tight_layout()
        plt.show()


def _write_point_geojson(
    rows: list[tuple[float, float, str, list[int]]],
    out_path: Path,
) -> None:
    """Write a QuPath-compatible Point GeoJSON from centroid rows (cx, cy, name, rgb)."""
    feats = [
        {
            "type": "Feature",
            "id": "PathDetectionObject",
            "geometry": {"type": "Point", "coordinates": [cx, cy]},
            "properties": {
                "isLocked": False,
                "measurements": [],
                "classification": {"name": name, "color": rgb},
            },
        }
        for cx, cy, name, rgb in rows
    ]
    with out_path.open("w", encoding="utf-8") as f:
        f.write('{"type":"FeatureCollection","features":[\n')
        for i, feat in enumerate(feats):
            tail = ",\n" if i + 1 < len(feats) else "\n"
            f.write(json.dumps(feat, ensure_ascii=False) + tail)
        f.write("]}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. QuPath export
# ═══════════════════════════════════════════════════════════════════════════════

def export_qupath_geojsons(
    all_stems: list[str],
    every_n: int,
    stardist_json_dir: Path,
    inst_labels_dir: Path,
    train_images_dir: Path,
    out_dir: Path,
    class_names: list[str],
    class_to_idx: dict[str, int],
    colors_viz: list[list[int]],
) -> int:
    """Write QuPath 0.6 polygon GeoJSONs for every *every_n*-th tile.

    Each GeoJSON has nuclear polygons coloured by tissue class.
    Also copies the matching tile images so you can open them directly in QuPath.
    Returns the number of GeoJSON files written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = all_stems[::every_n]
    print(f"Exporting QuPath GeoJSONs: {len(stems)} tiles (every {every_n} of {len(all_stems)})")

    exported = 0
    for stem in tqdm(stems, desc="QuPath export"):
        if _export_one_tile(stem, stardist_json_dir, inst_labels_dir,
                            out_dir, class_names, class_to_idx, colors_viz):
            exported += 1

    # Copy matching images
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)
    for stem in stems:
        for ext in (".png", ".jpg", ".tif"):
            src = train_images_dir / f"{stem}{ext}"
            if src.exists():
                dst = img_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
                break

    # Summary
    cls_counts: Counter = Counter()
    for stem in stems:
        gj = out_dir / f"{stem}.geojson"
        if not gj.exists():
            continue
        try:
            for feat in json.loads(gj.read_text(encoding="utf-8")).get("features", []):
                cls_counts[feat["properties"]["classification"]["name"]] += 1
        except Exception:
            pass

    total = sum(cls_counts.values())
    print(f"Done: {exported} GeoJSONs | {total:,} nuclei | output: {out_dir}")
    for name, cnt in cls_counts.most_common():
        print(f"  {name:16s}  {cnt:6,}  ({100*cnt/max(total,1):.1f}%)")
    return exported


def _export_one_tile(
    stem: str,
    stardist_json_dir: Path,
    inst_labels_dir: Path,
    out_dir: Path,
    class_names: list[str],
    class_to_idx: dict[str, int],
    colors_viz: list[list[int]],
) -> bool:
    sd_json = stardist_json_dir / f"{stem}.json"
    if not sd_json.exists():
        return False
    polys = json.loads(sd_json.read_bytes())
    if not polys:
        return False

    inst2cls: dict = {}
    jp = inst_labels_dir / f"{stem}_inst2class.json"
    if jp.exists():
        try:
            inst2cls = json.loads(jp.read_bytes().decode("utf-8-sig"))
        except Exception:
            pass

    feats = []
    for iid_0, obj in enumerate(polys):
        c = obj.get("contour", [])
        if not c:
            continue
        ring = [[float(x), float(y)] for x, y in zip(c[0][1], c[0][0])]
        if len(ring) < 4:
            continue
        if ring[0] != ring[-1]:
            ring.append(ring[0])

        name = resolve_class(inst2cls.get(str(iid_0 + 1)), class_to_idx)
        idx  = class_to_idx.get(name, -1)
        color        = list(colors_viz[idx]) if 0 <= idx < len(colors_viz) else [180, 180, 180]
        cls_display  = class_names[idx] if idx >= 0 else "unclassified"

        feats.append({
            "type": "Feature",
            "id":   "PathDetectionObject",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {
                "isLocked":    False,
                "measurements": [],
                "classification": {"name": cls_display, "color": color},
            },
        })

    if not feats:
        return False
    out = out_dir / f"{stem}.geojson"
    with out.open("w", encoding="utf-8") as f:
        f.write('{"type":"FeatureCollection","features":[\n')
        for i, feat in enumerate(feats):
            f.write(json.dumps(feat, ensure_ascii=False)
                    + (",\n" if i + 1 < len(feats) else "\n"))
        f.write("]}\n")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Config & training
# ═══════════════════════════════════════════════════════════════════════════════

def build_training_config(
    experiment_name: str,
    train_images_dir: Path,
    inst_labels_dir: Path,
    ckpt_out: Path,
    train_stems: list[str],
    val_stems: list[str],
    class_names: list[str],
    *,
    epochs: int             = 50,
    batch_size: int         = 4,
    lr: float               = 1e-4,
    lr_min: float           = 1e-6,
    weight_decay: float     = 0.01,
    freeze_backbone_epochs: int = 5,
    loss_w_cls: float       = 2.0,
    loss_w_inst: float      = 0.5,
    loss_w_dist: float      = 0.05,
    cls_semantic_dim: int   = 128,
) -> dict:
    """Build the training config dictionary.

    All hyperparameters have sensible defaults; pass keyword args to override.
    Returns a plain dict suitable for ``yaml.dump``.
    """
    return {
        "experiment_name": experiment_name,
        "data": {
            "train_images_dir": str(train_images_dir),
            "train_labels_dir": str(inst_labels_dir),
            "val_images_dir":   str(train_images_dir),
            "val_labels_dir":   str(inst_labels_dir),
            "train_stems":      train_stems,
            "val_stems":        val_stems,
            "cache_to_ram":     False,
        },
        "model": {
            "backbone":          "facebook/convnextv2-tiny-22k-224",
            "pretrained":        True,
            "n_rays":            32,
            "class_names":       class_names,
            "num_classes":       len(class_names),
            "decoder_channels":  128,
            "cls_semantic_dim":  cls_semantic_dim,
        },
        "train": {
            "patch_size":              256,
            "batch_size":              batch_size,
            "num_workers":             4,
            "epochs":                  epochs,
            "lr":                      lr,
            "lr_min":                  lr_min,
            "weight_decay":            weight_decay,
            "amp":                     True,
            "freeze_backbone_epochs":  freeze_backbone_epochs,
            "loss_w_prob":             1.0,
            "loss_w_dist":             loss_w_dist,
            "loss_w_cls":              loss_w_cls,
            "loss_w_inst":             loss_w_inst,
            "class_weights":           "auto",
            "cls_balanced_sampler":    True,
            "cls_only":                False,
        },
        "checkpoint": {
            "out_dir":    str(ckpt_out),
            "save_every": 1,
        },
        "infer": {
            "prob_thresh": 0.45,
            "nms_dist":    3,
        },
    }


def write_config(cfg: dict, config_path: Path) -> None:
    """Write *cfg* as YAML to *config_path* and print a summary."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.dump(cfg, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    t = cfg["train"]
    print(f"Config written: {config_path}")
    print(f"  experiment : {cfg['experiment_name']}")
    print(f"  stems      : train {len(cfg['data']['train_stems'])} / val {len(cfg['data']['val_stems'])}")
    print(f"  classes    : {cfg['model']['class_names']}")
    print(f"  epochs     : {t['epochs']}  batch: {t['batch_size']}  lr: {t['lr']} → {t['lr_min']}")
    print(f"  freeze_bb  : {t['freeze_backbone_epochs']} epochs")
    print(f"  loss w_cls/w_inst: {t['loss_w_cls']}/{t['loss_w_inst']}")
    print(f"  ckpt dir   : {cfg['checkpoint']['out_dir']}")


def run_training(
    config_path: Path,
    repo_root: Path,
    resume_checkpoint: Path | None = None,
    resume_strict: bool = True,
) -> int:
    """Launch train_v2.py as a subprocess.

    Returns the process exit code (0 = success).
    Pass *resume_checkpoint* to fine-tune from an existing checkpoint.
    Set *resume_strict* = False for a V1 → V2 warm-start (partial weight load).
    """
    cmd = [
        sys.executable, "-m",
        "shared_convnext_stardist_decoder.train_v2",
        "--config", str(config_path),
    ]
    if resume_checkpoint is not None:
        cmd += ["--resume", str(resume_checkpoint),
                "--resume_strict", str(resume_strict).lower()]
        print(f"Resuming from : {resume_checkpoint}  (strict={resume_strict})")
    else:
        print("Fresh training run.")

    print(f"Command: {' '.join(str(c) for c in cmd)}")
    print("-" * 70)
    result = subprocess.run(cmd, cwd=repo_root)
    print("-" * 70)
    print(f"Exit code: {result.returncode}")
    return result.returncode
