from __future__ import annotations

import csv
import json
import math
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn.functional as F
import yaml
from scipy.io import loadmat
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from tqdm import tqdm

try:
    import openslide
except Exception as e:
    openslide = None
    _OPENSLIDE_IMPORT_ERROR = e
else:
    _OPENSLIDE_IMPORT_ERROR = None

# =============================================================================
# MANIFEST BUILDING (Notebook 1)
# =============================================================================

def norm_stem_ndpi(p: Path) -> str:
    return p.stem

def norm_stem_geojson(p: Path) -> str:
    stem = p.stem
    suffixes = ["__CODAclass", "_CODAclass", "-CODAclass"]
    for s in suffixes:
        if stem.endswith(s):
            return stem[: -len(s)]
    return stem

def read_id_set(path: Path) -> Set[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON test/dev ID file must contain a list of slide IDs.")
        return {str(x).strip() for x in data if str(x).strip()}
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}

def apply_step_selection(ids: List[str], start_index: int, step: int, max_slides: int) -> List[str]:
    if step <= 0:
        raise ValueError("step must be >= 1")
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    chosen = ids[start_index::step]
    if max_slides > 0:
        chosen = chosen[:max_slides]
    return chosen

def build_manifest(
    slides_dir: str | Path,
    geojson_dir: str | Path,
    out_json: str | Path,
    test_ids_file: str | Path = "",
    dev_ids_file: str | Path = "",
    test_fraction: float = 0.20,
    seed: int = 1337,
    start_index: int = 0,
    step: int = 1,
    max_slides: int = 0,
) -> None:
    """Matches slides with GeoJSON files and creates a manifest for training/testing."""
    slides_dir = Path(slides_dir)
    geojson_dir = Path(geojson_dir)
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    ndpi_files = sorted(slides_dir.glob("*.ndpi"))
    geojson_files = sorted(geojson_dir.glob("*.geojson"))

    ndpi_map: Dict[str, Path] = {norm_stem_ndpi(p): p for p in ndpi_files}
    geojson_map: Dict[str, Path] = {norm_stem_geojson(p): p for p in geojson_files}

    matched_ids_all = sorted(set(ndpi_map) & set(geojson_map))
    
    explicit_test_ids: Set[str] = set()
    explicit_dev_ids: Set[str] = set()
    if test_ids_file:
        explicit_test_ids = read_id_set(Path(test_ids_file))
    if dev_ids_file:
        explicit_dev_ids = read_id_set(Path(dev_ids_file))

    filtered_ids = matched_ids_all[:]
    if explicit_dev_ids:
        filtered_ids = [sid for sid in filtered_ids if sid in explicit_dev_ids or sid in explicit_test_ids]

    selected_ids = apply_step_selection(
        filtered_ids,
        start_index=start_index,
        step=step,
        max_slides=max_slides,
    )

    if explicit_test_ids:
        test_ids = sorted([sid for sid in selected_ids if sid in explicit_test_ids])
        dev_ids = sorted([sid for sid in selected_ids if sid not in explicit_test_ids])
    else:
        rng = random.Random(seed)
        shuffled = selected_ids[:]
        rng.shuffle(shuffled)
        n_test = int(round(len(shuffled) * test_fraction))
        if test_fraction > 0 and len(shuffled) > 1:
            n_test = max(1, min(n_test, len(shuffled) - 1))
        else:
            n_test = 0
        test_ids = sorted(shuffled[:n_test])
        dev_ids = sorted(shuffled[n_test:])

    manifest: List[dict] = []
    for sid in dev_ids:
        manifest.append({"slide_id": sid, "image_path": str(ndpi_map[sid]), "geojson_path": str(geojson_map[sid]), "split": "dev"})
    for sid in test_ids:
        manifest.append({"slide_id": sid, "image_path": str(ndpi_map[sid]), "geojson_path": str(geojson_map[sid]), "split": "test"})

    manifest = sorted(manifest, key=lambda x: x["slide_id"])
    out_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[WROTE] {out_json} ({len(dev_ids)} dev, {len(test_ids)} test)")

# =============================================================================
# CODA INTEGRATION (Step 0)
# =============================================================================

def assign_coda_classifications_to_geojson(
    geojson_path: str | Path,
    mask_path: str | Path,
    mat_path: str | Path,
    out_path: str | Path,
    labels: List[str],
    colors: List[List[int]],
    mpp_20x: float = 0.5,
    mpp_bb: float = 4.0,
    mpp_mask: float = 2.0,
) -> None:
    """Enriches a nuclear GeoJSON with organ classifications from a CODA mask using a multi-scale bridge."""
    geojson_path, out_path = Path(geojson_path), Path(out_path)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Mask not found at {mask_path}")

    # Load MATLAB bounding box (bb) from .mat file
    # scipy.io.loadmat cannot read v7.3 (HDF5) files — fall back to h5py.
    def _load_bb_scipy(path: Path):
        mat_data = loadmat(str(path))
        bb = mat_data.get("bb")
        if bb is None:
            possible_keys = [k for k in mat_data.keys() if not k.startswith("__")]
            if len(possible_keys) == 1:
                bb = mat_data[possible_keys[0]]
            else:
                raise ValueError(f"Could not find variable 'bb' in {path}")
        return bb.flatten()

    def _load_bb_h5py(path: Path):
        import h5py
        with h5py.File(str(path), "r") as f:
            if "bb" in f:
                arr = f["bb"][()]
            else:
                keys = [k for k in f.keys()]
                if len(keys) == 1:
                    arr = f[keys[0]][()]
                else:
                    raise ValueError(f"Could not find variable 'bb' in v7.3 mat file {path}")
        # h5py stores arrays transposed relative to MATLAB column-major order
        return arr.flatten(order="F") if arr.ndim > 1 else arr.flatten()

    try:
        bb = _load_bb_scipy(mat_path)
    except NotImplementedError:
        # MATLAB v7.3 (HDF5) file — scipy refuses to read these
        try:
            bb = _load_bb_h5py(mat_path)
        except Exception as e:
            print(f"[ERROR] Failed to load v7.3 mat {mat_path}: {e}")
            return
    except Exception as e:
        print(f"[ERROR] Failed to load {mat_path}: {e}")
        return

    # bb = [xmin_matlab, xmax_matlab, ymin_matlab, ymax_matlab] (1-indexed)
    xmin_matlab, xmax_matlab, ymin_matlab, ymax_matlab = bb[0], bb[1], bb[2], bb[3]
    
    # Transform BB origin to 20x pixels (0-indexed)
    # Scale factor from BB (MPP_BB) to 20x (MPP_20X)
    scale_bb_to_20x = mpp_bb / mpp_20x
    xmin_20x = (xmin_matlab - 1) * scale_bb_to_20x
    ymin_20x = (ymin_matlab - 1) * scale_bb_to_20x
    
    # Scale factor from 20x (MPP_20X) to Mask (MPP_MASK)
    scale_20x_to_mask = mpp_20x / mpp_mask
    
    try:
        with geojson_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Corrupted/truncated GeoJSON, skipping {geojson_path.name}: {e}")
        return

    # Support both bare list-of-features and standard FeatureCollection dict
    if isinstance(data, list):
        feats = data
    else:
        feats = data.get("features", [])
    label_map = {i + 1: {"name": labels[i], "color": colors[i]} for i in range(len(labels))}
    
    n_assigned = 0
    for feat in feats:
        geom = feat.get("geometry", {})
        if geom.get("type") != "Polygon": continue
        ring = geom.get("coordinates", [[]])[0]
        pts_arr = np.asarray(ring, dtype=np.float64)
        if pts_arr.shape[0] < 3: continue
        cx_20x, cy_20x = polygon_centroid(pts_arr)
        if cx_20x is None: continue
        
        # 1. Transform Centroids: 20x absolute → Target mask relative
        # Subtract BB origin (in 20x coordinates)
        cx_20x_relative = cx_20x - xmin_20x
        cy_20x_relative = cy_20x - ymin_20x
        
        # Scale from 20x to Mask pixels
        mx_f = cx_20x_relative * scale_20x_to_mask
        my_f = cy_20x_relative * scale_20x_to_mask
        
        # Local coordinates in mask pixels
        mx, my = int(round(mx_f)), int(round(my_f))
        
        # 2. Sample mask
        if 0 <= mx < mask.shape[1] and 0 <= my < mask.shape[0]:
            val = mask[my, mx]
            if val in label_map:
                feat["properties"]["classification"] = label_map[val]
                n_assigned += 1
            else:
                feat["properties"]["classification"] = {"name": "Unassigned", "color": [128, 128, 128]}
        else:
            feat["properties"]["classification"] = {"name": "OutsideMask", "color": [0, 0, 0]}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    
    # print(f"[DONE] Slide {geojson_path.stem}: Assigned {n_assigned}/{len(feats)} labels.")

# =============================================================================
# DATASET BUILDING (Notebook 2)
# =============================================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def polygon_centroid(points: List[List[float]] | np.ndarray) -> Tuple[float, float] | Tuple[None, None]:
    pts_arr = np.asarray(points, dtype=np.float64)
    if pts_arr.shape[0] < 3:
        return None, None
    if not np.allclose(pts_arr[0], pts_arr[-1]):
        pts_arr = np.vstack([pts_arr, pts_arr[0]])
    x0, y0 = pts_arr[:-1, 0], pts_arr[:-1, 1]
    x1, y1 = pts_arr[1:, 0], pts_arr[1:, 1]
    cross = x0 * y1 - x1 * y0
    area2 = np.sum(cross)
    if abs(area2) < 1e-12:
        return float(np.mean(pts_arr[:-1, 0])), float(np.mean(pts_arr[:-1, 1]))
    cx = np.sum((x0 + x1) * cross) / (3.0 * area2)
    cy = np.sum((y0 + y1) * cross) / (3.0 * area2)
    return float(cx), float(cy)

def get_slide_mpp(slide, manifest_source_mpp: Optional[float] = None) -> float:
    if manifest_source_mpp is not None:
        return float(manifest_source_mpp)
    props = slide.properties
    candidate_keys = [getattr(openslide, "PROPERTY_NAME_MPP_X", "openslide.mpp-x"), "openslide.mpp-x", "hamamatsu.XResolution", "aperio.MPP"]
    for key in candidate_keys:
        v = props.get(key)
        try:
            v_f = float(v) if v is not None else None
            if v_f is not None and v_f > 0:
                if key == "hamamatsu.XResolution" and v_f > 10: continue
                return float(v_f)
        except Exception: continue
    raise RuntimeError("Could not infer source_mpp from slide metadata.")

def read_ndpi_tile_resampled(slide, x_target: int, y_target: int, tile_size_target: int, source_mpp: float, target_mpp: float) -> np.ndarray:
    scale = float(target_mpp) / float(source_mpp)
    x0_level0 = int(round(x_target * scale))
    y0_level0 = int(round(y_target * scale))
    read_w = max(1, int(round(tile_size_target * scale)))
    read_h = max(1, int(round(tile_size_target * scale)))
    rgba = np.asarray(slide.read_region((x0_level0, y0_level0), 0, (read_w, read_h)))
    rgb = rgba[..., :3]
    if rgb.shape[0] != tile_size_target or rgb.shape[1] != tile_size_target:
        rgb = cv2.resize(rgb, (tile_size_target, tile_size_target), interpolation=cv2.INTER_LINEAR)
    return rgb.astype(np.uint8)

def assign_points_to_tiles_nonoverlap(xy: np.ndarray, x0s: np.ndarray, y0s: np.ndarray, tile: int, stride: int) -> Dict[Tuple[int, int], np.ndarray]:
    if xy.shape[0] == 0: return {}
    nx, ny = len(x0s), len(y0s)
    ix, iy = (xy[:, 0] / stride).astype(np.int32), (xy[:, 1] / stride).astype(np.int32)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    groups = defaultdict(list)
    for p, a, b in zip(np.nonzero(valid)[0].tolist(), ix[valid].tolist(), iy[valid].tolist()):
        x0, y0 = int(x0s[a]), int(y0s[b])
        if x0 <= xy[p, 0] < x0 + tile and y0 <= xy[p, 1] < y0 + tile:
            groups[(a, b)].append(p)
    return {k: np.asarray(v, dtype=np.int32) for k, v in groups.items() if len(v) > 0}

def build_cellvit_dataset(
    manifest_json: str | Path,
    labels_json: str | Path,
    out_dir: str | Path,
    tile_size: int = 256,
    stride: int = 256,
    target_mpp: float = 0.25,
    max_tiles_per_slide: int = 1500,
    min_cells_per_tile: int = 5,
    sample_mode: str = "random",
    n_folds: int = 3,
    seed: int = 1337,
    overwrite: bool = False,
    write_qc_tiles: bool = True,
    max_qc_tiles_per_slide: int = 5,
    merge_bone_skull: bool = True,
    skip_class_names: str = "",
    cellvit_path: str = "",
):
    """Orchestrates dataset creation from manifest."""
    out_dir = Path(out_dir)
    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    
    images_dir = out_dir / "train" / "images"
    labels_dir = out_dir / "train" / "labels"
    qc_dir = out_dir / "train" / "qc"
    for d in [images_dir, labels_dir, qc_dir]: d.mkdir(parents=True, exist_ok=True)

    # 1. Load label spec
    label_data = json.loads(Path(labels_json).read_text(encoding="utf-8"))
    skip_set = {s.strip() for s in skip_class_names.split(",") if s.strip()}
    
    final_names = []
    class_remap = {}
    for rid, payload in sorted(label_data.items(), key=lambda x: int(x[0])):
        name = payload["name"]
        final_name = "bone_skull" if merge_bone_skull and name in {"bone", "skull"} else name
        class_remap[name] = final_name
        if final_name not in skip_set and final_name not in final_names:
            final_names.append(final_name)
    
    train_to_name = {i: n for i, n in enumerate(final_names)}
    name_to_train = {n: i for i, n in train_to_name.items()}

    # 2. Process slides
    manifest = json.loads(Path(manifest_json).read_text(encoding="utf-8"))
    slide_ids_by_split = defaultdict(list)
    tile_records = []
    global_class_counter = Counter()

    for entry in tqdm(manifest, desc="Tile extraction"):
        sid = entry["slide_id"]
        slide_ids_by_split[entry["split"]].append(sid)
        
        # Parse GeoJSON
        gj_path = Path(entry["geojson_path"])
        gj_data = json.loads(gj_path.read_text(encoding="utf-8"))
        feats = gj_data.get("features", [])
        centroids, cls_ids = [], []
        for feat in feats:
            cname = feat.get("properties", {}).get("classification", {}).get("name")
            if not cname: continue
            fname = class_remap.get(cname, cname)
            if fname not in name_to_train: continue
            
            geom = feat.get("geometry", {})
            if geom.get("type") != "Polygon": continue
            ring = geom.get("coordinates", [[]])[0]
            cx, cy = polygon_centroid(ring)
            if cx is not None:
                centroids.append((cx, cy))
                cls_ids.append(name_to_train[fname])
        
        if not centroids: continue
        xy = np.asarray(centroids, np.float32)
        labs = np.asarray(cls_ids, np.int32)

        # Slide access
        slide = openslide.OpenSlide(entry["image_path"])
        src_mpp = get_slide_mpp(slide)
        scale = src_mpp / target_mpp
        W_tgt, H_tgt = int(round(slide.dimensions[0] * scale)), int(round(slide.dimensions[1] * scale))
        
        x0s = np.arange(0, W_tgt - tile_size + 1, stride)
        y0s = np.arange(0, H_tgt - tile_size + 1, stride)
        groups = assign_points_to_tiles_nonoverlap(xy * scale, x0s, y0s, tile_size, stride)
        
        eligible = [k for k, v in groups.items() if len(v) >= min_cells_per_tile]
        if not eligible: 
            slide.close(); continue
        
        if sample_mode == "random":
            random.seed(seed); random.shuffle(eligible)
        elif sample_mode == "topk":
            eligible.sort(key=lambda k: len(groups[k]), reverse=True)
        
        if max_tiles_per_slide > 0: eligible = eligible[:max_tiles_per_slide]

        for i, (ax, ay) in enumerate(eligible):
            tid = f"{sid}_t{i}_{ax}_{ay}"
            x_tgt, y_tgt = x0s[ax], y0s[ay]
            tile_rgb = read_ndpi_tile_resampled(slide, x_tgt, y_tgt, tile_size, src_mpp, target_mpp)
            cv2.imwrite(str(images_dir / f"{tid}.png"), cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR))
            
            p_indices = groups[(ax, ay)]
            local_xy = (xy[p_indices] * scale) - [x_tgt, y_tgt]
            local_labs = labs[p_indices]
            
            with (labels_dir / f"{tid}.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                for (lx, ly), lc in zip(local_xy, local_labs):
                    writer.writerow([int(lx), int(ly), int(lc)])
                    if entry["split"] == "dev": global_class_counter.update([int(lc)])
            
            tile_records.append({"tile_id": tid, "slide_id": sid, "split": entry["split"]})
        
        slide.close()

    # 3. Splits & Configs
    df = pd.DataFrame(tile_records)
    test_ids = df[df["split"] == "test"]["tile_id"].tolist()
    (out_dir / "splits").mkdir(exist_ok=True)
    pd.DataFrame(test_ids).to_csv(out_dir / "splits" / "test.csv", index=False, header=False)
    
    dev_slides = slide_ids_by_split["dev"]
    random.seed(seed); random.shuffle(dev_slides)
    for f in range(n_folds):
        f_val_slides = dev_slides[f::n_folds]
        f_train_slides = [s for s in dev_slides if s not in f_val_slides]
        f_dir = out_dir / "splits" / f"fold_{f}"
        f_dir.mkdir(parents=True, exist_ok=True)
        df[df["slide_id"].isin(f_train_slides)]["tile_id"].to_csv(f_dir / "train.csv", index=False, header=False)
        df[df["slide_id"].isin(f_val_slides)]["tile_id"].to_csv(f_dir / "val.csv", index=False, header=False)
        
        # Write YAML
        cfg = {
            "logging": {"mode": "online", "project": "cellvit++", "log_comment": f"fold_{f}"},
            "data": {
                "dataset_path": str(out_dir), "num_classes": len(train_to_name),
                "train_filelist": str(f_dir / "train.csv"), "val_filelist": str(f_dir / "val.csv"),
                "label_map": train_to_name
            },
            "cellvit_path": cellvit_path, "training": {"batch_size": 64, "epochs": 30}
        }
        yaml_path = out_dir / "train_configs" / "ViT256" / f"fold_{f}.yaml"
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with yaml_path.open("w") as fy: yaml.safe_dump(cfg, fy)

    with (out_dir / "label_map.yaml").open("w") as fl: yaml.safe_dump({"labels": train_to_name}, fl)
    print(f"[DONE] Created dataset in {out_dir}")


# =============================================================================
# INFERENCE (Notebook 4)
# =============================================================================

class OmeTiffRegionReader:
    def __init__(self, image_path: Path):
        self.image_path = image_path
        self.tif = tifffile.TiffFile(str(image_path))
        self.series = self.tif.series[0]
        self.axes = self.series.axes
        self.shape = self.series.shape
        self.height, self.width, self.channels = 0, 0, 0
        if self.axes in ("YXS", "YXC"):
            self.height, self.width, self.channels = self.shape[0], self.shape[1], self.shape[2]
        elif self.axes == "CYX":
            self.channels, self.height, self.width = self.shape[0], self.shape[1], self.shape[2]
        self._arr = None
        try:
            import zarr
            self._arr = zarr.open(self.series.aszarr(), mode="r")
            if not hasattr(self._arr, "shape"): self._arr = self._arr[list(self._arr.keys())[0]]
        except Exception: pass

    def read_region(self, y0, y1, x0, x1) -> np.ndarray:
        y0, y1, x0, x1 = max(0, y0), min(self.height, y1), max(0, x0), min(self.width, x1)
        if y1 <= y0 or x1 <= x0: return np.zeros((0, 0, 3), dtype=np.uint8)
        if self._arr is not None:
            if self.axes in ("YXS", "YXC"): crop = np.asarray(self._arr[y0:y1, x0:x1, :3])
            else: crop = np.moveaxis(np.asarray(self._arr[:3, y0:y1, x0:x1]), 0, -1)
        else:
            full = self.series.asarray()
            if self.axes == "CYX": full = np.moveaxis(full, 0, -1)
            crop = full[y0:y1, x0:x1, :3]
        return np.clip(crop, 0, 255).astype(np.uint8)

    def close(self): self.tif.close()

def run_guided_inference(
    image_path: str | Path,
    geojson_path: str | Path,
    config_path: str | Path,
    ckpt_path: str | Path,
    out_dir: str | Path,
    cellvit_path: str = "",
    target_mpp: float = 0.25,
    source_mpp: Optional[float] = None,
    device: str = "cuda:0",
):
    """Runs inference on a WSI using GT centroids from a GeoJSON."""
    from cellvit.training.experiments.experiment_cell_classifier import ExperimentCellVitClassifier
    from cellvit.models.classifier.linear_classifier import LinearClassifier

    image_path, out_dir = Path(image_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with Path(config_path).open("r") as f: run_conf = yaml.safe_load(f)
    label_map = {int(k): str(v) for k, v in run_conf["data"]["label_map"].items()}
    label_map_inv = {v: k for k, v in label_map.items()}

    reader = OmeTiffRegionReader(image_path)
    scale = (source_mpp or 0.4415) / target_mpp
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    exp = ExperimentCellVitClassifier(default_conf=run_conf)
    cv_model, cv_conf = exp.load_cellvit_model(cellvit_path or run_conf["cellvit_path"])
    cv_model.to(device).eval()

    classifier = LinearClassifier(embed_dim=384, hidden_dim=run_conf["model"]["hidden_dim"], num_classes=run_conf["data"]["num_classes"])
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    sd = {k.replace("model.", "").replace("module.", "").replace("classifier.", "").replace("head.", ""): v for k, v in sd.items()}
    classifier.load_state_dict(sd, strict=False)
    classifier.to(device).eval()

    # Parse GeoJSON
    gj_data = json.loads(Path(geojson_path).read_text(encoding="utf-8"))
    objects = []
    for feat in gj_data.get("features", []):
        cname = feat.get("properties", {}).get("classification", {}).get("name")
        if cname not in label_map_inv: continue
        ring = feat.get("geometry", {}).get("coordinates", [[]])[0]
        cx, cy = polygon_centroid(ring)
        if cx is not None:
            objects.append({"centroid_src": np.asarray([cx, cy]), "true_id": label_map_inv[cname], "true_name": cname})

    tile_to_idx = defaultdict(list)
    for i, obj in enumerate(objects):
        c_tgt = obj["centroid_src"] * scale
        tile_to_idx[(int(c_tgt[0] // 256), int(c_tgt[1] // 256))].append(i)

    records = []
    for (tx, ty), indices in tqdm(tile_to_idx.items(), desc="Guided inference"):
        x0_s, y0_s = int(tx * 256 / scale), int(ty * 256 / scale)
        x1_s, y1_s = int((tx+1) * 256 / scale), int((ty+1) * 256 / scale)
        crop = reader.read_region(y0_s, y1_s, x0_s, x1_s)
        if crop.size == 0: continue
        tile_rgb = cv2.resize(crop, (256, 256))
        
        local_xy = np.asarray([(obj["centroid_src"][0] - x0_s) * (256 / (x1_s - x0_s)), 
                               (obj["centroid_src"][1] - y0_s) * (256 / (y1_s - y0_s))] 
                              for obj in [objects[i] for i in indices])
        
        t = torch.from_numpy(np.transpose(tile_rgb.astype(np.float32)/255.0, (2,0,1))).unsqueeze(0).to(device)
        with torch.no_grad():
            enc_out = cv_model.encoder(t)
            z = enc_out[2]
            b, n, c = z.shape
            s = int(math.sqrt(n-1))
            fmap = z[:, 1:, :].reshape(b, s, s, c).permute(0, 3, 1, 2)
            
            gx = np.clip(np.rint((local_xy[:,0]/255.0)*(s-1)).astype(np.int64), 0, s-1)
            gy = np.clip(np.rint((local_xy[:,1]/255.0)*(s-1)).astype(np.int64), 0, s-1)
            feats = torch.stack([fmap[0, :, gy[i], gx[i]] for i in range(len(indices))])
            probs = F.softmax(classifier(feats), dim=1)
            conf, pred_ids = torch.max(probs, dim=1)
        
        for j, oi in enumerate(indices):
            pid, cid = int(pred_ids[j]), int(conf[j])
            records.append({**objects[oi], "pred_id": pid, "pred_name": label_map[pid], "confidence": float(cid)})
    
    reader.close()
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "wsi_predictions.csv", index=False)
    print(f"[DONE] Inference results in {out_dir}")

# =============================================================================
# EVALUATION (Notebook 5)
# =============================================================================

def compute_metrics(predictions_csv: str | Path, label_map: Dict[int, str], out_dir: str | Path, prefix: str = "eval"):
    df = pd.read_csv(predictions_csv)
    out_dir = Path(out_dir)
    labels = sorted(label_map.keys())
    l_names = [label_map[i] for i in labels]
    
    y_true, y_pred = df["true_id"], df["pred_id"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im)
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)), xticklabels=l_names, yticklabels=l_names, title=f"Confusion Matrix ({prefix})", ylabel="True", xlabel="Predicted")
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_cm.png")
    
    report = classification_report(y_true, y_pred, labels=labels, target_names=l_names, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(out_dir / f"{prefix}_report.csv")
    print(f"[DONE] Metrics saved in {out_dir}")
