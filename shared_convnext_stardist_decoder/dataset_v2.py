"""
dataset_v2.py  —  StardistMultitaskTileDatasetV2

Changes vs dataset.py (v1):
  Fix #2 support — Returns the raw instance map (`inst`) in each batch dict so
                   losses_v2.py can run instance-level CE without re-loading data.

  Fix #4 (partial) — `cls_only` flag.  When True, tiles that have no inst2class
                      supervision are silently skipped; the DataLoader only sees
                      tiles where at least one nucleus has a known class label.
                      Use with WeightedRandomSampler (see train_v2.py) or as a
                      strict filter for cls-focused ablation runs.

  Everything else (path discovery, TIFF loading, JSON hardening, target assembly)
  is identical to v1 so existing configs and pre-processed labels work unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import tifffile

from .targets import assemble_targets


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ── Low-level helpers (identical to v1) ──────────────────────────────────────

def _load_instance_map(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            with tifffile.TiffFile(path) as tf:
                if len(tf.pages) == 0:
                    raise ValueError("TIFF contains no pages")
            arr = tifffile.imread(path)
        except Exception as e:
            raise FileNotFoundError(f"{path}: {e}") from e
        if arr is None or arr.size == 0:
            raise FileNotFoundError(f"{path}: empty read")
        return arr
    import cv2
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    return im


def _read_inst2class_raw(path: Path) -> dict | None:
    try:
        raw_bytes = path.read_bytes()
    except OSError:
        return None
    if not raw_bytes.strip():
        return None
    txt = raw_bytes.decode("utf-8-sig").strip()
    if not txt:
        return None
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return None


# ── Dataset ───────────────────────────────────────────────────────────────────

class StardistMultitaskTileDatasetV2(Dataset):
    """
    Returns the same keys as v1 PLUS:
        "inst": (H,W) int64 — raw instance label map (0 = background).

    Parameters
    ----------
    cls_only : bool
        When True, only tiles that have at least one nucleus with a known class
        label are included.  Use to ensure every batch has classification signal.
    """

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        *,
        n_rays: int,
        patch_size: int | None = None,
        class_to_idx: dict[str, int] | None = None,
        extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
        stems: list[str] | None = None,
        cache_to_ram: bool = False,
        cls_only: bool = False,    # Fix #4: skip tiles with no cls supervision
    ) -> None:
        super().__init__()
        self.images_dir   = Path(images_dir)
        self.labels_dir   = Path(labels_dir)
        self.n_rays       = int(n_rays)
        self.patch_size   = patch_size
        self.class_to_idx = class_to_idx or {}
        self.cache_to_ram = cache_to_ram
        self.cls_only     = cls_only
        self.ram_cache    = {}
        self._bad_inst2class_warn = 0

        # Collect image paths
        self.image_paths: list[Path] = []
        if stems:
            stems_set = set(stems)
            for ext in extensions:
                for p in self.images_dir.glob(f"*{ext}"):
                    if p.stem in stems_set:
                        self.image_paths.append(p)
            self.image_paths.sort(key=lambda x: x.stem)
        else:
            for ext in extensions:
                self.image_paths.extend(sorted(self.images_dir.glob(f"*{ext}")))

        if not self.image_paths:
            msg = f"No images in {self.images_dir}"
            if stems:
                msg += f" matching {len(stems)} stems"
            raise RuntimeError(msg)

        # Fix #4: filter to tiles that have inst2class coverage
        if cls_only and self.class_to_idx:
            before = len(self.image_paths)
            self.image_paths = [
                p for p in self.image_paths
                if self._has_cls_supervision(p.stem)
            ]
            print(
                f"cls_only=True: kept {len(self.image_paths)}/{before} tiles "
                f"that have inst2class supervision"
            )

        if self.cache_to_ram:
            print(f"Loading {len(self.image_paths)} samples to RAM …")
            from tqdm import tqdm
            for idx in tqdm(range(len(self.image_paths)), desc="Caching to RAM"):
                self.ram_cache[idx] = self._load_item(idx)

    def _has_cls_supervision(self, stem: str) -> bool:
        """Return True if a non-empty inst2class JSON exists for this stem."""
        p = self.labels_dir / f"{stem}_inst2class.json"
        if not p.is_file():
            return False
        raw = _read_inst2class_raw(p)
        return isinstance(raw, dict) and len(raw) > 0

    def _find_label_path(self, img_path: Path) -> Path | None:
        stem = img_path.stem
        for ext in (".tif", ".tiff", ".png"):
            cand = self.labels_dir / f"{stem}{ext}"
            if cand.is_file():
                return cand
        return None

    def _load_inst2class(self, stem: str) -> dict[int, int] | None:
        if not self.class_to_idx:
            return None
        p = self.labels_dir / f"{stem}_inst2class.json"
        if not p.is_file():
            return None
        raw = _read_inst2class_raw(p)
        if raw is None:
            if self._bad_inst2class_warn < 8:
                print(f"Warning: missing or invalid inst2class JSON for {p.name}")
            self._bad_inst2class_warn += 1
            return None
        if not isinstance(raw, dict):
            return None
        out: dict[int, int] = {}
        for k, name_or_id in raw.items():
            try:
                cls_idx = int(name_or_id)
            except (ValueError, TypeError):
                name = str(name_or_id).strip().lower()
                if name not in self.class_to_idx:
                    continue
                cls_idx = int(self.class_to_idx[name])
            out[int(k)] = cls_idx
        return out if out else None

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.cache_to_ram:
            return self.ram_cache[idx]
        return self._load_item(idx)

    def _load_item(self, idx: int) -> dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        lab_path = self._find_label_path(img_path)
        stem     = img_path.stem

        img = Image.open(img_path).convert("RGB")

        if lab_path is None:
            inst = np.zeros((img.height, img.width), dtype=np.int32)
        else:
            try:
                inst = _load_instance_map(lab_path)
                if inst is None or inst.size == 0:
                    inst = np.zeros((img.height, img.width), dtype=np.int32)
                elif inst.ndim != 2:
                    inst = inst.squeeze()
            except Exception as e:
                print(f"Warning: Failed to load mask {lab_path}: {e}")
                inst = np.zeros((img.height, img.width), dtype=np.int32)

        ps = int(self.patch_size) if self.patch_size else None
        if ps is not None and (img.width != ps or img.height != ps):
            import cv2
            img = img.resize((ps, ps), Image.BILINEAR)
            if inst.ndim >= 2 and inst.shape[0] > 0 and inst.shape[1] > 0:
                inst = cv2.resize(
                    inst.astype(np.float32), (ps, ps),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.int32)
            else:
                inst = np.zeros((ps, ps), dtype=np.int32)

        arr          = np.asarray(img, dtype=np.float32) / 255.0
        inst_to_cls  = self._load_inst2class(stem)
        prob, dist, cls, fg = assemble_targets(inst, inst_to_cls, self.n_rays, ignore_index=-100)

        x    = torch.from_numpy(arr).permute(2, 0, 1)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        x    = (x - mean) / std

        return {
            "image": x,
            "prob":  torch.from_numpy(prob).unsqueeze(0),
            "dist":  torch.from_numpy(dist).permute(2, 0, 1),
            "cls":   torch.from_numpy(cls),
            "fg":    torch.from_numpy(fg).unsqueeze(0),
            "inst":  torch.from_numpy(inst.astype(np.int64)),   # NEW for instance-level loss
        }


def build_class_to_idx_from_dir(labels_dir: Path) -> dict[str, int]:
    labels_dir = Path(labels_dir)
    names: set[str] = set()
    if not labels_dir.is_dir():
        return {}
    for p in labels_dir.glob("*_inst2class.json"):
        raw = _read_inst2class_raw(p)
        if not isinstance(raw, dict):
            continue
        for _, v in raw.items():
            try:
                names.add(str(v).strip().lower())
            except (AttributeError, TypeError):
                continue
    ordered = sorted(names)
    return {n: i for i, n in enumerate(ordered)}


def compute_class_weights(
    labels_dir: Path,
    class_to_idx: dict[str, int],
    mode: str = "inv_sqrt_freq",
) -> torch.Tensor:
    """
    Scan all *_inst2class.json files and count instances per class.
    Returns a (num_classes,) float32 tensor of loss weights.

    mode:
        "inv_freq"      — weight = 1 / count   (aggressive, good for very rare classes)
        "inv_sqrt_freq" — weight = 1/√count     (milder, usually safer)
        "uniform"       — weight = 1 for all    (disables reweighting)
    """
    n_cls = len(class_to_idx)
    counts = torch.zeros(n_cls, dtype=torch.float64)

    for p in Path(labels_dir).glob("*_inst2class.json"):
        raw = _read_inst2class_raw(p)
        if not isinstance(raw, dict):
            continue
        for _, v in raw.items():
            try:
                cls_idx = int(v)
            except (ValueError, TypeError):
                name = str(v).strip().lower()
                cls_idx = class_to_idx.get(name, -1)
            if 0 <= cls_idx < n_cls:
                counts[cls_idx] += 1

    if mode == "uniform":
        return torch.ones(n_cls, dtype=torch.float32)

    counts = counts.clamp(min=1.0)  # avoid div-by-zero for unseen classes
    if mode == "inv_freq":
        w = 1.0 / counts
    else:  # inv_sqrt_freq (default)
        w = 1.0 / counts.sqrt()

    w = w / w.mean()  # normalise so mean weight = 1 (keeps loss scale stable)
    print("Class weights (inv_sqrt_freq, normalised):")
    idx2name = {v: k for k, v in class_to_idx.items()}
    for i, wi in enumerate(w):
        print(f"  [{i:2d}] {idx2name.get(i, '?'):20s}  count={int(counts[i]):6d}  weight={wi:.3f}")
    return w.float()
