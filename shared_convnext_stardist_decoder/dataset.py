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
IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_instance_map(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        return tifffile.imread(path)
    import cv2

    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    return im


class StardistMultitaskTileDataset(Dataset):
    """
    One sample = one tile image + instance label image (same spatial size).

    Optional sidecar `{stem}_inst2class.json`:
        {"1": "liver", "2": "heart"}
    Class indices are 0..K-1 from alphabetically sorted unique names across the dataset
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
    ) -> None:
        super().__init__()
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.n_rays = int(n_rays)
        self.patch_size = patch_size
        self.class_to_idx = class_to_idx or {}
        self.cache_to_ram = cache_to_ram
        self.ram_cache = {}

        self.image_paths: list[Path] = []
        if stems:
            stems_set = set(stems)
            # Find images for the given stems
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

        if self.cache_to_ram:
            print(f"Loading {len(self.image_paths)} samples to RAM. Depending on size, this may take a few minutes...")
            from tqdm import tqdm
            for idx in tqdm(range(len(self.image_paths)), desc="Caching to RAM"):
                self.ram_cache[idx] = self._load_item(idx)

    def _find_label_path(self, img_path: Path) -> Path:
        stem = img_path.stem
        for ext in (".tif", ".tiff", ".png"):
            cand = self.labels_dir / f"{stem}{ext}"
            if cand.is_file():
                return cand
        return None  # Return None if no mask is found

    # def _load_inst2class(self, stem: str) -> dict[int, int] | None:
        # if not self.class_to_idx:
        #     return None
        # p = self.labels_dir / f"{stem}_inst2class.json"
        # if not p.is_file():
        #     return None
        # raw = json.loads(p.read_text(encoding="utf-8"))
        # out: dict[int, int] = {}
        # for k, name in raw.items():
        #     name = str(name).strip().lower()
        #     if name not in self.class_to_idx:
        #         continue
        #     out[int(k)] = int(self.class_to_idx[name])
        # return out if out else None
    def _load_inst2class(self, stem: str) -> dict[int, int] | None:
        if not self.class_to_idx:
            return None
        p = self.labels_dir / f"{stem}_inst2class.json"
        if not p.is_file():
            return None
        try:
            txt = p.read_text(encoding="utf-8").strip()
            if not txt:
                return None
            raw = json.loads(txt)
        except Exception as e:
            # Log once per file and skip
            print(f"Warning: bad inst2class JSON for {stem}: {e}")
            return None
        out: dict[int, int] = {}
        for k, name_or_id in raw.items():
            # Accept either class names or integer ids in the JSON
            try:
                # If value is a class id
                cls_idx = int(name_or_id)
            except (ValueError, TypeError):
                # Otherwise treat as name and map via class_to_idx
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
        stem = img_path.stem

        img = Image.open(img_path).convert("RGB")
        
        if lab_path is None:
            # If no mask was found, assume the patch is empty (no cells)
            inst = np.zeros((img.height, img.width), dtype=np.int32)
        else:
            try:
                inst = _load_instance_map(lab_path)
                if inst is None or inst.size == 0:
                    inst = np.zeros((img.height, img.width), dtype=np.int32)
                elif inst.ndim != 2:
                    inst = inst.squeeze()
            except Exception as e:
                # If tiff is corrupt or empty (no pages, etc)
                print(f"Warning: Failed to load mask {lab_path}: {e}")
                inst = np.zeros((img.height, img.width), dtype=np.int32)

        ps = int(self.patch_size) if self.patch_size else None
        if ps is not None and (img.width != ps or img.height != ps):
            import cv2

            img = img.resize((ps, ps), Image.BILINEAR)
            # Ensure inst is at least a 2D array with valid dimensions before resize
            if inst.ndim >= 2 and inst.shape[0] > 0 and inst.shape[1] > 0:
                inst = cv2.resize(
                    inst.astype(np.float32), (ps, ps), interpolation=cv2.INTER_NEAREST
                ).astype(np.int32)
            else:
                inst = np.zeros((ps, ps), dtype=np.int32)

        arr = np.asarray(img, dtype=np.float32) / 255.0

        inst_to_cls = self._load_inst2class(stem)
        prob, dist, cls, fg = assemble_targets(inst, inst_to_cls, self.n_rays, ignore_index=-100)

        # NCHW, RGB
        x = torch.from_numpy(arr).permute(2, 0, 1)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        x = (x - mean) / std

        return {
            "image": x,
            "prob": torch.from_numpy(prob).unsqueeze(0),
            "dist": torch.from_numpy(dist).permute(2, 0, 1),
            "cls": torch.from_numpy(cls),
            "fg": torch.from_numpy(fg).unsqueeze(0),
        }


def build_class_to_idx_from_dir(labels_dir: Path) -> dict[str, int]:
    """Scan `*_inst2class.json` in label folder; indices follow alphabetical order."""
    labels_dir = Path(labels_dir)
    names: set[str] = set()
    if not labels_dir.is_dir():
        return {}
    for p in labels_dir.glob("*_inst2class.json"):
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            for _, v in raw.items():
                names.add(str(v).strip().lower())
        except (json.JSONDecodeError, OSError):
            continue
    ordered = sorted(names)
    return {n: i for i, n in enumerate(ordered)}
