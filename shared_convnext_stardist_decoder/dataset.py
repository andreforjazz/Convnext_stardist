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
    (pass `class_to_idx` from training script after scanning).
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
    ) -> None:
        super().__init__()
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.n_rays = int(n_rays)
        self.patch_size = patch_size
        self.class_to_idx = class_to_idx or {}

        self.image_paths: list[Path] = []
        for ext in extensions:
            self.image_paths.extend(sorted(self.images_dir.glob(f"*{ext}")))
        if not self.image_paths:
            raise RuntimeError(f"No images in {self.images_dir}")

    def _find_label_path(self, img_path: Path) -> Path:
        stem = img_path.stem
        for ext in (".tif", ".tiff", ".png"):
            cand = self.labels_dir / f"{stem}{ext}"
            if cand.is_file():
                return cand
        raise FileNotFoundError(f"No label for {img_path.name} in {self.labels_dir}")

    def _load_inst2class(self, stem: str) -> dict[int, int] | None:
        if not self.class_to_idx:
            return None
        p = self.labels_dir / f"{stem}_inst2class.json"
        if not p.is_file():
            return None
        raw = json.loads(p.read_text(encoding="utf-8"))
        out: dict[int, int] = {}
        for k, name in raw.items():
            name = str(name).strip().lower()
            if name not in self.class_to_idx:
                continue
            out[int(k)] = int(self.class_to_idx[name])
        return out if out else None

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        lab_path = self._find_label_path(img_path)
        stem = img_path.stem

        img = Image.open(img_path).convert("RGB")
        inst = _load_instance_map(lab_path)
        if inst.ndim != 2:
            inst = inst.squeeze()
        ps = int(self.patch_size) if self.patch_size else None
        if ps is not None:
            import cv2

            img = img.resize((ps, ps), Image.BILINEAR)
            inst = cv2.resize(
                inst.astype(np.float32), (ps, ps), interpolation=cv2.INTER_NEAREST
            ).astype(np.int32)

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
