#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train ConvNeXt on a CellViT-style dataset:
- train/images/*.png
- train/labels/*.csv   (NO HEADER; col0=x, col1=y, col2=class)
- splits/fold_0/train.csv
- splits/fold_0/val.csv
- label_map.yaml

The script samples nucleus-centered patches from tile images using the centroid
coordinates stored in each tile label CSV.

This version is adapted to the GS40 fetal dataset.
"""

import re
import json
import time
import random
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
import yaml
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoImageProcessor, AutoModelForImageClassification
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "facebook/convnextv2-tiny-22k-224"

DATASET_ROOT = Path(r"\\kittyserverdw\Andre_kit\data\students\Diogo\data\fetal\GS40\cellvit_training\data_for_cellvit_GS40_balanced")
FOLD_DIR = DATASET_ROOT / "splits" / "fold_0"

TRAIN_IMAGES_DIR = DATASET_ROOT / "train" / "images"
TRAIN_LABELS_DIR = DATASET_ROOT / "train" / "labels"

LABEL_MAP_YAML = DATASET_ROOT / "label_map.yaml"
TRAIN_SPLIT_CSV = FOLD_DIR / "train.csv"
VAL_SPLIT_CSV   = FOLD_DIR / "val.csv"

OUT_DIR = DATASET_ROOT / "convnext_runs" / "run_fold0_convnextv2-tiny-22k-224"

PATCH_SIZE = 256
BATCH_SIZE = 64
NUM_EPOCHS = 20
LR = 3e-5
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 1337

NUM_WORKERS_TRAIN = 4
NUM_WORKERS_VAL = 4
PIN_MEMORY = True

USE_AMP = True
LOG_EVERY_STEPS = 50
EARLY_STOPPING_PATIENCE = 5

# If not None, randomly subsample this many training nuclei per epoch
TRAIN_SAMPLES_PER_EPOCH = None   # e.g. 200000
VAL_MAX_SAMPLES = None           # e.g. 50000

# Optional label merging
MERGE_SKULL_INTO_BONE = False
MERGE_SPLEEN2_INTO_SPLEEN = False

# Optional labels to drop
DROP_LABEL_NAMES = set()         # e.g. {"nontissue"}

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

# ============================================================


def seed_everything(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_name(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def load_label_map_yaml(path: Path) -> Dict[int, str]:
    data = load_yaml(path)
    if "labels" not in data:
        raise ValueError(f"'labels' key not found in {path}")
    out = {}
    for k, v in data["labels"].items():
        out[int(k)] = str(v)
    return out


def build_label_space(
    raw_id_to_name: Dict[int, str],
    merge_skull_into_bone: bool = False,
    merge_spleen2_into_spleen: bool = False,
    drop_label_names: Optional[set] = None,
):
    if drop_label_names is None:
        drop_label_names = set()

    drop_norm = {normalize_name(x) for x in drop_label_names}

    raw_to_final_name = {}
    for raw_id, name in raw_id_to_name.items():
        n = normalize_name(name)

        if merge_skull_into_bone and n == "skull":
            n = "bone"
        if merge_spleen2_into_spleen and n == "spleen2":
            n = "spleen"

        if n in drop_norm:
            raw_to_final_name[raw_id] = None
        else:
            raw_to_final_name[raw_id] = n

    final_names = sorted({v for v in raw_to_final_name.values() if v is not None})
    final_name_to_id = {name: i for i, name in enumerate(final_names)}
    final_id_to_name = {i: name for name, i in final_name_to_id.items()}

    raw_to_final_id = {}
    for raw_id, final_name in raw_to_final_name.items():
        if final_name is None:
            raw_to_final_id[raw_id] = None
        else:
            raw_to_final_id[raw_id] = final_name_to_id[final_name]

    return raw_to_final_id, final_id_to_name, final_name_to_id


def read_split_ids(split_csv: Path) -> List[str]:
    """
    Robust loader for CellViT split CSVs.
    Supports:
    - one column without header
    - one column with header
    - first column being tile id / filename
    """
    df = pd.read_csv(split_csv)
    if df.shape[1] == 0:
        raise ValueError(f"No columns found in {split_csv}")

    for col in ["id", "tile_id", "name", "filename", "file", "image"]:
        if col in df.columns:
            vals = df[col].astype(str).tolist()
            return [Path(v).stem for v in vals]

    vals = df.iloc[:, 0].astype(str).tolist()
    return [Path(v).stem for v in vals]


def find_existing_image(base_stem: str, images_dir: Path) -> Path:
    for ext in IMAGE_EXTENSIONS:
        p = images_dir / f"{base_stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find image for tile '{base_stem}' in {images_dir}")


def find_existing_label_csv(base_stem: str, labels_dir: Path) -> Path:
    p = labels_dir / f"{base_stem}.csv"
    if p.exists():
        return p
    raise FileNotFoundError(f"Could not find label CSV for tile '{base_stem}' in {labels_dir}")


def parse_class_value(v):
    if pd.isna(v):
        return None

    s = str(v).strip()
    if s == "":
        return None

    try:
        if "." in s:
            f = float(s)
            if f.is_integer():
                return int(f)
        return int(s)
    except Exception:
        return s


def crop_with_padding(img: Image.Image, cx: float, cy: float, patch_size: int) -> Image.Image:
    """
    Crop a patch centered at (cx, cy).
    Pads with white if the patch goes outside the tile.
    """
    w, h = img.size
    half = patch_size // 2

    left = int(round(cx - half))
    top = int(round(cy - half))
    right = left + patch_size
    bottom = top + patch_size

    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - w)
    pad_bottom = max(0, bottom - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        img = ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=(255, 255, 255))
        left += pad_left
        top += pad_top
        right += pad_left
        bottom += pad_top

    patch = img.crop((left, top, right, bottom))
    return patch


class CellRecord:
    __slots__ = ["tile_id", "image_path", "label_csv", "x", "y", "label_id"]

    def __init__(self, tile_id, image_path, label_csv, x, y, label_id):
        self.tile_id = tile_id
        self.image_path = image_path
        self.label_csv = label_csv
        self.x = float(x)
        self.y = float(y)
        self.label_id = int(label_id)


def build_records_from_split(
    split_ids: List[str],
    images_dir: Path,
    labels_dir: Path,
    raw_to_final_id: Dict[int, Optional[int]],
    final_name_to_id: Dict[str, int],
):
    records: List[CellRecord] = []
    skipped_missing = 0
    skipped_unknown = 0
    skipped_dropped = 0

    for tile_id in tqdm(split_ids, desc="Indexing tiles", dynamic_ncols=True):
        try:
            image_path = find_existing_image(tile_id, images_dir)
            label_csv = find_existing_label_csv(tile_id, labels_dir)
        except FileNotFoundError:
            skipped_missing += 1
            continue

        try:
            # GS40 format: no header, columns are x,y,class
            df = pd.read_csv(label_csv, header=None)
        except Exception:
            skipped_missing += 1
            continue

        if len(df) == 0:
            continue

        if len(df.columns) < 3:
            skipped_unknown += len(df)
            continue

        df = df.iloc[:, :3].copy()
        df.columns = ["x", "y", "class"]

        for _, row in df.iterrows():
            x = row["x"]
            y = row["y"]
            cls_val = parse_class_value(row["class"])

            if pd.isna(x) or pd.isna(y) or cls_val is None:
                skipped_unknown += 1
                continue

            final_id = None

            if isinstance(cls_val, int):
                final_id = raw_to_final_id.get(cls_val, None)
            else:
                cls_name_norm = normalize_name(cls_val)

                if MERGE_SKULL_INTO_BONE and cls_name_norm == "skull":
                    cls_name_norm = "bone"
                if MERGE_SPLEEN2_INTO_SPLEEN and cls_name_norm == "spleen2":
                    cls_name_norm = "spleen"

                if cls_name_norm in {normalize_name(x) for x in DROP_LABEL_NAMES}:
                    final_id = None
                else:
                    final_id = final_name_to_id.get(cls_name_norm, None)

            if final_id is None:
                skipped_dropped += 1
                continue

            records.append(CellRecord(tile_id, image_path, label_csv, x, y, final_id))

    return records, {
        "skipped_missing": skipped_missing,
        "skipped_unknown": skipped_unknown,
        "skipped_dropped": skipped_dropped,
    }


def print_class_distribution(name: str, records: List[CellRecord], id2label: Dict[int, str]):
    counts = Counter([r.label_id for r in records])
    total = len(records)
    print(f"\n[DIST] {name}: n={total:,}")
    for cid in sorted(counts):
        cname = id2label[cid]
        print(f"  {cid:2d} | {cname:12s} | {counts[cid]:8d} | {100.0 * counts[cid] / max(1, total):6.2f}%")


class CellTilePatchDataset(Dataset):
    def __init__(self, records: List[CellRecord], processor, patch_size: int):
        self.records = records
        self.processor = processor
        self.patch_size = patch_size
        self._img_cache = {}

    def __len__(self):
        return len(self.records)

    def _get_image(self, path: Path) -> Image.Image:
        key = str(path)
        if key not in self._img_cache:
            img = Image.open(path).convert("RGB")
            self._img_cache[key] = img
        return self._img_cache[key]

    def __getitem__(self, idx):
        r = self.records[idx]
        img = self._get_image(r.image_path)
        patch = crop_with_padding(img, r.x, r.y, self.patch_size)

        pixel_values = self.processor(images=patch, return_tensors="pt")["pixel_values"].squeeze(0)
        label = torch.tensor(r.label_id, dtype=torch.long)
        return pixel_values, label


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int):
    model.eval()

    total = 0
    total_correct = 0
    per_class_correct = np.zeros(num_classes, dtype=np.int64)
    per_class_total = np.zeros(num_classes, dtype=np.int64)
    running_loss = 0.0

    criterion = torch.nn.CrossEntropyLoss()

    for x, y in tqdm(loader, desc="Validating", dynamic_ncols=True):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(pixel_values=x).logits
        loss = criterion(logits, y)

        pred = logits.argmax(dim=1)

        running_loss += loss.item() * y.size(0)
        total += y.numel()
        total_correct += (pred == y).sum().item()

        for cls in range(num_classes):
            mask = (y == cls)
            per_class_total[cls] += mask.sum().item()
            per_class_correct[cls] += ((pred == y) & mask).sum().item()

    acc = total_correct / max(1, total)
    loss = running_loss / max(1, total)

    per_class_acc = []
    for cls in range(num_classes):
        if per_class_total[cls] == 0:
            per_class_acc.append(0.0)
        else:
            per_class_acc.append(per_class_correct[cls] / per_class_total[cls])

    macro_acc = float(np.mean(per_class_acc))
    return loss, acc, macro_acc, per_class_acc


def main():
    seed_everything(RANDOM_SEED)
    ensure_dir(OUT_DIR)

    raw_id_to_name = load_label_map_yaml(LABEL_MAP_YAML)

    raw_to_final_id, final_id_to_name, final_name_to_id = build_label_space(
        raw_id_to_name=raw_id_to_name,
        merge_skull_into_bone=MERGE_SKULL_INTO_BONE,
        merge_spleen2_into_spleen=MERGE_SPLEEN2_INTO_SPLEEN,
        drop_label_names=DROP_LABEL_NAMES,
    )

    num_labels = len(final_id_to_name)
    print(f"[INFO] num_labels = {num_labels}")
    print("[INFO] final label space:")
    for cid in sorted(final_id_to_name):
        print(f"  {cid}: {final_id_to_name[cid]}")

    train_ids = read_split_ids(TRAIN_SPLIT_CSV)
    val_ids = read_split_ids(VAL_SPLIT_CSV)

    print(f"[INFO] train tiles: {len(train_ids):,}")
    print(f"[INFO] val tiles:   {len(val_ids):,}")

    train_records, train_stats = build_records_from_split(
        train_ids, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, raw_to_final_id, final_name_to_id
    )
    val_records, val_stats = build_records_from_split(
        val_ids, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, raw_to_final_id, final_name_to_id
    )

    if VAL_MAX_SAMPLES is not None and len(val_records) > VAL_MAX_SAMPLES:
        rng = random.Random(RANDOM_SEED)
        rng.shuffle(val_records)
        val_records = val_records[:VAL_MAX_SAMPLES]

    print("\n[INFO] indexing stats:")
    print("train:", train_stats)
    print("val:  ", val_stats)

    if len(train_records) == 0:
        raise RuntimeError("No training records found.")
    if len(val_records) == 0:
        raise RuntimeError("No validation records found.")

    print_class_distribution("train", train_records, final_id_to_name)
    print_class_distribution("val", val_records, final_id_to_name)

    processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=False)

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        id2label={i: final_id_to_name[i] for i in range(num_labels)},
        label2id={final_id_to_name[i]: i for i in range(num_labels)},
        use_safetensors=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"\n[INFO] device: {device}")

    ds_train = CellTilePatchDataset(train_records, processor, PATCH_SIZE)
    ds_val = CellTilePatchDataset(val_records, processor, PATCH_SIZE)

    y_train = [r.label_id for r in train_records]
    class_counts = Counter(y_train)
    sample_weights = torch.tensor([1.0 / class_counts[y] for y in y_train], dtype=torch.double)

    if TRAIN_SAMPLES_PER_EPOCH is None:
        num_samples_epoch = len(train_records)
    else:
        num_samples_epoch = int(TRAIN_SAMPLES_PER_EPOCH)

    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples_epoch,
        replacement=True,
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=False,
        num_workers=NUM_WORKERS_TRAIN,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS_TRAIN > 0),
    )

    dl_val = DataLoader(
        ds_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS_VAL,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS_VAL > 0),
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and device == "cuda"))

    best_macro = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    history = []

    save_label_map = {
        "num_labels": num_labels,
        "id2label": {int(k): v for k, v in final_id_to_name.items()},
        "merge_skull_into_bone": MERGE_SKULL_INTO_BONE,
        "merge_spleen2_into_spleen": MERGE_SPLEEN2_INTO_SPLEEN,
        "drop_label_names": sorted(list(DROP_LABEL_NAMES)),
    }
    with open(OUT_DIR / "label_space.json", "w", encoding="utf-8") as f:
        json.dump(save_label_map, f, indent=2)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()

        running_loss = 0.0
        total_seen = 0

        t0 = time.time()
        pbar = tqdm(dl_train, desc=f"Epoch {epoch:02d} train", dynamic_ncols=True)

        for step, (x, y) in enumerate(pbar, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(USE_AMP and device == "cuda")):
                logits = model(pixel_values=x).logits
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * y.size(0)
            total_seen += y.size(0)

            if step % LOG_EVERY_STEPS == 0:
                imgs_s = total_seen / max(1e-6, time.time() - t0)
                pbar.set_postfix(loss=f"{loss.item():.4f}", imgs_s=f"{imgs_s:.1f}")

        train_loss = running_loss / max(1, total_seen)

        val_loss, val_acc, val_macro, per_class_acc = evaluate(
            model=model,
            loader=dl_val,
            device=device,
            num_classes=num_labels,
        )

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_macro_acc": float(val_macro),
        }
        history.append(row)

        print(
            f"\n[RESULT] epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_macro_acc={val_macro:.4f}"
        )

        print("[PER-CLASS VAL ACC]")
        for cid, acc in enumerate(per_class_acc):
            print(f"  {cid:2d} | {final_id_to_name[cid]:12s} | {acc:.4f}")

        pd.DataFrame(history).to_csv(OUT_DIR / "history.csv", index=False)

        epoch_dir = OUT_DIR / f"epoch_{epoch:02d}"
        ensure_dir(epoch_dir)
        model.save_pretrained(epoch_dir)
        processor.save_pretrained(epoch_dir)

        if val_macro > best_macro:
            best_macro = val_macro
            best_epoch = epoch
            epochs_without_improvement = 0

            best_dir = OUT_DIR / "best"
            ensure_dir(best_dir)
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)

            with open(best_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "val_macro_acc": val_macro,
                        "per_class_val_acc": {
                            final_id_to_name[i]: float(per_class_acc[i]) for i in range(num_labels)
                        },
                    },
                    f,
                    indent=2,
                )

            print(f"[SAVE] new best model at epoch {epoch} -> {best_dir}")
        else:
            epochs_without_improvement += 1
            print(f"[INFO] no improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"[STOP] early stopping triggered at epoch {epoch}")
            break

    print("\n[DONE]")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val macro acc: {best_macro:.4f}")
    print(f"Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()