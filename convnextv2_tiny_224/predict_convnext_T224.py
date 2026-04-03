#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Full-slide nucleus reclassification for GS40 fetal ConvNeXt model.

Supports BOTH input GeoJSON styles:
1) FeatureCollection:
   {"type":"FeatureCollection","features":[ ... ]}
2) Top-level list of features:
   [ {...}, {...}, ... ]

For each nucleus:
- compute centroid from polygon
- crop patch from NDPI
- run ConvNeXt classification
- write prediction into GeoJSON properties

Checkpoint:
- pass the checkpoint DIRECTORY (.../best), not model.safetensors directly

Label map:
- supports either label_space.json or label_map.yaml
"""

import argparse
import json
import time
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

import ijson
import torch
import openslide
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification


# ------------------------------------------------------------
# Optional colors for QuPath display
# ------------------------------------------------------------
CLASS_COLORS = {
    "bladder":     [230, 25, 75],
    "bone":        [245, 130, 48],
    "brain":       [255, 225, 25],
    "collagen":    [210, 245, 60],
    "ear":         [60, 180, 75],
    "eye":         [70, 240, 240],
    "gi":          [0, 130, 200],
    "heart":       [0, 0, 128],
    "kidney":      [145, 30, 180],
    "liver":       [240, 50, 230],
    "lungs":       [128, 128, 128],
    "mesokidney":  [170, 110, 40],
    "nontissue":   [255, 255, 255],
    "pancreas":    [128, 0, 0],
    "skull":       [170, 255, 195],
    "spleen":      [128, 128, 0],
    "spleen2":     [255, 215, 180],
    "thymus":      [0, 0, 0],
    "thyroid":     [250, 190, 190],
}


def normalize_name(s: str) -> str:
    return str(s).strip().lower()


def _json_default(o):
    if isinstance(o, Decimal):
        return float(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


# ---------- geometry helpers ----------
def centroid_of_ring_xy(ring):
    if not ring:
        return None

    def fpt(p):
        return float(p[0]), float(p[1])

    if len(ring) < 3:
        xs, ys = [], []
        for p in ring:
            x, y = fpt(p)
            xs.append(x)
            ys.append(y)
        return (sum(xs) / max(1, len(xs)), sum(ys) / max(1, len(ys)))

    if ring[0] == ring[-1]:
        ring = ring[:-1]

    A = 0.0
    Cx = 0.0
    Cy = 0.0
    n = len(ring)

    for i in range(n):
        x0, y0 = fpt(ring[i])
        x1, y1 = fpt(ring[(i + 1) % n])
        cross = x0 * y1 - x1 * y0
        A += cross
        Cx += (x0 + x1) * cross
        Cy += (y0 + y1) * cross

    A *= 0.5
    if abs(A) < 1e-9:
        xs, ys = [], []
        for p in ring:
            x, y = fpt(p)
            xs.append(x)
            ys.append(y)
        return (sum(xs) / n, sum(ys) / n)

    Cx /= (6.0 * A)
    Cy /= (6.0 * A)
    return (Cx, Cy)


def centroid_from_geom(geom):
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if not coords:
        return None
    if gtype == "Polygon":
        return centroid_of_ring_xy(coords[0])
    if gtype == "MultiPolygon":
        return centroid_of_ring_xy(coords[0][0])
    return None


# ---------- IO helpers ----------
def open_chunk_writer(out_dir: Path, slide_stem: str, part_idx: int):
    out_path = out_dir / f"{slide_stem}__pred_part{part_idx:03d}.geojson"
    f = out_path.open("w", encoding="utf-8")
    f.write('{"type":"FeatureCollection","features":[\n')
    return f, out_path


def close_chunk_writer(f):
    f.write("\n]}\n")
    f.close()


def iter_geojson_features(geojson_path: Path):
    """
    Supports both:
    1) {"type":"FeatureCollection","features":[...]}
    2) [{...}, {...}, ...]
    """
    with geojson_path.open("rb") as f:
        first_char = None
        while True:
            ch = f.read(1)
            if not ch:
                break
            if ch in b" \t\r\n":
                continue
            first_char = ch
            break

        if first_char is None:
            return

        f.seek(0)

        if first_char == b"{":
            yield from ijson.items(f, "features.item")
        elif first_char == b"[":
            yield from ijson.items(f, "item")
        else:
            raise RuntimeError(f"Unsupported geojson format: {geojson_path}")


def count_features_fast(geojson_path: Path) -> Optional[int]:
    try:
        n = 0
        for _ in iter_geojson_features(geojson_path):
            n += 1
        return n
    except Exception:
        return None


def load_label_space(path: Path):
    """
    Supports either:
    1) label_space.json with {"id2label": {...}}
    2) label_map.yaml with:
       labels:
         0: bone
         1: brain
         ...
    """
    suffix = path.suffix.lower()

    if suffix in [".yaml", ".yml"]:
        import yaml
        d = yaml.safe_load(path.read_text(encoding="utf-8"))
        labels = d.get("labels", {})
        id2label = {int(k): str(v) for k, v in labels.items()}
        return d, id2label

    d = json.loads(path.read_text(encoding="utf-8"))
    if "id2label" in d:
        id2label_raw = d.get("id2label", {})
        id2label = {int(k): str(v) for k, v in id2label_raw.items()}
    else:
        id2label = {int(k): str(v) for k, v in d.items()}
    return d, id2label


def get_model_id2label(model) -> Dict[int, str]:
    raw = getattr(model.config, "id2label", {})
    out = {}
    for k, v in raw.items():
        try:
            out[int(k)] = str(v)
        except Exception:
            pass
    return out


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndpi", required=True, help="Path to NDPI")
    ap.add_argument("--geojson", required=True, help="Input GeoJSON")
    ap.add_argument("--ckpt", required=True, help="Checkpoint directory (.../best)")
    ap.add_argument("--label_space", required=True, help="Path to label_space.json or label_map.yaml")
    ap.add_argument("--out_dir", required=True, help="Output dir for predicted GeoJSON chunks")

    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=50000, help="Features per output geojson file")
    ap.add_argument("--coord_scale", type=float, default=1.0)
    ap.add_argument("--swap_xy", action="store_true")
    ap.add_argument("--gt_key", type=str, default="class_id", help="Optional GT property key")
    ap.add_argument("--local_slide_dir", type=str, default="", help="If set, use local copy by basename")
    ap.add_argument("--max_n", type=int, default=0, help="If >0, stop after this many nuclei")
    ap.add_argument("--count_first", action="store_true", help="Extra pass to count total features")
    args = ap.parse_args()

    ndpi_path = Path(args.ndpi)
    geojson_path = Path(args.geojson)
    ckpt_dir = Path(args.ckpt)
    label_space_path = Path(args.label_space)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_dir.is_dir():
        raise ValueError(f"--ckpt must be a checkpoint directory, got: {ckpt_dir}")

    slide_stem = ndpi_path.stem

    if args.local_slide_dir:
        local_candidate = Path(args.local_slide_dir) / ndpi_path.name
        if local_candidate.exists():
            ndpi_path = local_candidate

    _, label_space_id2label = load_label_space(label_space_path)

    processor = AutoImageProcessor.from_pretrained(ckpt_dir, use_fast=False)
    model = AutoModelForImageClassification.from_pretrained(
        ckpt_dir,
        use_safetensors=True,
    ).eval()

    model_id2label = get_model_id2label(model)
    id2label = model_id2label if model_id2label else label_space_id2label
    if not id2label:
        raise RuntimeError("Could not recover id2label from checkpoint or label file")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    slide = openslide.OpenSlide(str(ndpi_path))

    patch = args.patch
    half = patch // 2
    coord_scale = args.coord_scale
    swap_xy = args.swap_xy

    print(f"[INFO] device: {device}")
    print(f"[INFO] checkpoint: {ckpt_dir}")
    print(f"[INFO] slide: {ndpi_path}")
    print(f"[INFO] geojson: {geojson_path}")
    print(f"[INFO] patch: {patch}")
    print(f"[INFO] batch: {args.batch}")
    print(f"[INFO] num_labels: {len(id2label)}")
    print("[INFO] classes:")
    for k in sorted(id2label):
        print(f"  {k}: {id2label[k]}")

    part_idx = 0
    f_out, out_path = open_chunk_writer(out_dir, slide_stem, part_idx)
    wrote_in_chunk = 0
    wrote_total = 0
    first_in_chunk = True

    imgs: List = []
    feats_buf: List[dict] = []
    centers_buf: List[tuple] = []

    total = None
    if args.count_first:
        print("[INFO] counting features (extra pass)...")
        total = count_features_fast(geojson_path)
        print(f"[INFO] feature count = {total:,}" if total is not None else "[WARN] could not count features")

    t0 = time.time()
    pbar = tqdm(total=total, desc="predict nuclei", dynamic_ncols=True)

    for feat in iter_geojson_features(geojson_path):
        geom = feat.get("geometry", None)
        if geom is None:
            pbar.update(1)
            continue

        c = centroid_from_geom(geom)
        if c is None:
            pbar.update(1)
            continue

        cx, cy = c
        x = float(cx) * coord_scale
        y = float(cy) * coord_scale
        if swap_xy:
            x, y = y, x

        x0 = int(round(x - half))
        y0 = int(round(y - half))

        rgba = slide.read_region((x0, y0), 0, (patch, patch))
        imgs.append(rgba.convert("RGB"))
        feats_buf.append(feat)
        centers_buf.append((float(cx), float(cy)))

        pbar.update(1)

        if len(imgs) == args.batch:
            batch = processor(images=imgs, return_tensors="pt")["pixel_values"].to(device)
            logits = model(pixel_values=batch).logits
            pred_ids = logits.argmax(dim=1).cpu().numpy().tolist()
            probs = torch.softmax(logits, dim=1).max(dim=1).values.cpu().numpy().tolist()

            for feat_i, pred_id, conf, (cent_x, cent_y) in zip(feats_buf, pred_ids, probs, centers_buf):
                pred_id = int(pred_id)
                pred_name = id2label.get(pred_id, f"class_{pred_id}")
                pred_name_norm = normalize_name(pred_name)
                pred_rgb = CLASS_COLORS.get(pred_name_norm, [255, 255, 255])

                props = feat_i.setdefault("properties", {})
                gt = props.get(args.gt_key, None)

                props["class_id_gt"] = int(gt) if gt is not None else None
                props["class_id_pred"] = pred_id
                props["class_name_pred"] = pred_name
                props["pred_conf"] = float(conf)
                props["centroid_x"] = float(cent_x)
                props["centroid_y"] = float(cent_y)
                props["classification"] = {
                    "name": pred_name,
                    "color": pred_rgb
                }

                if not first_in_chunk:
                    f_out.write(",\n")
                first_in_chunk = False
                f_out.write(json.dumps(feat_i, ensure_ascii=False, default=_json_default))

                wrote_in_chunk += 1
                wrote_total += 1

                if args.max_n and wrote_total >= args.max_n:
                    imgs = []
                    feats_buf = []
                    centers_buf = []
                    close_chunk_writer(f_out)
                    pbar.close()
                    print(f"[WROTE] {out_path} | n={wrote_in_chunk:,} | total={wrote_total:,}")
                    print(f"[DONE EARLY] output folder: {out_dir}")
                    return

                if wrote_in_chunk >= args.chunk:
                    close_chunk_writer(f_out)
                    print(f"[WROTE] {out_path} | n={wrote_in_chunk:,} | total={wrote_total:,}")
                    part_idx += 1
                    f_out, out_path = open_chunk_writer(out_dir, slide_stem, part_idx)
                    wrote_in_chunk = 0
                    first_in_chunk = True

            imgs = []
            feats_buf = []
            centers_buf = []

            if wrote_total and (wrote_total % (args.chunk * 2) == 0):
                secs = max(1e-6, time.time() - t0)
                pbar.set_postfix(exported=f"{wrote_total:,}", nuc_s=f"{wrote_total / secs:.1f}")

    if imgs:
        batch = processor(images=imgs, return_tensors="pt")["pixel_values"].to(device)
        #logits = model(pixel_values=batch).logits
        #pred_ids = logits.argmax(dim=1).cpu().numpy().tolist()
        #probs = torch.softmax(logits, dim=1).max(dim=1).values.cpu().numpy().tolist()

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits = model(pixel_values=batch).logits

        pred_ids = logits.argmax(dim=1).cpu().numpy().tolist()
        probs = torch.softmax(logits, dim=1).max(dim=1).values.cpu().numpy().tolist()

        for feat_i, pred_id, conf, (cent_x, cent_y) in zip(feats_buf, pred_ids, probs, centers_buf):
            pred_id = int(pred_id)
            pred_name = id2label.get(pred_id, f"class_{pred_id}")
            pred_name_norm = normalize_name(pred_name)
            pred_rgb = CLASS_COLORS.get(pred_name_norm, [255, 255, 255])

            props = feat_i.setdefault("properties", {})
            gt = props.get(args.gt_key, None)

            props["class_id_gt"] = int(gt) if gt is not None else None
            props["class_id_pred"] = pred_id
            props["class_name_pred"] = pred_name
            props["pred_conf"] = float(conf)
            props["centroid_x"] = float(cent_x)
            props["centroid_y"] = float(cent_y)
            props["classification"] = {
                "name": pred_name,
                "color": pred_rgb
            }

            if not first_in_chunk:
                f_out.write(",\n")
            first_in_chunk = False
            f_out.write(json.dumps(feat_i, ensure_ascii=False, default=_json_default))

            wrote_in_chunk += 1
            wrote_total += 1

            if args.max_n and wrote_total >= args.max_n:
                close_chunk_writer(f_out)
                pbar.close()
                print(f"[WROTE] {out_path} | n={wrote_in_chunk:,} | total={wrote_total:,}")
                print(f"[DONE EARLY] output folder: {out_dir}")
                return

    close_chunk_writer(f_out)
    pbar.close()
    print(f"[WROTE] {out_path} | n={wrote_in_chunk:,} | total={wrote_total:,}")
    print(f"[DONE] output folder: {out_dir}")
    print("Import the slide__pred_partXXX.geojson files into QuPath.")
    print("Predictions are stored in properties['classification'], ['class_name_pred'], ['class_id_pred'].")


if __name__ == "__main__":
    main()