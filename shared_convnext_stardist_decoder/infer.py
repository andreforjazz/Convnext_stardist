from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from .dataset import IMAGENET_MEAN, IMAGENET_STD
from .geometry import dist_at_points, dist_to_coord, local_peaks, polygon_ring_rowcol, vote_class
from .model import build_model


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_chw(arr_hwc: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(arr_hwc.astype(np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def run_model_on_tensor(model: torch.nn.Module, x: torch.Tensor, device: torch.device):
    x = x.unsqueeze(0).to(device)
    prob_logit, dist, cls_logit = model(x)
    prob = torch.sigmoid(prob_logit)[0, 0].float().cpu().numpy()
    dist = dist[0].float().cpu().numpy()
    cls_logit = cls_logit[0].float().cpu().numpy()
    return prob, dist, cls_logit


def predict_image_tiled(
    model: torch.nn.Module,
    rgb: np.ndarray,
    tile: int,
    offsets: list[int],
    *,
    device: torch.device,
    prob_thresh: float,
    nms_dist: int,
    id2label: dict[str, str],
) -> list[dict]:
    """
    rgb: uint8 (H,W,3)
    Returns list of GeoJSON-ready feature dicts (geometry + properties).
    """
    h, w = rgb.shape[:2]
    feats: list[dict] = []
    seen = 0

    for oy in offsets:
        for ox in offsets:
            for y0 in range(oy, h, tile):
                for x0 in range(ox, w, tile):
                    y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                    if y1 - y0 < 8 or x1 - x0 < 8:
                        continue
                    patch = rgb[y0:y1, x0:x1]
                    # pad to multiple of 32
                    ph, pw = patch.shape[0], patch.shape[1]
                    nh = int(np.ceil(ph / 32) * 32)
                    nw = int(np.ceil(pw / 32) * 32)
                    if nh != ph or nw != pw:
                        pad = np.zeros((nh, nw, 3), dtype=np.uint8)
                        pad[:ph, :pw] = patch
                        patch = pad

                    x = _normalize_chw(patch)
                    prob, dist_map, cls_log = run_model_on_tensor(model, x, device)

                    prob = prob[:ph, :pw]
                    dist_map = dist_map[:, :ph, :pw]
                    cls_log = cls_log[:, :ph, :pw]

                    peaks = local_peaks(prob, min_distance=int(nms_dist), thresh=float(prob_thresh))
                    if len(peaks) == 0:
                        continue
                    dists = dist_at_points(np.transpose(dist_map, (1, 2, 0)), peaks)
                    coords = dist_to_coord(dists, peaks.astype(np.float32))
                    for k in range(coords.shape[0]):
                        rc = coords[k]
                        rc_global = rc.copy()
                        rc_global[0] += y0
                        rc_global[1] += x0
                        cls_id, probs = vote_class(cls_log, rc, (ph, pw))
                        ring = polygon_ring_rowcol(rc)
                        ring_global = ring + np.array([x0, y0], dtype=np.float32)
                        name = id2label.get(str(cls_id), f"class_{cls_id}")
                        props = {
                            "classification": {
                                "name": name,
                                "index": int(cls_id),
                            },
                            "prob_peak": float(prob[int(peaks[k, 0]), int(peaks[k, 1])]),
                            "class_probs": {
                                id2label.get(str(i), str(i)): float(p)
                                for i, p in enumerate(probs)
                            },
                        }

                        feats.append(
                            {
                                "type": "Feature",
                                "id": f"{seen}",
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [ring_global.tolist()],
                                },
                                "properties": props,
                            }
                        )
                        seen += 1

    return feats


def write_geojson(features: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write('{"type":"FeatureCollection","features":[\n')
        for i, feat in enumerate(features):
            tail = ",\n" if i + 1 < len(features) else "\n"
            f.write(json.dumps(feat, ensure_ascii=False) + tail)
        f.write("]}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Infer multitask StarDist+ConvNeXt model; write GeoJSON.")
    ap.add_argument("--config", type=Path, required=True, help="Training config yaml (model + infer sections).")
    ap.add_argument("--weights", type=Path, required=True, help="best.pt checkpoint")
    ap.add_argument("--image", type=Path, required=True, help="RGB image (PNG/JPEG) or use with large tiles via crop")
    ap.add_argument("--out", type=Path, required=True, help="output .geojson")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    infer = cfg.get("infer", {})
    m = cfg["model"]

    model = build_model(cfg).to(device)
    sd = torch.load(args.weights, map_location=device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    idx_path = args.weights.parent / "idx2label.json"
    if idx_path.is_file():
        raw = json.loads(idx_path.read_text(encoding="utf-8"))
        id2label = {str(k): v for k, v in raw.items()}
    else:
        cn = m.get("class_names") or []
        id2label = {str(i): n for i, n in enumerate(cn)}

    img = np.asarray(Image.open(args.image).convert("RGB"))
    feats = predict_image_tiled(
        model,
        img,
        tile=int(cfg["train"]["patch_size"]),
        offsets=[int(o) for o in infer.get("sample_offsets", [0])],
        device=device,
        prob_thresh=float(infer.get("prob_thresh", 0.45)),
        nms_dist=int(infer.get("nms_dist", 3)),
        id2label=id2label,
    )
    write_geojson(feats, args.out)
    print(f"Wrote {len(feats)} features to {args.out}")


if __name__ == "__main__":
    main()
