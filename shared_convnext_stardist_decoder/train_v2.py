"""
train_v2.py  —  Training script for StardistMultitaskNetV2

Wires together all v2 improvements:
  Fix #1 — model_v2: stage-4 semantic skip to cls head
  Fix #2 — losses_v2: instance-level CE loss (inst_map in batch)
  Fix #3 — losses_v2 + dataset_v2: class-weighted CE, w_cls=2.0
  Fix #4 — dataset_v2: cls_only flag + compute_class_weights

New / changed config keys  (add to your .yaml under each section):
  model:
    cls_semantic_dim: 64     # semantic skip channels; 0 disables the skip

  train:
    loss_w_cls:  2.0         # was 0.5
    loss_w_inst: 0.5         # instance-level CE weight (0 to disable)
    class_weights: auto      # "auto" | null | [w0,w1,...,w18]
    cls_only: false          # if true, skip tiles without cls supervision

Usage:
  python -m shared_convnext_stardist_decoder.train_v2 --config path/to/config.yaml

  Warm-start from a V1 checkpoint (seg decoder transfers; cls head re-inits):
    --resume path/to/best.pt --resume_strict false
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .dataset_v2 import (
    StardistMultitaskTileDatasetV2,
    build_class_to_idx_from_dir,
    compute_class_weights,
)
from .losses_v2 import multitask_loss_v2
from .model_v2 import build_model_v2, load_v1_weights_into_v2


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_backbone_trainable(model: torch.nn.Module, trainable: bool) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = trainable


class _RunningMean:
    def __init__(self) -> None:
        self._sum = 0.0; self._n = 0

    def update(self, val: float, n: int = 1) -> None:
        self._sum += val * n; self._n += n

    @property
    def mean(self) -> float:
        return self._sum / self._n if self._n else 0.0

    def reset(self) -> None:
        self._sum = 0.0; self._n = 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train StardistMultitaskNetV2 (semantic-skip cls + instance CE + class weights)."
    )
    ap.add_argument("--config",        type=Path, required=True)
    ap.add_argument("--resume",        type=Path, default=None,
                    help="Checkpoint to resume from (.pt).")
    ap.add_argument("--resume_strict", type=lambda x: x.lower() != "false",
                    default=True,
                    help="strict=False allows warm-starting from a V1 checkpoint.")
    args = ap.parse_args()

    cfg      = load_config(args.config)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data     = cfg["data"]
    tr_cfg   = cfg["train"]
    ckpt_cfg = cfg["checkpoint"]
    m_cfg    = cfg["model"]

    # ── Class index mapping ───────────────────────────────────────────────────
    cn = m_cfg.get("class_names")
    if cn:
        class_to_idx = {str(n).strip().lower(): i for i, n in enumerate(cn)}
    else:
        class_to_idx = build_class_to_idx_from_dir(Path(data["train_labels_dir"]))

    n_cls = int(m_cfg["num_classes"])
    if cn and len(cn) != n_cls:
        raise ValueError("model.num_classes must equal len(model.class_names)")

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = Path(ckpt_cfg["out_dir"]) / cfg.get("experiment_name", "experiment_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    idx2label = {i: n for n, i in class_to_idx.items()} if class_to_idx else {}
    (out_dir / "idx2label.json").write_text(json.dumps(idx2label, indent=2), encoding="utf-8")

    # ── Loss weights ──────────────────────────────────────────────────────────
    w_cls  = float(tr_cfg.get("loss_w_cls",  2.0))
    w_inst = float(tr_cfg.get("loss_w_inst", 0.5))

    if w_cls > 1e-8 and not class_to_idx:
        raise ValueError("Provide model.class_names or inst2class sidecars when loss_w_cls > 0")

    # ── Class weights (fix #3) ────────────────────────────────────────────────
    cw_cfg = tr_cfg.get("class_weights", "auto")
    class_weights: torch.Tensor | None = None
    if cw_cfg == "auto" and class_to_idx:
        print("Computing class weights from training labels …")
        class_weights = compute_class_weights(
            Path(data["train_labels_dir"]), class_to_idx, mode="inv_sqrt_freq"
        ).to(device)
    elif isinstance(cw_cfg, list) and len(cw_cfg) == n_cls:
        class_weights = torch.tensor(cw_cfg, dtype=torch.float32, device=device)
        print(f"Using manually specified class weights: {class_weights.tolist()}")
    else:
        print("Class weights: uniform (disabled)")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model_v2(cfg).to(device)
    if args.resume and args.resume.is_file():
        sd = torch.load(args.resume, map_location=device, weights_only=True)
        if args.resume_strict:
            model.load_state_dict(sd, strict=True)
            print(f"Resumed V2 checkpoint: {args.resume}")
        else:
            load_v1_weights_into_v2(model, sd, device)
            print(f"Warm-started from V1 checkpoint: {args.resume}")

    # ── Datasets (fix #4: cls_only + inst in batch) ───────────────────────────
    ps           = int(tr_cfg["patch_size"])
    cache_to_ram = bool(data.get("cache_to_ram", False))
    cls_only     = bool(tr_cfg.get("cls_only", False))

    train_ds = StardistMultitaskTileDatasetV2(
        Path(data["train_images_dir"]),
        Path(data["train_labels_dir"]),
        n_rays=int(m_cfg["n_rays"]),
        patch_size=ps,
        class_to_idx=class_to_idx if class_to_idx else None,
        stems=data.get("train_stems"),
        cache_to_ram=cache_to_ram,
        cls_only=cls_only,
    )
    val_ds = StardistMultitaskTileDatasetV2(
        Path(data["val_images_dir"]),
        Path(data["val_labels_dir"]),
        n_rays=int(m_cfg["n_rays"]),
        patch_size=ps,
        class_to_idx=class_to_idx if class_to_idx else None,
        stems=data.get("val_stems"),
        cache_to_ram=cache_to_ram,
        cls_only=False,   # always evaluate on full val set
    )

    # Optional: WeightedRandomSampler to over-sample tiles with cls supervision
    # (useful when cls_only=False but coverage is ~74%)
    use_balanced_sampler = bool(tr_cfg.get("cls_balanced_sampler", False))
    if use_balanced_sampler and class_to_idx:
        print("Building cls-balanced sampler …")
        sample_weights = torch.tensor([
            2.0 if train_ds._has_cls_supervision(p.stem) else 1.0
            for p in train_ds.image_paths
        ])
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=int(tr_cfg["batch_size"]),
            sampler=sampler,
            num_workers=int(tr_cfg["num_workers"]),
            pin_memory=device.type == "cuda",
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=int(tr_cfg["batch_size"]),
            shuffle=True,
            num_workers=int(tr_cfg["num_workers"]),
            pin_memory=device.type == "cuda",
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(tr_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(tr_cfg["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    print(f"Train: {len(train_ds):,} tiles  |  Val: {len(val_ds):,} tiles")

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    opt    = torch.optim.AdamW(
        model.parameters(),
        lr=float(tr_cfg["lr"]),
        weight_decay=float(tr_cfg["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and bool(tr_cfg.get("amp", True))
    )
    epochs          = int(tr_cfg["epochs"])
    freeze_epochs   = int(tr_cfg.get("freeze_backbone_epochs", 0))
    unfrozen_epochs = max(epochs - freeze_epochs, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=unfrozen_epochs, eta_min=float(tr_cfg.get("lr_min", 1e-6))
    )

    # ── Log file ──────────────────────────────────────────────────────────────
    log_path = out_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text(
            "epoch,lr,train_loss,train_bce,train_dist,train_cls_pixel,train_cls_inst,"
            "val_loss,val_bce,val_dist,val_cls_pixel,val_cls_inst\n",
            encoding="utf-8",
        )

    best_val  = float("inf")
    logged_cls_density = False

    rm = {k: _RunningMean() for k in ("loss", "bce", "dist", "cls_pixel", "cls_inst")}

    for epoch in range(1, epochs + 1):
        backbone_frozen = epoch <= freeze_epochs
        set_backbone_trainable(model, not backbone_frozen)

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        for r in rm.values():
            r.reset()
        pbar = tqdm(train_loader, desc=f"train e{epoch}")

        for batch in pbar:
            x      = batch["image"].to(device, non_blocking=True)
            prob_t = batch["prob"].to(device, non_blocking=True)
            dist_t = batch["dist"].to(device, non_blocking=True)
            cls_t  = batch["cls"].to(device, non_blocking=True)
            fg     = batch["fg"].to(device, non_blocking=True)
            inst   = batch["inst"].to(device, non_blocking=True)   # NEW

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                prob_logit, dist_p, cls_log = model(x)
                loss, parts = multitask_loss_v2(
                    prob_logit, dist_p, cls_log,
                    prob_t, dist_t, cls_t, fg, inst,
                    w_prob=float(tr_cfg.get("loss_w_prob", 1.0)),
                    w_dist=float(tr_cfg.get("loss_w_dist", 0.05)),
                    w_cls=w_cls,
                    w_inst=w_inst,
                    class_weights=class_weights,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            bs = x.shape[0]
            for k in rm:
                rm[k].update(parts[k], bs)

            # One-shot diagnostic on first batch of epoch 1
            if w_cls > 1e-8 and epoch == 1 and not logged_cls_density:
                density = (cls_t >= 0).float().mean().item()
                n_inst  = (inst > 0).float().mean().item()
                logged_cls_density = True
                print(
                    f"[Epoch 1 first batch]  "
                    f"cls pixels={100*density:.1f}%  "
                    f"fg pixels={100*n_inst:.1f}%  "
                    f"cls_inst={parts['cls_inst']:.4f}"
                )

            pbar.set_postfix(
                loss=f"{rm['loss'].mean:.4f}",
                bce=f"{rm['bce'].mean:.4f}",
                dist=f"{rm['dist'].mean:.4f}",
                cls_px=f"{rm['cls_pixel'].mean:.4f}",
                cls_in=f"{rm['cls_inst'].mean:.4f}",
            )

        if not backbone_frozen:
            scheduler.step()
        current_lr = opt.param_groups[0]["lr"]

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        vm = {k: _RunningMean() for k in ("loss", "bce", "dist", "cls_pixel", "cls_inst")}

        with torch.no_grad():
            for batch in val_loader:
                x      = batch["image"].to(device, non_blocking=True)
                prob_t = batch["prob"].to(device, non_blocking=True)
                dist_t = batch["dist"].to(device, non_blocking=True)
                cls_t  = batch["cls"].to(device, non_blocking=True)
                fg     = batch["fg"].to(device, non_blocking=True)
                inst   = batch["inst"].to(device, non_blocking=True)
                prob_logit, dist_p, cls_log = model(x)
                _, parts = multitask_loss_v2(
                    prob_logit, dist_p, cls_log,
                    prob_t, dist_t, cls_t, fg, inst,
                    w_prob=float(tr_cfg.get("loss_w_prob", 1.0)),
                    w_dist=float(tr_cfg.get("loss_w_dist", 0.05)),
                    w_cls=w_cls,
                    w_inst=w_inst,
                    class_weights=class_weights,
                )
                bs = x.shape[0]
                for k in vm:
                    vm[k].update(parts[k], bs)

        val_loss    = vm["loss"].mean
        frozen_tag  = " [backbone frozen]" if backbone_frozen else ""
        print(
            f"epoch {epoch:03d}{frozen_tag}  lr={current_lr:.2e}"
            f"  train  loss={rm['loss'].mean:.4f}  bce={rm['bce'].mean:.4f}"
            f"  dist={rm['dist'].mean:.4f}"
            f"  cls_px={rm['cls_pixel'].mean:.4f}  cls_in={rm['cls_inst'].mean:.4f}"
            f"  |  val  loss={vm['loss'].mean:.4f}  bce={vm['bce'].mean:.4f}"
            f"  dist={vm['dist'].mean:.4f}"
            f"  cls_px={vm['cls_pixel'].mean:.4f}  cls_in={vm['cls_inst'].mean:.4f}"
        )

        with log_path.open("a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{current_lr:.6e},"
                f"{rm['loss'].mean:.6f},{rm['bce'].mean:.6f},{rm['dist'].mean:.6f},"
                f"{rm['cls_pixel'].mean:.6f},{rm['cls_inst'].mean:.6f},"
                f"{vm['loss'].mean:.6f},{vm['bce'].mean:.6f},{vm['dist'].mean:.6f},"
                f"{vm['cls_pixel'].mean:.6f},{vm['cls_inst'].mean:.6f}\n"
            )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_dir / "best.pt")
            print(f"  -> new best val_loss={best_val:.5f}  checkpoint saved.")

        if epoch % int(ckpt_cfg.get("save_every", 1)) == 0:
            torch.save(model.state_dict(), out_dir / f"epoch_{epoch:03d}.pt")


if __name__ == "__main__":
    main()
