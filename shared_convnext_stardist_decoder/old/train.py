from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import StardistMultitaskTileDataset, build_class_to_idx_from_dir
from .losses import multitask_loss
from .model import build_model


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_backbone_trainable(model: torch.nn.Module, trainable: bool) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = trainable


class _RunningMean:
    """Lightweight online mean — no extra dependencies."""
    def __init__(self) -> None:
        self._sum = 0.0
        self._n = 0

    def update(self, val: float, n: int = 1) -> None:
        self._sum += val * n
        self._n += n

    @property
    def mean(self) -> float:
        return self._sum / self._n if self._n else 0.0

    def reset(self) -> None:
        self._sum = 0.0
        self._n = 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Train shared ConvNeXt + StarDist-multitask decoder.")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = cfg["data"]
    tr_cfg = cfg["train"]
    ckpt_cfg = cfg["checkpoint"]
    m_cfg = cfg["model"]

    cn = m_cfg.get("class_names")
    if cn:
        class_to_idx = {str(n).strip().lower(): i for i, n in enumerate(cn)}
    else:
        class_to_idx = build_class_to_idx_from_dir(Path(data["train_labels_dir"]))

    n_cls = int(m_cfg["num_classes"])
    if cn and len(cn) != n_cls:
        raise ValueError("model.num_classes must equal len(model.class_names)")
    if not cn and class_to_idx and len(class_to_idx) != n_cls:
        raise ValueError("model.num_classes must match scanned inst2class names when class_names is omitted")

    out_dir = Path(ckpt_cfg["out_dir"]) / cfg.get("experiment_name", "experiment_run")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    idx2label = {i: n for n, i in class_to_idx.items()} if class_to_idx else {}
    (out_dir / "idx2label.json").write_text(json.dumps(idx2label, indent=2), encoding="utf-8")

    w_cls = float(tr_cfg["loss_w_cls"])
    if w_cls > 1e-8 and not class_to_idx:
        raise ValueError("Provide model.class_names or *_inst2class.json sidecars when loss_w_cls > 0")

    model = build_model(cfg).to(device)
    if args.resume and args.resume.is_file():
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Loaded weights from {args.resume}")

    ps = int(tr_cfg["patch_size"])
    cache_to_ram = bool(data.get("cache_to_ram", False))
    train_ds = StardistMultitaskTileDataset(
        Path(data["train_images_dir"]),
        Path(data["train_labels_dir"]),
        n_rays=int(m_cfg["n_rays"]),
        patch_size=ps,
        class_to_idx=class_to_idx if class_to_idx else None,
        stems=data.get("train_stems"),
        cache_to_ram=cache_to_ram,
    )
    val_ds = StardistMultitaskTileDataset(
        Path(data["val_images_dir"]),
        Path(data["val_labels_dir"]),
        n_rays=int(m_cfg["n_rays"]),
        patch_size=ps,
        class_to_idx=class_to_idx if class_to_idx else None,
        stems=data.get("val_stems"),
        cache_to_ram=cache_to_ram,
    )

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

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tr_cfg["lr"]),
        weight_decay=float(tr_cfg["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and bool(tr_cfg.get("amp", True)))
    epochs = int(tr_cfg["epochs"])
    freeze_epochs = int(tr_cfg.get("freeze_backbone_epochs", 0))

    # Cosine annealing: decays LR from initial to eta_min over the unfrozen epochs.
    # Frozen epochs use a constant LR (scheduler steps only after unfreeze).
    unfrozen_epochs = max(epochs - freeze_epochs, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=unfrozen_epochs,
        eta_min=float(tr_cfg.get("lr_min", 1e-6)),
    )

    # CSV log file next to checkpoints
    log_path = out_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text(
            "epoch,lr,train_loss,train_bce,train_dist,train_cls,"
            "val_loss,val_bce,val_dist,val_cls\n",
            encoding="utf-8",
        )

    best_val = float("inf")
    logged_cls_density = False

    rm_loss = _RunningMean()
    rm_bce  = _RunningMean()
    rm_dist = _RunningMean()
    rm_cls  = _RunningMean()

    for epoch in range(1, epochs + 1):
        backbone_frozen = epoch <= freeze_epochs
        set_backbone_trainable(model, not backbone_frozen)

        model.train()
        rm_loss.reset(); rm_bce.reset(); rm_dist.reset(); rm_cls.reset()
        pbar = tqdm(train_loader, desc=f"train e{epoch}")

        for batch in pbar:
            x      = batch["image"].to(device, non_blocking=True)
            prob_t = batch["prob"].to(device, non_blocking=True)
            dist_t = batch["dist"].to(device, non_blocking=True)
            cls_t  = batch["cls"].to(device, non_blocking=True)
            fg     = batch["fg"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                prob_logit, dist_p, cls_log = model(x)
                loss, parts = multitask_loss(
                    prob_logit, dist_p, cls_log,
                    prob_t, dist_t, cls_t, fg,
                    w_prob=float(tr_cfg["loss_w_prob"]),
                    w_dist=float(tr_cfg["loss_w_dist"]),
                    w_cls=float(tr_cfg["loss_w_cls"]),
                )

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            bs = x.shape[0]
            rm_loss.update(parts["loss"], bs)
            rm_bce.update(parts["bce"],  bs)
            rm_dist.update(parts["dist"], bs)
            rm_cls.update(parts["cls"],  bs)

            if w_cls > 1e-8 and epoch == 1 and not logged_cls_density:
                density = (cls_t >= 0).float().mean().item()
                logged_cls_density = True
                print(
                    f"Classification targets: {100.0 * density:.1f}% of pixels labeled (first train batch). "
                    "If ~0%%, *_inst2class.json files are missing, empty, or IDs/names do not match model.class_names."
                )

            pbar.set_postfix(
                loss=f"{rm_loss.mean:.4f}",
                bce=f"{rm_bce.mean:.4f}",
                dist=f"{rm_dist.mean:.4f}",
                cls=f"{rm_cls.mean:.4f}",
            )

        # Step scheduler only after backbone is unfrozen (or from first unfrozen epoch)
        if not backbone_frozen:
            scheduler.step()

        current_lr = opt.param_groups[0]["lr"]

        # --- Validation ---
        model.eval()
        vr_loss = _RunningMean(); vr_bce = _RunningMean()
        vr_dist = _RunningMean(); vr_cls = _RunningMean()

        with torch.no_grad():
            for batch in val_loader:
                x      = batch["image"].to(device, non_blocking=True)
                prob_t = batch["prob"].to(device, non_blocking=True)
                dist_t = batch["dist"].to(device, non_blocking=True)
                cls_t  = batch["cls"].to(device, non_blocking=True)
                fg     = batch["fg"].to(device, non_blocking=True)
                prob_logit, dist_p, cls_log = model(x)
                _, parts = multitask_loss(
                    prob_logit, dist_p, cls_log,
                    prob_t, dist_t, cls_t, fg,
                    w_prob=float(tr_cfg["loss_w_prob"]),
                    w_dist=float(tr_cfg["loss_w_dist"]),
                    w_cls=float(tr_cfg["loss_w_cls"]) if w_cls > 1e-8 else 0.0,
                )
                bs = x.shape[0]
                vr_loss.update(parts["loss"], bs)
                vr_bce.update(parts["bce"],  bs)
                vr_dist.update(parts["dist"], bs)
                vr_cls.update(parts["cls"],  bs)

        val_loss = vr_loss.mean
        frozen_tag = " [backbone frozen]" if backbone_frozen else ""
        print(
            f"epoch {epoch:03d}{frozen_tag}  lr={current_lr:.2e}"
            f"  train  loss={rm_loss.mean:.4f}  bce={rm_bce.mean:.4f}"
            f"  dist={rm_dist.mean:.4f}  cls={rm_cls.mean:.4f}"
            f"  |  val  loss={vr_loss.mean:.4f}  bce={vr_bce.mean:.4f}"
            f"  dist={vr_dist.mean:.4f}  cls={vr_cls.mean:.4f}"
        )

        with log_path.open("a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{current_lr:.6e},"
                f"{rm_loss.mean:.6f},{rm_bce.mean:.6f},{rm_dist.mean:.6f},{rm_cls.mean:.6f},"
                f"{vr_loss.mean:.6f},{vr_bce.mean:.6f},{vr_dist.mean:.6f},{vr_cls.mean:.6f}\n"
            )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_dir / "best.pt")
            print(f"  -> new best val_loss={best_val:.5f}  checkpoint saved.")

        if epoch % int(ckpt_cfg.get("save_every", 1)) == 0:
            torch.save(model.state_dict(), out_dir / f"epoch_{epoch:03d}.pt")


if __name__ == "__main__":
    main()
