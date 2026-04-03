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

    out_dir = Path(ckpt_cfg["out_dir"])
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
    train_ds = StardistMultitaskTileDataset(
        Path(data["train_images_dir"]),
        Path(data["train_labels_dir"]),
        n_rays=int(m_cfg["n_rays"]),
        patch_size=ps,
        class_to_idx=class_to_idx if class_to_idx else None,
    )
    val_ds = StardistMultitaskTileDataset(
        Path(data["val_images_dir"]),
        Path(data["val_labels_dir"]),
        n_rays=int(m_cfg["n_rays"]),
        patch_size=ps,
        class_to_idx=class_to_idx if class_to_idx else None,
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

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        if epoch <= freeze_epochs:
            set_backbone_trainable(model, False)
        else:
            set_backbone_trainable(model, True)

        model.train()
        pbar = tqdm(train_loader, desc=f"train e{epoch}")
        for batch in pbar:
            x = batch["image"].to(device, non_blocking=True)
            prob_t = batch["prob"].to(device, non_blocking=True)
            dist_t = batch["dist"].to(device, non_blocking=True)
            cls_t = batch["cls"].to(device, non_blocking=True)
            fg = batch["fg"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                prob_logit, dist_p, cls_log = model(x)
                loss, parts = multitask_loss(
                    prob_logit,
                    dist_p,
                    cls_log,
                    prob_t,
                    dist_t,
                    cls_t,
                    fg,
                    w_prob=float(tr_cfg["loss_w_prob"]),
                    w_dist=float(tr_cfg["loss_w_dist"]),
                    w_cls=float(tr_cfg["loss_w_cls"]),
                )

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            pbar.set_postfix(loss=f'{parts["loss"]:.4f}', bce=f'{parts["bce"]:.4f}', cls=f'{parts["cls"]:.4f}')

        model.eval()
        v_sum = 0.0
        v_n = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device, non_blocking=True)
                prob_t = batch["prob"].to(device, non_blocking=True)
                dist_t = batch["dist"].to(device, non_blocking=True)
                cls_t = batch["cls"].to(device, non_blocking=True)
                fg = batch["fg"].to(device, non_blocking=True)
                prob_logit, dist_p, cls_log = model(x)
                loss, parts = multitask_loss(
                    prob_logit,
                    dist_p,
                    cls_log,
                    prob_t,
                    dist_t,
                    cls_t,
                    fg,
                    w_prob=float(tr_cfg["loss_w_prob"]),
                    w_dist=float(tr_cfg["loss_w_dist"]),
                    w_cls=float(tr_cfg["loss_w_cls"]) if w_cls > 1e-8 else 0.0,
                )
                v_sum += parts["loss"] * x.shape[0]
                v_n += x.shape[0]
        val_loss = v_sum / max(v_n, 1)
        print(f"epoch {epoch} val_loss={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_dir / "best.pt")

        if epoch % int(ckpt_cfg.get("save_every", 1)) == 0:
            torch.save(model.state_dict(), out_dir / f"epoch_{epoch:03d}.pt")


if __name__ == "__main__":
    main()
