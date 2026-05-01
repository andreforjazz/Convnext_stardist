"""Build a unified multi-cohort training bundle.

Takes the three current source roots (GS40 / GS55 / GS33), each with its own
quirky on-disk layout, and produces a single bundle where every cohort has the
same internal structure:

    <out>/<COHORT>/
        train/{images,labels}/
        val/{images,labels}/
        splits/fold_0/{train,val}.csv

The bundle is what the training notebook + slurm job consume via
`shared_convnext_stardist_decoder.aux_codes.cohorts.cohort_paths`.

GS40 quirks handled here:
- train/ and val/ images currently share `train/images/`; routed by split CSV.
- labels are nested under `stardist_multitask_ready/train_instance_labels/`;
  flattened into `<COHORT>/{train,val}/labels/`.

GS55 / GS33 are already in the target shape; the bundler just copies them
in 1:1 (so the unified contract is satisfied and the output is self-contained).

Usage
-----
    python make_training_dataset/make_bundle.py \\
        --gs40-root  //kittyserverdw/.../GS40/.../dataset_256_40k_48_slides \\
        --gs55-root  //kittyserverdw/.../GS55/.../dataset_256_50k_clsbal_GS55 \\
        --gs33-root  //kittyserverdw/.../GS33/.../dataset_256_30k_GS33 \\
        --out        //kittyserverdw/Andre_kit/data/data_on_demand \\
        [--workers 8] [--dry-run] [--checksum-sample 64]

Idempotent: re-running skips files whose destination already exists with the
same byte length. Use --force to overwrite.

Hard-fails if a stem appears in both train.csv and val.csv (data leak).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

# ────────────────────────────────────────────────────────────────────────────
# Per-cohort source layout. Lookup is the single source of truth for "where do
# I find images / labels / splits in cohort X's source root".
# ────────────────────────────────────────────────────────────────────────────
SOURCE_LAYOUTS: dict[str, dict] = {
    "GS40": {
        # train and val tiles both live in train/images/; route by split CSV.
        "train_images": "train/images",
        "val_images":   "train/images",
        # labels are flat in this nested folder, applies to both splits.
        "train_labels": "stardist_multitask_ready/train_instance_labels",
        "val_labels":   "stardist_multitask_ready/train_instance_labels",
        "train_split":  "splits/fold_0/train.csv",
        "val_split":    "splits/fold_0/val.csv",
    },
    "GS55": {
        "train_images": "train/images",
        "train_labels": "train/labels",
        "val_images":   "val/images",
        "val_labels":   "val/labels",
        "train_split":  "splits/fold_0/train.csv",
        "val_split":    "splits/fold_0/val.csv",
    },
    "GS33": {  # same shape as GS55
        "train_images": "train/images",
        "train_labels": "train/labels",
        "val_images":   "val/images",
        "val_labels":   "val/labels",
        "train_split":  "splits/fold_0/train.csv",
        "val_split":    "splits/fold_0/val.csv",
    },
}

# Image and label filename patterns. Each tile is a triple
# (image .png, instance-mask .png in labels/, inst2class JSON sidecar).
IMG_EXT = ".png"


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def read_stem_csv(p: Path) -> list[str]:
    """Read a split CSV. Accepts either a header row or no header. One stem per row."""
    with p.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    # Strip header if first row looks like one (single non-stem-shaped string).
    first = rows[0][0].strip()
    if first.lower() in {"stem", "name", "id", "tile", "filename"}:
        rows = rows[1:]
    return [r[0].strip() for r in rows if r and r[0].strip()]


def need_copy(src: Path, dst: Path, force: bool) -> bool:
    """True if `dst` is missing, wrong-sized, or `force` is set."""
    if force or not dst.exists():
        return True
    try:
        return src.stat().st_size != dst.stat().st_size
    except OSError:
        return True


def copy_file(src: Path, dst: Path, *, force: bool, dry_run: bool) -> tuple[str, int, str]:
    """Returns (status, bytes_copied, src_path).

    status ∈ {copied, skipped, dryrun, missing}; src_path is included so the
    caller can log which source path was missing without rebuilding it.
    """
    src_str = str(src)
    if not src.exists():
        return ("missing", 0, src_str)
    if not need_copy(src, dst, force):
        return ("skipped", 0, src_str)
    if dry_run:
        return ("dryrun", src.stat().st_size, src_str)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return ("copied", dst.stat().st_size, src_str)


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ────────────────────────────────────────────────────────────────────────────
# Per-cohort bundling
# ────────────────────────────────────────────────────────────────────────────

def stage_cohort(
    name: str,
    src_root: Path,
    out_root: Path,
    *,
    workers: int,
    dry_run: bool,
    force: bool,
    checksum_sample: int,
) -> dict:
    """Bundle one cohort. Returns the manifest entry."""
    layout = SOURCE_LAYOUTS[name]
    train_csv = src_root / layout["train_split"]
    val_csv   = src_root / layout["val_split"]
    if not train_csv.is_file() or not val_csv.is_file():
        raise FileNotFoundError(f"{name}: missing split CSVs at {train_csv} / {val_csv}")

    # Dedupe within each CSV (some pipelines duplicate rows).
    train_stems_raw = read_stem_csv(train_csv)
    val_stems_raw   = read_stem_csv(val_csv)
    train_stems = list(dict.fromkeys(train_stems_raw))
    val_stems   = list(dict.fromkeys(val_stems_raw))
    n_train_dupes = len(train_stems_raw) - len(train_stems)
    n_val_dupes   = len(val_stems_raw)   - len(val_stems)
    if n_train_dupes or n_val_dupes:
        print(f"[{name}] CSV duplicates removed: train={n_train_dupes}, val={n_val_dupes}")

    overlap = set(train_stems) & set(val_stems)
    if overlap:
        raise RuntimeError(
            f"{name}: data leak — {len(overlap)} stems appear in BOTH train and val "
            f"CSVs. First 5: {sorted(overlap)[:5]}"
        )

    cohort_out = out_root / name
    print(f"\n[{name}] source: {src_root}")
    print(f"[{name}] dest:   {cohort_out}")
    print(f"[{name}] stems:  train={len(train_stems):,}  val={len(val_stems):,}")

    # ── Build (src, dst) copy plan for every file ───────────────────────────
    plan: list[tuple[Path, Path]] = []
    for split, stems in (("train", train_stems), ("val", val_stems)):
        img_src_dir = src_root / layout[f"{split}_images"]
        lbl_src_dir = src_root / layout[f"{split}_labels"]
        img_dst_dir = cohort_out / split / "images"
        lbl_dst_dir = cohort_out / split / "labels"
        for stem in stems:
            plan.append((img_src_dir / f"{stem}{IMG_EXT}",            img_dst_dir / f"{stem}{IMG_EXT}"))
            plan.append((lbl_src_dir / f"{stem}{IMG_EXT}",            lbl_dst_dir / f"{stem}{IMG_EXT}"))
            plan.append((lbl_src_dir / f"{stem}_inst2class.json",     lbl_dst_dir / f"{stem}_inst2class.json"))

    # Splits CSVs go alongside the bundle so the layout is self-contained.
    splits_dst = cohort_out / "splits" / "fold_0"
    plan.append((train_csv, splits_dst / "train.csv"))
    plan.append((val_csv,   splits_dst / "val.csv"))

    # ── Execute the plan in a thread pool (SMB copy is I/O-bound) ───────────
    counts = {"copied": 0, "skipped": 0, "dryrun": 0, "missing": 0}
    missing_paths: list[str] = []
    bytes_done = 0
    t0 = time.perf_counter()
    last_print = t0

    def _do(pair):
        return copy_file(pair[0], pair[1], force=force, dry_run=dry_run)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_do, pair) for pair in plan]
        for i, fut in enumerate(as_completed(futures)):
            status, nbytes, src_str = fut.result()
            counts[status] += 1
            bytes_done += nbytes
            if status == "missing":
                missing_paths.append(src_str)
            now = time.perf_counter()
            if now - last_print > 5:  # every 5 s
                done = sum(counts.values())
                print(
                    f"  [{name}] {done:,}/{len(plan):,}  "
                    f"copied={counts['copied']:,} skipped={counts['skipped']:,} "
                    f"missing={counts['missing']:,}  "
                    f"{bytes_done/1e9:.2f} GB  "
                    f"({(now-t0):.0f}s)"
                )
                last_print = now

    elapsed = time.perf_counter() - t0
    print(
        f"[{name}] done in {elapsed:.0f}s — "
        f"copied={counts['copied']:,} skipped={counts['skipped']:,} "
        f"missing={counts['missing']:,} dryrun={counts['dryrun']:,}  "
        f"{bytes_done/1e9:.2f} GB"
    )

    # Don't hard-fail on missing files — continue to next cohort, record in
    # manifest, and let the caller decide. Per-cohort exit is signalled at
    # the end of main().
    if counts["missing"] > 0 and not dry_run:
        print(f"[{name}] WARNING: {counts['missing']} source files missing. "
              f"First 10:")
        for p in missing_paths[:10]:
            print(f"    {p}")
        if len(missing_paths) > 10:
            print(f"    ... and {len(missing_paths) - 10} more")

    # ── Optional: SHA-256 of N random sample stems for transport-verify ─────
    sample_hashes: dict[str, str] = {}
    if checksum_sample > 0 and not dry_run:
        all_stems = train_stems + val_stems
        rng = random.Random(0xBEEF)  # deterministic
        sample = rng.sample(all_stems, min(checksum_sample, len(all_stems)))
        for stem in sample:
            for split in ("train", "val"):
                cand = cohort_out / split / "images" / f"{stem}{IMG_EXT}"
                if cand.exists():
                    sample_hashes[stem] = sha256_file(cand)
                    break

    return {
        "source_root":    str(src_root),
        "destination":    str(cohort_out),
        "train_stems":    len(train_stems),
        "val_stems":      len(val_stems),
        "files_planned":  len(plan),
        "files_copied":   counts["copied"],
        "files_skipped":  counts["skipped"],
        "files_missing":  counts["missing"],
        "missing_paths":  missing_paths,
        "bytes_copied":   bytes_done,
        "elapsed_seconds": round(elapsed, 1),
        "sha256_sample":  sample_hashes,
    }


# ────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ────────────────────────────────────────────────────────────────────────────

def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gs40-root", required=True, type=Path)
    ap.add_argument("--gs55-root", required=True, type=Path)
    ap.add_argument("--gs33-root", required=True, type=Path)
    ap.add_argument("--out",       required=True, type=Path,
                    help="Output bundle root. Cohorts will be placed at <out>/{GS40,GS55,GS33}/.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true",
                    help="Walk the plan and count, but do not copy.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing destination files (default: skip if same size).")
    ap.add_argument("--checksum-sample", type=int, default=32,
                    help="Hash N random sample tiles per cohort for post-transport verify (0 disables).")
    args = ap.parse_args(argv)

    sources = {"GS40": args.gs40_root, "GS55": args.gs55_root, "GS33": args.gs33_root}
    for name, src in sources.items():
        if not src.is_dir():
            print(f"ERROR: {name} source not a directory: {src}", file=sys.stderr)
            return 2

    args.out.mkdir(parents=True, exist_ok=True)

    print("="*72)
    print(f"Building bundle at: {args.out}")
    print(f"Workers: {args.workers}   dry_run={args.dry_run}   force={args.force}")
    print("="*72)

    manifest = {
        "schema_version": 1,
        "created_at_unix": int(time.time()),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "cohorts": {},
    }

    for name, src in sources.items():
        manifest["cohorts"][name] = stage_cohort(
            name, src, args.out,
            workers=args.workers,
            dry_run=args.dry_run,
            force=args.force,
            checksum_sample=args.checksum_sample,
        )

    if not args.dry_run:
        (args.out / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(f"\nmanifest → {args.out / 'manifest.json'}")

    total_bytes   = sum(c["bytes_copied"]  for c in manifest["cohorts"].values())
    total_files   = sum(c["files_copied"] + c["files_skipped"] for c in manifest["cohorts"].values())
    total_missing = sum(c["files_missing"] for c in manifest["cohorts"].values())
    print(f"\nTOTAL: {total_files:,} files in scope, {total_bytes/1e9:.1f} GB copied, "
          f"{total_missing} missing source files")
    return 1 if total_missing > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
