"""Verify a bundle produced by make_bundle.py.

Walks `<bundle>/<COHORT>/{train,val}/{images,labels}/` and checks:
  - file counts match `manifest.json`
  - every *.png in images/ has a matching *.png + *_inst2class.json in labels/
  - splits CSVs exist and contain no train/val stem overlap
  - (optional) SHA-256 of the manifest's sample stems still matches

Use after bundling AND after transport. Exit 0 on success, 1 on any drift.

Usage
-----
    python make_training_dataset/verify_bundle.py --bundle /path/to/bundle [--checksum]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


def _sha256(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _read_stems(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    if rows[0][0].strip().lower() in {"stem", "name", "id", "tile", "filename"}:
        rows = rows[1:]
    return [r[0].strip() for r in rows if r and r[0].strip()]


def verify_cohort(bundle: Path, name: str, manifest_entry: dict, do_checksum: bool) -> list[str]:
    """Return a list of failure messages (empty list = OK)."""
    fails: list[str] = []
    cohort = bundle / name
    if not cohort.is_dir():
        fails.append(f"{name}: missing directory {cohort}")
        return fails

    # ── splits + leak check ─────────────────────────────────────────────────
    train_csv = cohort / "splits" / "fold_0" / "train.csv"
    val_csv   = cohort / "splits" / "fold_0" / "val.csv"
    if not train_csv.is_file() or not val_csv.is_file():
        fails.append(f"{name}: missing splits CSVs ({train_csv}, {val_csv})")
        return fails
    # Dedupe within each CSV (matches the runtime dataset loader and the bundler).
    train_stems = list(dict.fromkeys(_read_stems(train_csv)))
    val_stems   = list(dict.fromkeys(_read_stems(val_csv)))
    overlap = set(train_stems) & set(val_stems)
    if overlap:
        fails.append(f"{name}: train/val stem overlap: {len(overlap)} stems "
                     f"(first 3: {sorted(overlap)[:3]})")
    expected_train = manifest_entry["train_stems"]
    expected_val   = manifest_entry["val_stems"]
    # Allow the bundle's stem count to be SMALLER than the manifest if the user
    # cleaned out orphan stems post-bundle (e.g. dropped rows missing inst2class JSONs).
    # A larger count would still be a real drift.
    if len(train_stems) > expected_train:
        fails.append(f"{name}: train.csv has {len(train_stems)} unique stems, manifest says {expected_train}")
    if len(val_stems) > expected_val:
        fails.append(f"{name}: val.csv has {len(val_stems)} unique stems, manifest says {expected_val}")

    # ── per-stem file presence ──────────────────────────────────────────────
    # Listing each directory once is ~1000x faster than per-file .exists() over SMB.
    for split, stems in (("train", train_stems), ("val", val_stems)):
        img_dir = cohort / split / "images"
        lbl_dir = cohort / split / "labels"
        try:
            img_set = {p.name for p in img_dir.iterdir()}
            lbl_set = {p.name for p in lbl_dir.iterdir()}
        except OSError as exc:
            fails.append(f"{name}/{split}: cannot list images/labels dir ({exc})")
            continue
        missing_img = missing_lbl = missing_json = 0
        for stem in stems:
            if f"{stem}.png" not in img_set:
                missing_img += 1
            if f"{stem}.png" not in lbl_set:
                missing_lbl += 1
            if f"{stem}_inst2class.json" not in lbl_set:
                missing_json += 1
        if missing_img or missing_lbl or missing_json:
            fails.append(
                f"{name}/{split}: missing files -- "
                f"images={missing_img}, label_pngs={missing_lbl}, label_jsons={missing_json}"
            )

    # ── checksum sample ─────────────────────────────────────────────────────
    if do_checksum:
        for stem, expected in manifest_entry.get("sha256_sample", {}).items():
            for split in ("train", "val"):
                cand = cohort / split / "images" / f"{stem}.png"
                if cand.exists():
                    got = _sha256(cand)
                    if got != expected:
                        fails.append(f"{name}: SHA-256 mismatch for {stem} (expected {expected[:8]}, got {got[:8]})")
                    break

    if not fails:
        print(f"  {name}: OK  ({len(train_stems):,} train / {len(val_stems):,} val)")
    else:
        for f in fails:
            print(f"  {name}: FAIL — {f}")
    return fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--checksum", action="store_true",
                    help="Re-compute SHA-256 for the manifest's sample stems.")
    args = ap.parse_args(argv)

    manifest_path = args.bundle / "manifest.json"
    if not manifest_path.is_file():
        print(f"ERROR: no manifest at {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    print(f"Verifying bundle: {args.bundle}")
    print(f"  schema_version: {manifest.get('schema_version')}")
    print(f"  cohorts: {list(manifest['cohorts'])}")

    all_fails: list[str] = []
    for name, entry in manifest["cohorts"].items():
        all_fails.extend(verify_cohort(args.bundle, name, entry, args.checksum))

    if all_fails:
        print(f"\n{len(all_fails)} failures total — bundle is NOT clean.")
        return 1
    print("\nAll cohorts verified OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
