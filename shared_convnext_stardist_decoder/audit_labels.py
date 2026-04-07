"""
Audit label directories for corrupt TIFFs, empty / invalid inst2class JSON files,
and mismatched class names.

Usage (from repo root):
    python -m shared_convnext_stardist_decoder.audit_labels \
        --labels_dir  "\\\\server\\...\\train_instance_labels" \
        --class_names bladder bone brain collagen ear eye gi heart kidney liver \
                      lungs mesokidney nontissue pancreas skull spleen spleen2 thymus thyroid
        [--delete_bad]   # actually remove the bad files (default: dry-run only)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _tiff_page_count(path: Path) -> int:
    try:
        import tifffile
        with tifffile.TiffFile(path) as tf:
            return len(tf.pages)
    except Exception:
        return -1


def audit(labels_dir: Path, class_names: list[str], delete_bad: bool) -> None:
    labels_dir = Path(labels_dir)
    if not labels_dir.is_dir():
        print(f"ERROR: {labels_dir} does not exist or is not a directory.")
        return

    known_names = {n.strip().lower() for n in class_names}

    bad_tiffs: list[Path] = []
    bad_jsons: list[Path] = []
    unknown_class_jsons: list[tuple[Path, set[str]]] = []
    empty_jsons: list[Path] = []
    ok_tiffs = 0
    ok_jsons = 0

    tiff_files  = list(labels_dir.glob("*.tif")) + list(labels_dir.glob("*.tiff"))
    json_files  = list(labels_dir.glob("*_inst2class.json"))

    print(f"\nScanning {labels_dir}")
    print(f"  Found {len(tiff_files)} TIFF masks, {len(json_files)} inst2class JSONs")

    for p in tiff_files:
        n = _tiff_page_count(p)
        if n == 0:
            bad_tiffs.append(p)
        elif n < 0:
            bad_tiffs.append(p)
        else:
            ok_tiffs += 1

    for p in json_files:
        try:
            raw_bytes = p.read_bytes()
        except OSError:
            bad_jsons.append(p)
            continue
        txt = raw_bytes.decode("utf-8-sig").strip()
        if not txt:
            empty_jsons.append(p)
            continue
        try:
            data = json.loads(txt)
        except json.JSONDecodeError:
            bad_jsons.append(p)
            continue
        if not isinstance(data, dict):
            bad_jsons.append(p)
            continue

        if known_names:
            foreign: set[str] = set()
            for v in data.values():
                name = str(v).strip().lower()
                if name not in known_names:
                    foreign.add(name)
            if foreign:
                unknown_class_jsons.append((p, foreign))
        ok_jsons += 1

    print(f"\n--- Results ---")
    print(f"  OK TIFFs:  {ok_tiffs}  |  Bad/empty TIFFs:  {len(bad_tiffs)}")
    print(f"  OK JSONs:  {ok_jsons}  |  Bad JSONs: {len(bad_jsons)}  |  Empty JSONs: {len(empty_jsons)}")
    print(f"  JSONs with unknown class names: {len(unknown_class_jsons)}")

    if bad_tiffs:
        print(f"\nBad TIFFs ({len(bad_tiffs)}):")
        for p in bad_tiffs:
            print(f"  {p.name}")
        if delete_bad:
            for p in bad_tiffs:
                p.unlink()
            print(f"  Deleted {len(bad_tiffs)} bad TIFFs.")

    if bad_jsons:
        print(f"\nCorrupt JSONs ({len(bad_jsons)}):")
        for p in bad_jsons:
            print(f"  {p.name}")
        if delete_bad:
            for p in bad_jsons:
                p.unlink()
            print(f"  Deleted {len(bad_jsons)} corrupt JSONs.")

    if empty_jsons:
        print(f"\nEmpty JSONs ({len(empty_jsons)}):")
        for p in empty_jsons:
            print(f"  {p.name}")
        if delete_bad:
            for p in empty_jsons:
                p.unlink()
            print(f"  Deleted {len(empty_jsons)} empty JSONs.")

    if unknown_class_jsons:
        all_foreign: set[str] = set()
        print(f"\nJSONs with class names NOT in model.class_names ({len(unknown_class_jsons)}) — first 20:")
        for p, foreign in unknown_class_jsons[:20]:
            print(f"  {p.name}  ->  {sorted(foreign)}")
            all_foreign |= foreign
        print(f"\n  All foreign names found: {sorted(all_foreign)}")
        print(
            "  ACTION: Either add these names to model.class_names in your YAML, "
            "or fix the JSON values to match existing class names."
        )

    if not any([bad_tiffs, bad_jsons, empty_jsons, unknown_class_jsons]):
        print("\n  All label files look clean.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit multitask label directories.")
    ap.add_argument("--labels_dir", type=Path, required=True,
                    help="Path to instance labels folder (contains *.tif / *.png + *_inst2class.json).")
    ap.add_argument("--class_names", nargs="*", default=[],
                    help="Expected class names (from model.class_names in YAML). "
                         "If provided, JSONs with unknown names are flagged.")
    ap.add_argument("--delete_bad", action="store_true",
                    help="Delete bad/empty files (default: dry-run, only report).")
    args = ap.parse_args()

    audit(args.labels_dir, args.class_names, args.delete_bad)


if __name__ == "__main__":
    main()
