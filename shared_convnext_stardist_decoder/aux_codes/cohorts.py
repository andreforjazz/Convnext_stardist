"""Resolve dataset paths from a single root, supporting laptop / SMB / HPC.

Used by the training notebook to keep the PARAMETERS cell tiny: instead of
hardcoding 7 subpaths × 3 cohorts, the notebook now does

    DATASETS_ROOT = resolve_datasets_root(...)
    COHORTS = {n: cohort_paths(n, DATASETS_ROOT) for n in ("GS40","GS55","GS33")}

The bundle on disk is expected to follow this contract:

    <DATASETS_ROOT>/<COHORT>/
        train/{images,labels}/
        val/{images,labels}/
        splits/fold_0/{train,val}.csv

(Produced by `make_training_dataset/make_bundle.py`.)
"""
from __future__ import annotations

import os
from pathlib import Path

_KEYS = (
    "train_images", "train_labels",
    "val_images",   "val_labels",
    "train_split",  "val_split",
)


def cohort_paths(name: str, root: Path | str) -> dict:
    """Return the 7 paths for one cohort under the unified bundle layout.

    Validates that all six required paths exist; raises FileNotFoundError
    with a precise list of what's missing.
    """
    base = Path(root) / name
    p: dict = {
        "root":         base,
        "train_images": base / "train"  / "images",
        "train_labels": base / "train"  / "labels",
        "val_images":   base / "val"    / "images",
        "val_labels":   base / "val"    / "labels",
        "train_split":  base / "splits" / "fold_0" / "train.csv",
        "val_split":    base / "splits" / "fold_0" / "val.csv",
    }
    missing = [k for k in _KEYS if not p[k].exists()]
    if missing:
        raise FileNotFoundError(
            f"Cohort {name!r} under {base} is missing: {missing}. "
            f"Expected unified bundle layout — produce it with "
            f"`python make_training_dataset/make_bundle.py`."
        )
    return p


def resolve_datasets_root(
    explicit: Path | str | None = None,
    candidates: tuple[Path | str | None, ...] = (),
    env_var: str = "STARDIST_DATASETS_ROOT",
) -> Path:
    """Pick the first existing root: explicit > $env_var > candidate list.

    Order:
      1. ``explicit`` if given (must exist; clear error if not).
      2. The ``env_var`` environment variable (e.g. set by a slurm wrapper).
      3. Each entry in ``candidates`` (None entries are skipped).

    Raises FileNotFoundError with a useful message if none resolve.
    """
    if explicit is not None:
        p = Path(explicit)
        if p.is_dir():
            return p
        raise FileNotFoundError(
            f"DATASETS_ROOT explicitly set to {p}, which does not exist."
        )

    env = os.environ.get(env_var)
    if env and Path(env).is_dir():
        return Path(env)

    for c in candidates:
        if c is None:
            continue
        p = Path(c)
        if p.is_dir():
            return p

    tried = [f"${env_var}={env or '(unset)'}"] + [str(c) for c in candidates if c is not None]
    raise FileNotFoundError(
        "No DATASETS_ROOT found. Tried (in order): " + ", ".join(tried) + ". "
        f"Set ${env_var} or pass `explicit=Path(...)`."
    )
