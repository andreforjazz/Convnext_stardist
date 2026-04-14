"""
Build the three GS55 training-dataset notebooks programmatically.
Run once:  py -3 _build_notebooks.py
"""
import json
from pathlib import Path

BASE = Path(__file__).parent


def nb(cells):
    """Minimal notebook skeleton."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip()}


def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.strip(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Notebook 0 — Build CODA GeoJSONs  (no bounding box)
# ═══════════════════════════════════════════════════════════════════════════════

NB0_IMPORTS = """\
import sys
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

# dataset_utils is in the same folder as this notebook
sys.path.insert(0, str(Path.cwd()))
from dataset_utils import assign_coda_to_geojson, polygon_centroid

print("Imports OK")
"""

NB0_PARAMS = """\
# ── Paths ──────────────────────────────────────────────────────────────────────
# StarDist nuclear polygons at 20x (input)
GEOJSON_DIR = Path(r"\\\\kittyserverdw\\Andre_kit\\data\\monkey_fetus\\bissected_monkey_GS55\\StarDist_10_10_2025_cross_fetal_species\\json\\geojsons\\32_polys_20x")

# CODA whole-slide classification masks at 5x (input)
MASK_DIR = Path(r"\\\\kittyserverdw\\Andre_kit\\data\\monkey_fetus\\bissected_monkey_GS55\\5x\\classification_MODEL1_5x_GS40_GS55_06_10_2025_45_big_tiles_inceptionresnetv2")

# Output: CODA-labelled GeoJSONs
OUT_DIR = Path(r"\\\\kittyserverdw\\Andre_kit\\data\\students\\Diogo\\data\\fetal\\GS55\\geojson_CODAclass")

# ── Resolution ─────────────────────────────────────────────────────────────────
# GS55 scanned at 20x; CODA masks are at 5x (MPP_MASK = 2 µm/px)
MPP_20X  = 0.4416   # µm/px at 20x (GeoJSON coordinate space)
MPP_MASK = 2.0      # µm/px at 5x  (CODA mask pixel space)
# No bounding-box files needed — GS55 CODA masks cover the full slide.

# ── Class palette (must match training) ────────────────────────────────────────
LABELS = [
    "bone",   "brain",  "eye",       "heart",     "lungs",
    "GI",     "liver",  "spleen",    "pancreas",  "kidney",
    "mesokidney", "collagen", "ear", "nontissue", "thymus",
    "thyroid", "bladder", "skull",   "spleen2",
]
COLORS = [
    [214,212,161], [247,184, 67], [136,232, 95], [140, 13, 13], [ 38, 27,166],
    [ 13,125, 11], [179, 50,108], [228,235,131], [156, 96,235], [ 46,190,230],
    [150,255,245], [254,222,255], [235,154,108], [255,255,255], [  9, 64,116],
    [255,255, 74], [178,178,  0], [214,212,161], [ 54, 83, 89],
]

print(f"GeoJSON dir  : {GEOJSON_DIR}")
print(f"  exists     : {GEOJSON_DIR.exists()}")
print(f"Mask dir     : {MASK_DIR}")
print(f"  exists     : {MASK_DIR.exists()}")
print(f"Output dir   : {OUT_DIR}")
"""

NB0_RUN = """\
# ── Process all slides ─────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)

gj_files = sorted(GEOJSON_DIR.glob("*.geojson"))
print(f"Found {len(gj_files)} GeoJSON files")

skipped = []
for gj_path in tqdm(gj_files, desc="Assigning CODA labels"):
    out_path = OUT_DIR / gj_path.name
    if out_path.exists():
        continue  # already processed — delete file to redo

    # Try .tif first, fall back to .png
    mask_path = MASK_DIR / f"{gj_path.stem}.tif"
    if not mask_path.exists():
        mask_path = MASK_DIR / f"{gj_path.stem}.png"

    if not mask_path.exists():
        skipped.append(gj_path.stem)
        continue

    n = assign_coda_to_geojson(
        geojson_path=gj_path,
        mask_path=mask_path,
        out_path=out_path,
        labels=LABELS,
        colors=COLORS,
        mpp_20x=MPP_20X,
        mpp_mask=MPP_MASK,
    )

print(f"\\nDone. Skipped {len(skipped)} slides (no matching mask):")
for s in skipped[:10]:
    print(f"  {s}")
"""

NB0_SANITY_MD = """\
---
## Sanity check — single slide

Run the cell below to confirm centroids land correctly inside the CODA mask.
Pick a different `SAMPLE_IDX` if the default slide is not representative.
"""

NB0_SANITY = """\
SAMPLE_IDX = 0   # ← change to check a different slide

gj_files = sorted(GEOJSON_DIR.glob("*.geojson"))
gj_path  = gj_files[SAMPLE_IDX]

mask_path = MASK_DIR / f"{gj_path.stem}.tif"
if not mask_path.exists():
    mask_path = MASK_DIR / f"{gj_path.stem}.png"

scale = MPP_20X / MPP_MASK   # 20x → 5x coordinate scale
mask  = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
data  = json.loads(gj_path.read_text(encoding="utf-8"))
feats = data if isinstance(data, list) else data.get("features", [])

cx_list, cy_list = [], []
for feat in feats:
    ring = feat.get("geometry", {}).get("coordinates", [[]])[0]
    cx, cy = polygon_centroid(ring)
    if cx is not None:
        cx_list.append(cx)
        cy_list.append(cy)

cx_20x, cy_20x = np.array(cx_list), np.array(cy_list)
cx_5x  = cx_20x * scale
cy_5x  = cy_20x * scale

in_mask = (
    (cx_5x >= 0) & (cx_5x < mask.shape[1]) &
    (cy_5x >= 0) & (cy_5x < mask.shape[0])
)
print(f"Slide      : {gj_path.stem}")
print(f"Mask shape : {mask.shape[1]} x {mask.shape[0]} px")
print(f"Centroids  : {len(cx_20x):,} nuclei — {in_mask.mean()*100:.1f}% inside mask bounds")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].imshow(mask, cmap="tab20b")
axes[0].scatter(cx_5x[in_mask],  cy_5x[in_mask],  s=0.3, color="lime",  label="in mask")
axes[0].scatter(cx_5x[~in_mask], cy_5x[~in_mask], s=0.3, color="red",   label="outside", alpha=0.4)
axes[0].set_title(f"CODA mask + centroids ({in_mask.mean()*100:.1f}% on-target)")
axes[0].legend(markerscale=10)

axes[1].scatter(cx_5x, cy_5x, s=0.1, alpha=0.3, color="steelblue")
axes[1].set_xlim(0, mask.shape[1])
axes[1].set_ylim(mask.shape[0], 0)
axes[1].set_title(f"Centroids in 5x space  ({len(cx_5x):,} total)")
axes[1].set_aspect("equal")

plt.tight_layout()
plt.show()
"""

nb0 = nb([
    md("# GS55 — Step 0: Assign CODA organ labels to StarDist GeoJSONs\n\n"
       "Reads each StarDist polygon GeoJSON (20x coordinates) and the matching\n"
       "whole-slide CODA classification mask (5x), then writes a new GeoJSON where\n"
       "every nucleus has a `classification.name` property.\n\n"
       "**No bounding-box files are needed** — the GS55 CODA masks cover the full\n"
       "slide, so a single scale factor maps 20x → 5x coordinates.\n\n"
       "| Cell | Purpose |\n"
       "|------|---------|\n"
       "| 1 | Imports |\n"
       "| **2** | **Parameters ← edit here** |\n"
       "| 3 | Process all slides |\n"
       "| 4 | Sanity-check overlay plot |"),
    md("---\n## 1 · Imports"),
    code(NB0_IMPORTS),
    md("---\n## 2 · Parameters  ← edit here"),
    code(NB0_PARAMS),
    md("---\n## 3 · Run: assign CODA labels to all slides"),
    code(NB0_RUN),
    md(NB0_SANITY_MD),
    code(NB0_SANITY),
])

(BASE / "0_build_CODA_geojsons.ipynb").write_text(
    json.dumps(nb0, indent=1, ensure_ascii=False), encoding="utf-8"
)
print("Written: 0_build_CODA_geojsons.ipynb")


# ═══════════════════════════════════════════════════════════════════════════════
# Notebook 1 — Cell-type analysis
# ═══════════════════════════════════════════════════════════════════════════════

NB1_IMPORTS = """\
import sys
from pathlib import Path
from collections import Counter
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from tqdm.auto import tqdm

sys.path.insert(0, str(Path.cwd()))
from dataset_utils import extract_cell_types_from_geojson, normalize_slide_stem

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (16, 8)
print("Imports OK")
"""

NB1_PARAMS = """\
# ── Paths ──────────────────────────────────────────────────────────────────────
GEOJSON_DIR = Path(r"\\\\kittyserverdw\\Andre_kit\\data\\students\\Diogo\\data\\fetal\\GS55\\geojson_CODAclass")
OUT_DIR     = Path(r"\\\\kittyserverdw\\Andre_kit\\data\\students\\Diogo\\data\\fetal\\GS55\\cellvit_training\\cell_type_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Optional filters ───────────────────────────────────────────────────────────
# Leave empty to process every slide; fill with slide IDs to restrict.
FILTER_SLIDE_IDS  = []
EXCLUDE_CELL_TYPES = ["Unassigned", "OutsideMask"]

print(f"GeoJSON dir : {GEOJSON_DIR}  (exists: {GEOJSON_DIR.exists()})")
print(f"Output dir  : {OUT_DIR}")
"""

NB1_LOAD = """\
# ── Load all GeoJSON files ─────────────────────────────────────────────────────
gj_files = sorted(GEOJSON_DIR.glob("*.geojson"))
if not gj_files:
    raise FileNotFoundError(f"No GeoJSON files in {GEOJSON_DIR}")

if FILTER_SLIDE_IDS:
    gj_files = [f for f in gj_files if normalize_slide_stem(f) in FILTER_SLIDE_IDS]

print(f"Processing {len(gj_files)} slides")

rows = []
errors = []
for gj_path in tqdm(gj_files, desc="Extracting cell types"):
    slide_id = normalize_slide_stem(gj_path)
    counts = extract_cell_types_from_geojson(gj_path)
    if not counts:
        errors.append(gj_path.name)
        continue
    for cell_type, count in counts.items():
        if cell_type not in EXCLUDE_CELL_TYPES:
            rows.append({"slide_id": slide_id, "cell_type": cell_type, "count": count})

df = pd.DataFrame(rows)
if df.empty:
    raise ValueError("No cell data extracted — check GEOJSON_DIR and EXCLUDE_CELL_TYPES.")

total_cells = df["count"].sum()
n_slides    = df["slide_id"].nunique()
print(f"Slides: {n_slides}  |  Cell types: {df['cell_type'].nunique()}  |  Total cells: {total_cells:,}")
if errors:
    print(f"Skipped {len(errors)} files with no cells: {errors[:5]}")
display(df.head())
"""

NB1_STATS = """\
# ── Overall cell-type distribution ────────────────────────────────────────────
cell_type_totals = df.groupby("cell_type")["count"].sum().sort_values(ascending=False)

print(f"{'Cell Type':<25} {'Count':>12} {'Percentage':>12}")
print("-" * 55)
for ct, cnt in cell_type_totals.items():
    print(f"{ct:<25} {cnt:>12,} {cnt/total_cells*100:>11.2f}%")

# ── Per-slide pivot table ──────────────────────────────────────────────────────
pivot_df = (
    df.pivot_table(index="slide_id", columns="cell_type", values="count",
                   fill_value=0, aggfunc="sum")
)
pivot_df["TOTAL"] = pivot_df.sum(axis=1)
pivot_df = pivot_df.sort_values("TOTAL", ascending=False)
print(f"\\nPivot table: {pivot_df.shape[0]} slides × {pivot_df.shape[1]-1} cell types")
display(pivot_df.head(10))
"""

NB1_PLOTS = """\
# ── Overall bar chart ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
ct_sorted = cell_type_totals.sort_values(ascending=True)
colors    = plt.cm.tab20(np.linspace(0, 1, len(ct_sorted)))
ax.barh(ct_sorted.index, ct_sorted.values, color=colors)
for i, (ct, cnt) in enumerate(ct_sorted.items()):
    ax.text(cnt + total_cells * 0.005, i, f"{cnt:,} ({cnt/total_cells*100:.1f}%)", va="center", fontsize=8)
ax.set_xlabel("Number of Cells"); ax.set_title(f"GS55 cell-type distribution — {n_slides} slides, {total_cells:,} cells")
ax.grid(axis="x", alpha=0.3); plt.tight_layout()
plt.savefig(OUT_DIR / "01_cell_type_distribution_overall.png", dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved: {OUT_DIR / '01_cell_type_distribution_overall.png'}")

# ── Per-slide stacked bar (top 40 slides) ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 7))
top_n    = min(40, len(pivot_df))
plot_df  = pivot_df.drop(columns=["TOTAL"]).head(top_n)
plot_df.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", width=0.8,
             edgecolor="white", linewidth=0.4)
ax.set_title(f"Cell-type distribution by slide (top {top_n})"); ax.set_xlabel("Slide")
ax.legend(title="Cell type", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
plt.xticks(rotation=90); plt.tight_layout()
plt.savefig(OUT_DIR / "02_cell_type_distribution_per_slide.png", dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved: {OUT_DIR / '02_cell_type_distribution_per_slide.png'}")
"""

NB1_EXPORT = """\
# ── Export CSVs ────────────────────────────────────────────────────────────────
pivot_df.to_csv(OUT_DIR / "cell_counts_per_slide.csv")
df.to_csv(OUT_DIR / "cell_counts_detailed.csv", index=False)

summary = pd.DataFrame({
    "cell_type":       cell_type_totals.index,
    "total_count":     cell_type_totals.values,
    "percentage":      (cell_type_totals.values / total_cells * 100).round(2),
}).reset_index(drop=True)
summary.to_csv(OUT_DIR / "cell_type_summary.csv", index=False)

slide_totals = df.groupby("slide_id")["count"].sum().sort_values(ascending=False)
slide_summary = pd.DataFrame({"slide_id": slide_totals.index, "total_cells": slide_totals.values})
slide_summary.to_csv(OUT_DIR / "slide_summary.csv", index=False)

print("Saved:")
for f in ["cell_counts_per_slide.csv", "cell_counts_detailed.csv",
          "cell_type_summary.csv", "slide_summary.csv"]:
    print(f"  {OUT_DIR / f}")

print(f"\\nTop 5 cell types:")
for i, (ct, cnt) in enumerate(cell_type_totals.head(5).items(), 1):
    print(f"  {i}. {ct}: {cnt:,} ({cnt/total_cells*100:.2f}%)")
"""

nb1 = nb([
    md("# GS55 — Step 1: Cell-type distribution analysis\n\n"
       "Reads the CODA-labelled GeoJSONs produced by Step 0, counts nuclei per\n"
       "cell type per slide, and exports summary CSVs used by Step 2.\n\n"
       "| Cell | Purpose |\n"
       "|------|---------|\n"
       "| 1 | Imports |\n"
       "| **2** | **Parameters ← edit here** |\n"
       "| 3 | Load & count |\n"
       "| 4 | Statistics table |\n"
       "| 5 | Plots |\n"
       "| 6 | Export CSVs |"),
    md("---\n## 1 · Imports"),
    code(NB1_IMPORTS),
    md("---\n## 2 · Parameters  ← edit here"),
    code(NB1_PARAMS),
    md("---\n## 3 · Load data"),
    code(NB1_LOAD),
    md("---\n## 4 · Statistics"),
    code(NB1_STATS),
    md("---\n## 5 · Plots"),
    code(NB1_PLOTS),
    md("---\n## 6 · Export CSVs"),
    code(NB1_EXPORT),
])

(BASE / "1_get_celltypes_from_slides.ipynb").write_text(
    json.dumps(nb1, indent=1, ensure_ascii=False), encoding="utf-8"
)
print("Written: 1_get_celltypes_from_slides.ipynb")


# ═══════════════════════════════════════════════════════════════════════════════
# Notebook 2 — Build tile dataset
# ═══════════════════════════════════════════════════════════════════════════════

NB2_IMPORTS = """\
import csv
import json
import re
import sys
import yaml
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openslide
from tqdm.auto import tqdm

sys.path.insert(0, str(Path.cwd()))
from dataset_utils import (
    calculate_hybrid_weights,
    get_slide_mpp,
    assign_cells_to_tiles,
    augment_image,
    augment_coords,
    polygon_centroid,
)

print("Imports OK")
"""

NB2_PARAMS = """\
# ── Paths ──────────────────────────────────────────────────────────────────────
CELL_COUNTS_CSV = Path(r"\\\\kittyserverdw\\Andre_kit\\data\\students\\Diogo\\data\\fetal\\GS55\\cellvit_training\\cell_type_analysis\\cell_counts_per_slide.csv")
GEOJSON_DIR     = Path(r"\\\\kittyserverdw\\Andre_kit\\data\\students\\Diogo\\data\\fetal\\GS55\\geojson_CODAclass")
# TODO: update NDPI_BASE_DIR to the folder containing GS55 whole-slide .ndpi files
NDPI_BASE_DIR   = Path(r"\\\\kittyserverdw\\Andre_kit\\data\\monkey_fetus\\bissected_monkey_GS55")
OUT_DIR         = Path(r"\\\\kittyserverdw\\Andre_kit\\data\\students\\Diogo\\data\\fetal\\GS55\\cellvit_training")
OUT_BASE        = OUT_DIR / "dataset_256_40k_GS55"

# ── Class palette ──────────────────────────────────────────────────────────────
LABELS = [
    "bone",   "brain",  "eye",       "heart",     "lungs",
    "GI",     "liver",  "spleen",    "pancreas",  "kidney",
    "mesokidney", "collagen", "ear", "nontissue", "thymus",
    "thyroid", "bladder", "skull",   "spleen2",
]
COLORS = [
    [214,212,161], [247,184, 67], [136,232, 95], [140, 13, 13], [ 38, 27,166],
    [ 13,125, 11], [179, 50,108], [228,235,131], [156, 96,235], [ 46,190,230],
    [150,255,245], [254,222,255], [235,154,108], [255,255,255], [  9, 64,116],
    [255,255, 74], [178,178,  0], [214,212,161], [ 54, 83, 89],
]

# ── Tile extraction ────────────────────────────────────────────────────────────
TILE_SIZE       = 256   # px at 20x
STRIDE          = 256   # non-overlapping grid
MIN_CELLS_TILE  = 5     # discard tiles with fewer cells than this
NONTISSUE_FRAC  = 0.75  # discard tiles where >75% of cells are "nontissue"
TARGET_TILES    = 40000 # unique tiles to sample before oversampling
SEED            = 1337

# ── Train / val / test split ───────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20
# TEST_RATIO  = 1 - TRAIN_RATIO - VAL_RATIO (computed automatically)

name_to_id = {name: i for i, name in enumerate(LABELS)}
id_to_name = {i: name for i, name in enumerate(LABELS)}

print(f"Cell counts CSV : {CELL_COUNTS_CSV.exists()}")
print(f"GeoJSON dir     : {GEOJSON_DIR.exists()}")
print(f"NDPI base dir   : {NDPI_BASE_DIR.exists()}")
print(f"Output base     : {OUT_BASE}")
"""

NB2_WEIGHTS = """\
# ── Load class distribution and compute sampling weights ───────────────────────
df_counts   = pd.read_csv(CELL_COUNTS_CSV, index_col=0)
if "TOTAL" in df_counts.columns:
    df_counts = df_counts.drop(columns=["TOTAL"])

class_totals   = df_counts.sum(axis=0).sort_values(ascending=False)
total_cells    = int(class_totals.sum())
hybrid_weights = calculate_hybrid_weights(class_totals)

print(f"Total cells: {total_cells:,}  |  Classes: {len(class_totals)}")
print(f"\\n{'Class':<20} {'Count':>12} {'%':>7} {'Weight':>9}")
print("-" * 55)
for cls, cnt in class_totals.items():
    print(f"{cls:<20} {cnt:>12,} {cnt/total_cells*100:>6.1f}% {hybrid_weights[cls]:>9.3f}")
"""

NB2_MANIFEST = """\
# ── Build slide manifest (GeoJSON + NDPI path pairs) ──────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)

gj_files   = sorted(GEOJSON_DIR.glob("*.geojson"))
manifest   = []
no_ndpi    = []

for gj_path in gj_files:
    ndpi_path = NDPI_BASE_DIR / f"{gj_path.stem}.ndpi"
    if ndpi_path.exists():
        manifest.append({
            "slide_id":     gj_path.stem,
            "image_path":   str(ndpi_path),
            "geojson_path": str(gj_path),
        })
    else:
        no_ndpi.append(gj_path.stem)

manifest_path = OUT_DIR / "GS55_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print(f"Matched   : {len(manifest)} / {len(gj_files)} slides")
print(f"No NDPI   : {len(no_ndpi)}{' — check NDPI_BASE_DIR' if no_ndpi else ''}")
print(f"Manifest  → {manifest_path}")
if no_ndpi[:5]:
    print("  Missing:", no_ndpi[:5])
"""

NB2_PARSE = """\
# ── Parse GeoJSON annotations → per-slide cell arrays ─────────────────────────
slide_data = {}

for entry in tqdm(manifest, desc="Parsing GeoJSONs"):
    gj_path  = Path(entry["geojson_path"])
    raw      = json.loads(gj_path.read_text(encoding="utf-8"))
    features = raw if isinstance(raw, list) else raw.get("features", [])

    centroids, labels = [], []
    for feat in features:
        class_name = feat.get("properties", {}).get("classification", {}).get("name")
        if not class_name or class_name not in name_to_id:
            continue
        geom = feat.get("geometry", {})
        if geom.get("type") != "Polygon":
            continue
        ring = geom.get("coordinates", [[]])[0]
        cx, cy = polygon_centroid(ring)
        if cx is None:
            continue
        centroids.append([cx, cy])
        labels.append(name_to_id[class_name])

    if centroids:
        slide_data[entry["slide_id"]] = {
            "image_path":   entry["image_path"],
            "cells_xy":     np.array(centroids, dtype=np.float32),
            "cells_labels": np.array(labels,    dtype=np.int32),
        }

total_parsed = sum(len(v["cells_xy"]) for v in slide_data.values())
print(f"Parsed {len(slide_data)} slides, {total_parsed:,} cells")
"""

NB2_TILES = """\
# ── Generate tile candidates ───────────────────────────────────────────────────
NONTISSUE_ID = name_to_id.get("nontissue", -1)

all_tiles = []
for slide_id, data in tqdm(slide_data.items(), desc="Building tile grid"):
    cells_xy     = data["cells_xy"]
    cells_labels = data["cells_labels"]
    tile_groups  = assign_cells_to_tiles(cells_xy, TILE_SIZE, STRIDE)

    for (tx, ty), cell_idxs in tile_groups.items():
        if len(cell_idxs) < MIN_CELLS_TILE:
            continue
        nt_frac = sum(1 for ci in cell_idxs if cells_labels[ci] == NONTISSUE_ID) / len(cell_idxs)
        if nt_frac > NONTISSUE_FRAC:
            continue
        class_counts = Counter(cells_labels[cell_idxs].tolist())
        all_tiles.append({
            "slide_id":    slide_id,
            "tile_x":      tx,
            "tile_y":      ty,
            "cell_indices": cell_idxs,
            "num_cells":   len(cell_idxs),
            "class_counts": class_counts,
        })

print(f"Candidate tiles: {len(all_tiles):,}")
"""

NB2_SELECT = """\
# ── Weighted unique tile selection ────────────────────────────────────────────
BIG5_NAMES = ["collagen", "bone", "brain", "liver", "nontissue"]
BIG5_IDS   = {name_to_id[n] for n in BIG5_NAMES if n in name_to_id}
RARE_IDS   = {
    name_to_id[n] for n in LABELS
    if n in class_totals.index and class_totals[n] / total_cells < 0.01
}

weights = []
valid   = []
for idx, tile in enumerate(all_tiles):
    tc = tile["num_cells"]
    if tc == 0:
        continue
    value = 0.0
    for cls_id, cnt in tile["class_counts"].items():
        cname = id_to_name.get(cls_id, "")
        if cls_id in BIG5_IDS:
            value += cnt * (0.01 if cname == "collagen" else 0.02 if cname == "nontissue" else 0.03)
        elif cls_id in RARE_IDS:
            value += cnt * 10.0
        else:
            value += cnt * 1.0
    value /= tc
    b5 = sum(tile["class_counts"].get(b, 0) for b in BIG5_IDS)
    if b5 / tc > 0.6:   value *= 0.01
    elif b5 / tc > 0.4: value *= 0.05
    valid.append(idx)
    weights.append(value)

weights = np.array(weights, dtype=float)
probs   = np.maximum(weights, 1e-10)
probs  /= probs.sum()

np.random.seed(SEED)
n_sample       = min(TARGET_TILES, len(valid))
sel_indices    = np.random.choice(valid, size=n_sample, replace=False, p=probs)
selected_tiles = [all_tiles[i] for i in sel_indices]

print(f"Selected {len(selected_tiles):,} unique tiles (target: {TARGET_TILES:,})")

# ── Class distribution after selection ────────────────────────────────────────
sampled_cc = Counter()
for tile in selected_tiles:
    sampled_cc.update(tile["class_counts"])

print(f"\\n{'Class':<18} {'Original':>10} {'Sampled':>10} {'Ratio':>8}")
print("-" * 50)
for cname in LABELS:
    cid  = name_to_id[cname]
    orig = class_totals.get(cname, 0)
    samp = sampled_cc.get(cid, 0)
    print(f"{cname:<18} {orig:>10,} {samp:>10,} {samp/orig if orig else 0:>7.2f}x")
"""

NB2_SPLIT = """\
# ── Slide-level train / val / test split ──────────────────────────────────────
tiles_by_slide = defaultdict(list)
for tile in selected_tiles:
    tiles_by_slide[tile["slide_id"]].append(tile)

slide_ids   = list(tiles_by_slide.keys())
total_count = len(selected_tiles)
target_train = int(total_count * TRAIN_RATIO)
target_val   = int(total_count * VAL_RATIO)

# Phase 1: ensure rare-class slides appear in all splits
class_to_slides = defaultdict(list)
for sid, tiles in tiles_by_slide.items():
    for tile in tiles:
        for cls_id in tile["class_counts"]:
            class_to_slides[cls_id].append(sid)
class_to_slides = {k: list(set(v)) for k, v in class_to_slides.items()}

np.random.seed(SEED)
train_slides, val_slides, test_slides = set(), set(), set()

for cls_id in sorted(RARE_IDS):
    slides = [s for s in class_to_slides.get(cls_id, []) if s not in train_slides | val_slides | test_slides]
    if len(slides) >= 3:
        slides.sort(key=lambda s: len(tiles_by_slide[s]))
        train_slides.add(slides[0])
        val_slides.add(slides[len(slides) // 2])
        test_slides.add(slides[-1])
    elif len(slides) == 2:
        train_slides.add(slides[0]); val_slides.add(slides[1])
    elif len(slides) == 1:
        train_slides.add(slides[0])

# Phase 2: greedily assign remaining slides to balance tile counts
remaining = sorted(
    [s for s in slide_ids if s not in train_slides | val_slides | test_slides],
    key=lambda s: len(tiles_by_slide[s]), reverse=True
)
for sid in remaining:
    tr = sum(len(tiles_by_slide[s]) for s in train_slides)
    va = sum(len(tiles_by_slide[s]) for s in val_slides)
    te = sum(len(tiles_by_slide[s]) for s in test_slides)
    if   target_train - tr >= max(target_val - va, total_count - target_train - target_val - te):
        train_slides.add(sid)
    elif target_val - va >= total_count - target_train - target_val - te:
        val_slides.add(sid)
    else:
        test_slides.add(sid)

train_tiles_unique = [t for t in selected_tiles if t["slide_id"] in train_slides]
val_tiles          = [t for t in selected_tiles if t["slide_id"] in val_slides]
test_tiles         = [t for t in selected_tiles if t["slide_id"] in test_slides]

print(f"TRAIN : {len(train_slides):2d} slides → {len(train_tiles_unique):,} tiles ({len(train_tiles_unique)/total_count*100:.1f}%)")
print(f"VAL   : {len(val_slides):2d} slides → {len(val_tiles):,} tiles ({len(val_tiles)/total_count*100:.1f}%)")
print(f"TEST  : {len(test_slides):2d} slides → {len(test_tiles):,} tiles ({len(test_tiles)/total_count*100:.1f}%)")
"""

NB2_OVERSAMPLE = """\
# ── Oversample rare classes in training split (augmented duplicates) ───────────
# Each rare class is oversampled until it has at least OVERSAMPLE_TARGET tiles.
# Copies get a _dup{k} suffix and a deterministic geometric augmentation.

OVERSAMPLE_TARGETS = {
    "kidney":     2500, "pancreas":   2000, "bladder":    2500,
    "spleen":     1500, "spleen2":    1200, "thyroid":    2000,
    "ear":        1500, "eye":        1200, "mesokidney": 1500, "thymus": 1500,
}

train_by_cls = defaultdict(list)
for tile in train_tiles_unique:
    for cls_id in tile["class_counts"]:
        train_by_cls[cls_id].append(tile)

oversampled = []
np.random.seed(SEED + 1)

print(f"{'Class':<13} {'Have':>8} {'Target':>8} {'Added':>8}")
print("-" * 45)
for cls_name, tgt in OVERSAMPLE_TARGETS.items():
    cls_id    = name_to_id[cls_name]
    available = train_by_cls.get(cls_id, [])
    n_have    = len(available)
    if n_have == 0 or n_have >= tgt:
        print(f"  {cls_name:<11} {n_have:>8,} {tgt:>8,} {'—':>8}")
        continue
    tgt     = min(tgt, n_have * 8)  # cap at 8 augmentation variants
    n_add   = tgt - n_have
    chosen  = np.random.choice(n_have, size=n_add, replace=True)
    dup_ctr = {}
    for orig_idx in chosen:
        k = dup_ctr.get(orig_idx, 0) + 1
        dup_ctr[orig_idx] = k
        dup = dict(available[orig_idx])
        dup["dup_suffix"] = f"_dup{k}"
        oversampled.append(dup)
    print(f"  {cls_name:<11} {n_have:>8,} {tgt:>8,} {n_add:>8,}")

train_tiles = train_tiles_unique + oversampled

for t in train_tiles: t["split"] = "train"
for t in val_tiles:   t["split"] = "val"
for t in test_tiles:  t["split"] = "test"

print(f"\\nFinal: TRAIN {len(train_tiles):,}  VAL {len(val_tiles):,}  TEST {len(test_tiles):,}")
"""

NB2_EXTRACT = """\
# ── Create output directory structure ─────────────────────────────────────────
images_dir = OUT_BASE / "train" / "images"
labels_dir = OUT_BASE / "train" / "labels"
splits_dir = OUT_BASE / "splits"
for d in (images_dir, labels_dir, splits_dir):
    d.mkdir(parents=True, exist_ok=True)

(OUT_BASE / "label_map.json").write_text(
    json.dumps({str(k): v for k, v in id_to_name.items()}, indent=2), encoding="utf-8"
)
print(f"Output directory: {OUT_BASE}")

# ── Extract PNG tiles + CSV labels ────────────────────────────────────────────
all_output = train_tiles + val_tiles + test_tiles
by_slide   = defaultdict(list)
for tile in all_output:
    by_slide[tile["slide_id"]].append(tile)

print(f"Processing {len(by_slide)} slides, {len(all_output):,} tiles total")

tile_records = []
failed       = []

for slide_id, slide_tiles in tqdm(by_slide.items(), desc="Slides"):
    cells_xy     = slide_data[slide_id]["cells_xy"]
    cells_labels = slide_data[slide_id]["cells_labels"]

    try:
        slide = openslide.OpenSlide(slide_data[slide_id]["image_path"])
    except Exception as exc:
        print(f"  ERROR opening {slide_id}: {exc}")
        failed += [(f"{slide_id}_{t['tile_x']}_{t['tile_y']}", str(exc)) for t in slide_tiles]
        continue

    for tile in slide_tiles:
        dup_suffix = tile.get("dup_suffix", "")
        tile_id    = f"{slide_id}_x{tile['tile_x']}_y{tile['tile_y']}{dup_suffix}"
        m          = re.search(r"_dup(\\d+)$", dup_suffix)
        aug_id     = (int(m.group(1)) % 8) if m else 0

        try:
            pil = slide.read_region((tile["tile_x"], tile["tile_y"]), 0, (TILE_SIZE, TILE_SIZE))
            img = np.array(pil.convert("RGB"))
            if img.shape[:2] != (TILE_SIZE, TILE_SIZE):
                img = cv2.resize(img, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_LINEAR)
            img = augment_image(img, aug_id)

            cv2.imwrite(
                str(images_dir / f"{tile_id}.png"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )
            with (labels_dir / f"{tile_id}.csv").open("w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["x_local", "y_local", "class_id"])
                for ci in tile["cell_indices"]:
                    xr = int(cells_xy[ci, 0] - tile["tile_x"])
                    yr = int(cells_xy[ci, 1] - tile["tile_y"])
                    xa, ya = augment_coords(xr, yr, aug_id, TILE_SIZE)
                    writer.writerow([xa, ya, int(cells_labels[ci])])

            tile_records.append({
                "tile_id":      tile_id,
                "slide_id":     slide_id,
                "split":        tile["split"],
                "num_cells":    tile["num_cells"],
                "is_duplicate": bool(dup_suffix),
                "aug_id":       aug_id,
            })
        except Exception as exc:
            failed.append((tile_id, str(exc)))

    slide.close()

n_ok  = len(tile_records)
n_dup = sum(1 for r in tile_records if r["is_duplicate"])
print(f"\\n[DONE] {n_ok:,} tiles saved  ({n_ok-n_dup:,} unique + {n_dup:,} augmented duplicates)")
if failed:
    print(f"Failed: {len(failed)}")
    for tid, err in failed[:5]:
        print(f"  {tid}: {err}")
"""

NB2_SPLITS = """\
# ── Write split CSV files and training config ──────────────────────────────────
df_tiles  = pd.DataFrame(tile_records)

train_ids = df_tiles[df_tiles["split"] == "train"]["tile_id"].tolist()
val_ids   = df_tiles[df_tiles["split"] == "val"]["tile_id"].tolist()
test_ids  = df_tiles[df_tiles["split"] == "test"]["tile_id"].tolist()

fold_dir = splits_dir / "fold_0"
fold_dir.mkdir(parents=True, exist_ok=True)

pd.DataFrame(train_ids).to_csv(fold_dir / "train.csv", index=False, header=False)
pd.DataFrame(val_ids).to_csv(fold_dir / "val.csv",     index=False, header=False)
pd.DataFrame(test_ids).to_csv(splits_dir / "test.csv", index=False, header=False)

config = {
    "logging": {
        "mode": "online",
        "project": "cellvit_GS55",
        "log_comment": "GS55_single_split_20x",
    },
    "data": {
        "dataset_path":   str(OUT_BASE),
        "num_classes":    len(LABELS),
        "train_filelist": str(fold_dir / "train.csv"),
        "val_filelist":   str(fold_dir / "val.csv"),
        "test_filelist":  str(splits_dir / "test.csv"),
        "label_map":      id_to_name,
    },
    "cellvit_path": r"\\\\kittyserverdw\\Andre_kit\\data\\students\\Diogo\\codes\\CellViT_plus_plus\\checkpoints\\CellViT-256-x40-AMP.pth",
    "training": {"batch_size": 32, "epochs": 50, "learning_rate": 0.0001, "fold": 0},
}

config_dir  = OUT_BASE / "train_configs" / "ViT256"
config_dir.mkdir(parents=True, exist_ok=True)
config_path = config_dir / "train_config.yaml"
config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

(OUT_BASE / "label_map.yaml").write_text(
    yaml.safe_dump({"labels": id_to_name}), encoding="utf-8"
)

print(f"TRAIN: {len(train_ids):,} tiles  VAL: {len(val_ids):,}  TEST: {len(test_ids):,}")
print(f"Splits → {splits_dir}")
print(f"Config → {config_path}")
print("\\nDataset creation complete!")
"""

nb2 = nb([
    md("# GS55 — Step 2: Build CellViT tile dataset\n\n"
       "Builds a balanced 256×256 tile dataset from the GS55 whole-slide images\n"
       "and CODA-labelled GeoJSONs produced by Steps 0–1.\n\n"
       "**Pipeline:**\n"
       "1. Load per-slide cell counts → compute 4-tier sampling weights\n"
       "2. Build slide manifest (NDPI + GeoJSON pairs)\n"
       "3. Parse GeoJSON centroids and class labels\n"
       "4. Generate non-overlapping tile grid, filter low-cell tiles\n"
       "5. Weighted unique tile selection (~40k tiles)\n"
       "6. Slide-level train/val/test split\n"
       "7. Oversample rare classes (train only, augmented `_dup{k}` copies)\n"
       "8. Extract PNG tiles + per-tile CSV labels\n"
       "9. Write split CSVs and CellViT training config\n\n"
       "| Cell | Purpose |\n"
       "|------|---------|\n"
       "| 1 | Imports |\n"
       "| **2** | **Parameters ← edit here** |\n"
       "| 3 | Class distribution + sampling weights |\n"
       "| 4 | Build manifest |\n"
       "| 5 | Parse GeoJSON annotations |\n"
       "| 6 | Generate tile candidates |\n"
       "| 7 | Weighted tile selection |\n"
       "| 8 | Train / val / test split |\n"
       "| 9 | Oversample rare classes |\n"
       "| 10 | Extract tiles + save PNGs and CSVs |\n"
       "| 11 | Write splits and training config |"),
    md("---\n## 1 · Imports"),
    code(NB2_IMPORTS),
    md("---\n## 2 · Parameters  ← edit here"),
    code(NB2_PARAMS),
    md("---\n## 3 · Class distribution and sampling weights"),
    code(NB2_WEIGHTS),
    md("---\n## 4 · Build slide manifest"),
    code(NB2_MANIFEST),
    md("---\n## 5 · Parse GeoJSON annotations"),
    code(NB2_PARSE),
    md("---\n## 6 · Generate tile candidates"),
    code(NB2_TILES),
    md("---\n## 7 · Weighted unique tile selection"),
    code(NB2_SELECT),
    md("---\n## 8 · Train / val / test split"),
    code(NB2_SPLIT),
    md("---\n## 9 · Oversample rare classes (train only)"),
    code(NB2_OVERSAMPLE),
    md("---\n## 10 · Extract tiles — save PNGs and CSV labels"),
    code(NB2_EXTRACT),
    md("---\n## 11 · Write split files and training config"),
    code(NB2_SPLITS),
])

(BASE / "2_build_dataset_gs55_256.ipynb").write_text(
    json.dumps(nb2, indent=1, ensure_ascii=False), encoding="utf-8"
)
print("Written: 2_build_dataset_gs55_256.ipynb")
print("\nAll done.")
