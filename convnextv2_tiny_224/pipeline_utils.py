"""
pipeline_utils.py — Reusable functions for the Xenium cell type labeling pipeline.

All heavy-lifting logic lives here so the Jupyter notebook stays lean
(orchestration, config, and plots only).
"""

import gc
import hashlib
import json
import time
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import celltypist
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
from pathlib import Path
from scipy.sparse import csr_matrix, issparse
from shapely.geometry import Polygon, MultiPolygon
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(tag, adatas, checkpoint_dir, dataset_names):
    """Write every AnnData in *adatas* to ``checkpoint_dir/tag/{name}.h5ad``."""
    d = Path(checkpoint_dir) / tag
    d.mkdir(parents=True, exist_ok=True)
    for name in dataset_names:
        adatas[name].write_h5ad(d / f"{name}.h5ad")
    print(f"  Checkpoint '{tag}' saved ({len(adatas)} datasets).")


def load_checkpoint(tag, checkpoint_dir, dataset_names):
    """Return dict of AnnData if checkpoint exists, else *None*."""
    d = Path(checkpoint_dir) / tag
    if not d.exists():
        return None
    loaded = {}
    for name in dataset_names:
        p = d / f"{name}.h5ad"
        if not p.exists():
            return None
        loaded[name] = sc.read_h5ad(p)
    print(f"  Checkpoint '{tag}' loaded ({len(loaded)} datasets).")
    return loaded


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_xenium_dataset(name, outs_path):
    """Load a single Xenium dataset from its ``_outs`` directory.

    Reads ``cell_feature_matrix.h5`` via scanpy and merges cell metadata
    from ``cells.parquet`` (spatial coordinates, QC columns).
    """
    outs_path = Path(outs_path)
    h5_path = outs_path / "cell_feature_matrix.h5"
    cells_path = outs_path / "cells.parquet"

    adata = sc.read_10x_h5(str(h5_path))
    adata.var_names_make_unique()

    cells_df = pd.read_parquet(str(cells_path))
    cells_df = cells_df.set_index("cell_id")
    cells_df.index = cells_df.index.astype(str)
    adata.obs.index = adata.obs.index.astype(str)

    shared_idx = adata.obs.index.intersection(cells_df.index)
    adata = adata[shared_idx].copy()
    cells_df = cells_df.loc[shared_idx]

    for col in cells_df.columns:
        adata.obs[col] = cells_df[col].values

    if "x_centroid" in adata.obs.columns:
        adata.obsm["spatial"] = (
            adata.obs[["x_centroid", "y_centroid"]].values.astype(np.float32)
        )

    adata.obs["dataset"] = name
    adata.var["gene_name"] = adata.var_names
    print(f"  {name}: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    return adata


# ---------------------------------------------------------------------------
# Quality Control
# ---------------------------------------------------------------------------

def compute_qc_metrics(adata):
    """Add ``total_counts``, ``n_genes_detected``, and negative-control columns."""
    X = adata.X.toarray() if issparse(adata.X) else adata.X

    adata.obs["total_counts"] = np.asarray(X.sum(axis=1)).flatten()
    adata.obs["n_genes_detected"] = np.asarray((X > 0).sum(axis=1)).flatten()

    neg_mask = adata.var_names.str.startswith(("NegControl", "BLANK", "negcontrol"))
    if neg_mask.any():
        neg_counts = np.asarray(X[:, neg_mask].sum(axis=1)).flatten()
        adata.obs["neg_control_counts"] = neg_counts
        total = adata.obs["total_counts"].values
        adata.obs["neg_control_fraction"] = np.where(total > 0, neg_counts / total, 0.0)
    else:
        adata.obs["neg_control_counts"] = 0.0
        adata.obs["neg_control_fraction"] = 0.0

    return adata


def filter_qc(adata, min_transcripts=10, max_neg_frac=0.05, min_cells_per_gene=5):
    """Apply QC filters and remove negative-control genes.

    Returns
    -------
    adata_filtered : AnnData
    stats : dict  with keys cells_before, cells_after, genes_after, etc.
    """
    n_before = adata.n_obs

    cell_mask = (
        (adata.obs["total_counts"] >= min_transcripts)
        & (adata.obs["neg_control_fraction"] <= max_neg_frac)
    )
    adata = adata[cell_mask].copy()

    X = adata.X.toarray() if issparse(adata.X) else adata.X
    gene_mask = np.asarray((X > 0).sum(axis=0) >= min_cells_per_gene).flatten()
    adata = adata[:, gene_mask].copy()

    neg_mask = adata.var_names.str.startswith(("NegControl", "BLANK", "negcontrol"))
    adata = adata[:, ~neg_mask].copy()

    stats = dict(
        cells_before=n_before,
        cells_after=adata.n_obs,
        cells_removed=n_before - adata.n_obs,
        pct_removed=100 * (n_before - adata.n_obs) / max(n_before, 1),
        genes_after=adata.n_vars,
    )
    return adata, stats


def compute_qc_metrics_sparse(adata):
    """Like ``compute_qc_metrics`` but avoids densifying the full count matrix.

    For nonnegative Xenium counts, per-row/column nonzero counts match
    ``(X > 0).sum`` along that axis. Falls back to dense logic if ``X`` is dense.
    """
    X = adata.X
    if not issparse(X):
        return compute_qc_metrics(adata)

    if not isinstance(X, csr_matrix):
        X = X.tocsr()
        adata.X = X

    adata.obs["total_counts"] = np.asarray(X.sum(axis=1)).ravel()
    adata.obs["n_genes_detected"] = np.asarray(X.getnnz(axis=1)).ravel()

    neg_mask = adata.var_names.str.startswith(("NegControl", "BLANK", "negcontrol"))
    if neg_mask.any():
        neg_counts = np.asarray(X[:, neg_mask].sum(axis=1)).ravel()
        adata.obs["neg_control_counts"] = neg_counts
        total = adata.obs["total_counts"].values
        adata.obs["neg_control_fraction"] = np.where(total > 0, neg_counts / total, 0.0)
    else:
        adata.obs["neg_control_counts"] = 0.0
        adata.obs["neg_control_fraction"] = 0.0

    return adata


def filter_qc_sparse(adata, min_transcripts=10, max_neg_frac=0.05, min_cells_per_gene=5):
    """Like ``filter_qc`` but gene filtering uses sparse column ``nnz`` (no full ``toarray``)."""
    n_before = adata.n_obs

    cell_mask = (
        (adata.obs["total_counts"] >= min_transcripts)
        & (adata.obs["neg_control_fraction"] <= max_neg_frac)
    )
    adata = adata[cell_mask].copy()

    X = adata.X
    if issparse(X):
        if not isinstance(X, csr_matrix):
            X = X.tocsr()
            adata.X = X
        gene_mask = np.asarray(X.getnnz(axis=0) >= min_cells_per_gene).ravel()
    else:
        gene_mask = np.asarray((X > 0).sum(axis=0) >= min_cells_per_gene).flatten()

    adata = adata[:, gene_mask].copy()

    neg_mask = adata.var_names.str.startswith(("NegControl", "BLANK", "negcontrol"))
    adata = adata[:, ~neg_mask].copy()

    stats = dict(
        cells_before=n_before,
        cells_after=adata.n_obs,
        cells_removed=n_before - adata.n_obs,
        pct_removed=100 * (n_before - adata.n_obs) / max(n_before, 1),
        genes_after=adata.n_vars,
    )
    return adata, stats


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_adata(adata, name):
    """Normalize, log-transform, HVG, PCA, neighbors, UMAP, Leiden."""
    print(f"  Preprocessing {name} ({adata.n_obs:,} cells, {adata.n_vars:,} genes)...")

    adata.layers["raw_counts"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata.copy()

    n_hvg = min(2000, adata.n_vars)
    if adata.n_vars > 50:
        sc.pp.highly_variable_genes(
            adata, n_top_genes=n_hvg, flavor="seurat_v3", layer="raw_counts"
        )
        print(f"    HVGs: {adata.var['highly_variable'].sum()} / {adata.n_vars}")
    else:
        adata.var["highly_variable"] = True

    n_pcs = min(50, adata.n_vars - 1, adata.n_obs - 1)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_pcs)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=n_pcs)
    sc.tl.umap(adata)

    for res in [0.3, 0.5, 1.0]:
        key = f"leiden_{res}"
        sc.tl.leiden(adata, resolution=res, key_added=key)
        n_clusters = adata.obs[key].nunique()
        print(f"    Leiden res={res}: {n_clusters} clusters")

    return adata


# ---------------------------------------------------------------------------
# Label Transfer via sc.tl.ingest (kNN-based, no training)
# ---------------------------------------------------------------------------

DS4_TUMOR_LABELS = {
    "Tumor Cells", "Proliferative Tumor Cells", "Inflammatory Tumor Cells",
    "SOX2-OT+ Tumor Cells", "VEGFA+ Tumor Cells", "Malignant Cells Lining Cyst",
}

IMMUNE_CELL_TYPES = {
    "B Cells", "T Cells", "NK Cells", "Plasma Cells",
    "Macrophages / Monocytes", "Dendritic Cells",
}

STROMAL_VASCULAR_CELL_TYPES = {
    "Fibroblasts",
    "Endothelial Cells",
    "Smooth Muscle Cells",
    "Pericytes",
    "Smooth Muscle / Pericytes",  # v6 merged label after apply_pericyte_merge
}

EPITHELIAL_TUMOR_MARKERS = {
    "Epithelial Cells": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT7", "MSLN"],
    "Tumor Cells": ["PAX8", "MKI67", "TOP2A", "WT1", "MUC16", "MUC1"],
}

# Pancreas FFPE / Xenium v1: de-emphasize ovarian-centric tumor drivers (PAX8, WT1) for
# epithelial-vs-tumor scoring; genes absent from the panel are ignored by _compute_marker_score.
EPITHELIAL_TUMOR_MARKERS_PANCREAS = {
    "Epithelial Cells": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT7", "CDH1", "CLDN4", "MSLN"],
    "Tumor Cells": ["MKI67", "TOP2A", "CEACAM5", "EGFR", "MET", "MUC16", "MUC1", "KRT7"],
}

# Post-merge rescue: acinar- and intestinal-enriched cells mis-called Tumor from ingest/CT.
ACINAR_RESCUE_MARKER_GENES = [
    "PRSS1", "PRSS2", "CPA1", "CTRB1", "CTRB2", "AMY2A", "REG1A", "CPB1", "CEL",
]
INTESTINAL_EPITHELIAL_MARKER_GENES = [
    "REG4", "MUC2", "VIL1", "FABP1", "OLFM4", "FABP2", "SPINK4", "DEFA5", "DEFA6",
]

IMMUNE_MARKERS = {
    "T Cells": ["CD3D", "CD3E", "TRBC1", "TRBC2", "CD2"],
    "B Cells": ["MS4A1", "CD79A", "CD74", "CD19"],
    "NK Cells": ["NKG7", "GNLY", "KLRD1", "TRAC"],
    "Plasma Cells": ["JCHAIN", "MZB1", "SDC1", "XBP1"],
    "Macrophages / Monocytes": ["LST1", "TYROBP", "FCER1G", "CD68", "CSF1R"],
    "Dendritic Cells": ["FCER1A", "CD1C", "CLEC10A", "CLEC4C"],
}


def transfer_labels_ingest(target_adata, ref_adata, label_col="cell_type_unified"):
    """Project *target_adata* into *ref_adata*'s PCA space and transfer labels via kNN.

    Returns the transferred labels as a numpy array of strings.
    Requires that *ref_adata* has PCA and neighbors already computed.
    """
    shared_genes = sorted(set(target_adata.var_names) & set(ref_adata.var_names))
    print(f"    Ingest: {len(shared_genes)} shared genes")

    ref_sub = ref_adata[:, shared_genes].copy()
    target_sub = target_adata[:, shared_genes].copy()

    if issparse(ref_sub.X):
        ref_sub.X = ref_sub.X.toarray()
    if issparse(target_sub.X):
        target_sub.X = target_sub.X.toarray()

    n_pcs = min(50, len(shared_genes) - 1, ref_sub.n_obs - 1)
    sc.pp.scale(ref_sub, max_value=10)
    sc.tl.pca(ref_sub, n_comps=n_pcs)
    sc.pp.neighbors(ref_sub, n_neighbors=15, n_pcs=n_pcs)
    sc.tl.umap(ref_sub)

    sc.tl.ingest(target_sub, ref_sub, obs=label_col)

    transferred = target_sub.obs[label_col].values.astype(str)
    print(f"    Ingest: transferred {len(transferred):,} labels")
    return transferred


def transfer_labels_ingest_fast(
    target_adata,
    ref_adata,
    label_col="cell_type_unified",
    ingest_cache_dir=None,
    embedding_method="pca",
):
    """Label transfer with ingest — optimized for repeated panels.

    - ``embedding_method='pca'`` skips reference UMAP; kNN labels use the same PCA
      ``rep`` as the default ingest path, so outputs match the previous pipeline
      in practice while avoiding an expensive UMAP on a freshly subset reference.
    - If *ingest_cache_dir* is set, the scaled subset reference (PCA + neighbors)
      is saved/loaded by hash of shared gene names, so later datasets that share
      the same gene intersection skip rebuilding the 400k×G reference graph.
    """
    shared_genes = sorted(set(target_adata.var_names) & set(ref_adata.var_names))
    print(f"    Ingest: {len(shared_genes)} shared genes")

    cache_path = None
    if ingest_cache_dir is not None:
        ingest_cache_dir = Path(ingest_cache_dir)
        ingest_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = hashlib.sha256("|".join(shared_genes).encode("utf-8")).hexdigest()[:24]
        cache_path = ingest_cache_dir / f"ref_ingest_{cache_key}.h5ad"

    if cache_path is not None and cache_path.exists():
        ref_sub = sc.read_h5ad(cache_path)
        print(f"    Ingest: loaded cached reference sub ({cache_path.name})")
    else:
        ref_sub = ref_adata[:, shared_genes].copy()
        if issparse(ref_sub.X):
            ref_sub.X = ref_sub.X.toarray()
        n_pcs = min(50, len(shared_genes) - 1, ref_sub.n_obs - 1)
        sc.pp.scale(ref_sub, max_value=10)
        sc.tl.pca(ref_sub, n_comps=n_pcs)
        sc.pp.neighbors(ref_sub, n_neighbors=15, n_pcs=n_pcs)
        if cache_path is not None:
            ref_sub.write_h5ad(cache_path)
            print(f"    Ingest: saved reference ingest cache ({cache_path.name})")

    target_sub = target_adata[:, shared_genes].copy()
    if issparse(target_sub.X):
        target_sub.X = target_sub.X.toarray()
    sc.pp.scale(target_sub, max_value=10)

    sc.tl.ingest(
        target_sub,
        ref_sub,
        obs=label_col,
        embedding_method=embedding_method,
    )

    transferred = target_sub.obs[label_col].values.astype(str)
    print(f"    Ingest: transferred {len(transferred):,} labels")
    return transferred


# ---------------------------------------------------------------------------
# CellTypist Annotation
# ---------------------------------------------------------------------------

def run_celltypist_annotation(adata, model, celltypist_map):
    """Run CellTypist and map raw labels to unified cell types.

    Parameters
    ----------
    adata : AnnData  (must be log-normalized, with ``.raw`` set)
    model : celltypist.Model
    celltypist_map : dict  mapping CellTypist label -> unified label

    Returns
    -------
    mapped_labels : np.ndarray of str
    raw_labels : np.ndarray of str  (original CellTypist output)
    """
    input_adata = adata.raw.to_adata() if adata.raw is not None else adata.copy()

    result = celltypist.annotate(input_adata, model=model, majority_voting=False)
    raw_labels = result.predicted_labels["predicted_labels"].values

    mapped = np.array([
        celltypist_map.get(lbl, "Unassigned") for lbl in raw_labels
    ])
    return mapped, raw_labels


# ---------------------------------------------------------------------------
# Annotation Merging
# ---------------------------------------------------------------------------

def merge_annotations(celltypist_labels, ingest_labels, is_healthy=False):
    """Combine CellTypist labels with ingest-transferred DS4 labels.

    For cancer tissue: where ingest transferred a tumor label from DS4,
    override the CellTypist label with "Tumor Cells".
    For healthy tissue: skip tumor override entirely.
    """
    final = celltypist_labels.copy()

    if is_healthy:
        return final

    for i in range(len(final)):
        if ingest_labels[i] in DS4_TUMOR_LABELS:
            final[i] = "Tumor Cells"

    return final


def _compute_marker_score(adata, genes):
    """Compute a simple mean-expression marker score for available genes."""
    source = adata.raw.to_adata() if getattr(adata, "raw", None) is not None else adata
    present = [g for g in genes if g in source.var_names]
    if not present:
        return np.zeros(source.n_obs, dtype=float)
    X = source[:, present].X
    X = X.toarray() if issparse(X) else np.asarray(X)
    return X.mean(axis=1).astype(float)


def merge_annotations_v2(
    adata,
    celltypist_labels,
    ingest_labels,
    is_healthy=False,
    tumor_margin=0.15,
    epithelial_tumor_markers=None,
    require_tumor_evidence_for_non_epithelial_ct=False,
    epithelial_ct_extra_tumor_margin=0.0,
    unassigned_ingest_tumor_rescue=False,
    unassigned_tumor_min_delta=None,
):
    """Safer merge that protects vascular/epithelial classes from overcalling.

    Strategy
    --------
    1) Start from CellTypist labels.
    2) Rescue vascular/stromal classes from ingest when CellTypist says immune.
    3) For tumor override, require evidence when CellTypist says epithelial:
       only override to tumor if tumor_score - epithelial_score > tumor_margin
       (+ optional *epithelial_ct_extra_tumor_margin* for stricter ductal protection).
    4) Optionally require the same marker evidence for non-epithelial CellTypist calls when
       ingest says tumor (v6 pancreas: reduces false tumor from cancer-reference ingest).
    5) Optional: rescue ``Unassigned`` → ``Tumor Cells`` when ingest is tumor-class but
       CellTypist did not map (uses *unassigned_tumor_min_delta*, looser than *tumor_margin*).
    """
    ct = np.asarray(celltypist_labels, dtype=object)
    ig = np.asarray(ingest_labels, dtype=object)
    final = ct.copy()

    markers = epithelial_tumor_markers or EPITHELIAL_TUMOR_MARKERS
    epithelial_score = _compute_marker_score(adata, markers["Epithelial Cells"])
    tumor_score = _compute_marker_score(adata, markers["Tumor Cells"])

    ct_is_immune = np.isin(ct, list(IMMUNE_CELL_TYPES))
    ig_is_structural = np.isin(ig, list(STROMAL_VASCULAR_CELL_TYPES))
    rescue_mask = ct_is_immune & ig_is_structural
    final[rescue_mask] = ig[rescue_mask]

    if is_healthy:
        return final, {
            "n_structural_rescued": int(rescue_mask.sum()),
            "n_tumor_overrides": 0,
            "n_epithelial_tumor_blocked": 0,
            "n_unassigned_tumor_rescued": 0,
        }

    ig_is_tumor = np.isin(ig, list(DS4_TUMOR_LABELS))
    ct_is_epithelial = ct == "Epithelial Cells"
    extra = float(epithelial_ct_extra_tumor_margin)
    tumor_evidence_non_epi = (tumor_score - epithelial_score) > float(tumor_margin)
    tumor_evidence_epi_ct = (tumor_score - epithelial_score) > (float(tumor_margin) + extra)

    epithelial_to_tumor_mask = ig_is_tumor & ct_is_epithelial & tumor_evidence_epi_ct
    epithelial_to_tumor_blocked = ig_is_tumor & ct_is_epithelial & ~tumor_evidence_epi_ct
    if require_tumor_evidence_for_non_epithelial_ct:
        non_epithelial_tumor_mask = ig_is_tumor & ~ct_is_epithelial & tumor_evidence_non_epi
    else:
        non_epithelial_tumor_mask = ig_is_tumor & ~ct_is_epithelial

    final[epithelial_to_tumor_mask] = "Tumor Cells"
    final[non_epithelial_tumor_mask] = "Tumor Cells"

    n_unassigned_rescue = 0
    if unassigned_ingest_tumor_rescue and unassigned_tumor_min_delta is not None:
        umask = (
            (final == "Unassigned")
            & ig_is_tumor
            & ((tumor_score - epithelial_score) > float(unassigned_tumor_min_delta))
        )
        n_unassigned_rescue = int(umask.sum())
        final[umask] = "Tumor Cells"

    return final, {
        "n_structural_rescued": int(rescue_mask.sum()),
        "n_tumor_overrides": int(
            epithelial_to_tumor_mask.sum() + non_epithelial_tumor_mask.sum() + n_unassigned_rescue
        ),
        "n_epithelial_tumor_blocked": int(epithelial_to_tumor_blocked.sum()),
        "n_unassigned_tumor_rescued": n_unassigned_rescue,
    }


def apply_pancreas_tumor_tissue_rescue(
    adata,
    final,
    *,
    epithelial_tumor_markers,
    acinar_rescue_delta=0.08,
    acinar_rescue_soft_delta=None,
    intestinal_rescue_delta=0.06,
    enabled=True,
):
    """Demote spurious ``Tumor Cells`` using acinar / intestinal marker means (panel-aware).

    Runs after ingest-driven tumor overrides so acinar-rich and duodenum-like regions
    can recover ``Acinar Cells`` or ``Epithelial Cells`` when tumor scores do not dominate.

    *acinar_rescue_soft_delta* (optional): second, looser pass for residual acinar→tumor
    stragglers (e.g. 0.04–0.05).
    """
    if not enabled:
        return np.asarray(final, dtype=object), {
            "n_acinar_rescued": 0,
            "n_acinar_soft_rescued": 0,
            "n_intestinal_rescued": 0,
        }

    final = np.asarray(final, dtype=object).copy()
    epi = _compute_marker_score(adata, epithelial_tumor_markers["Epithelial Cells"])
    tum = _compute_marker_score(adata, epithelial_tumor_markers["Tumor Cells"])
    ac = _compute_marker_score(adata, ACINAR_RESCUE_MARKER_GENES)
    gut = _compute_marker_score(adata, INTESTINAL_EPITHELIAL_MARKER_GENES)

    m_tumor = final == "Tumor Cells"
    m_ac = m_tumor & (ac >= tum + float(acinar_rescue_delta))
    n_ac = int(m_ac.sum())
    final[m_ac] = "Acinar Cells"

    n_ac_soft = 0
    if acinar_rescue_soft_delta is not None:
        m_tumor = final == "Tumor Cells"
        m_soft = m_tumor & (ac >= tum + float(acinar_rescue_soft_delta))
        n_ac_soft = int(m_soft.sum())
        final[m_soft] = "Acinar Cells"

    m_tumor = final == "Tumor Cells"
    m_gut = m_tumor & (gut >= tum + float(intestinal_rescue_delta)) & (gut > epi)
    n_gut = int(m_gut.sum())
    final[m_gut] = "Epithelial Cells"

    return final, {
        "n_acinar_rescued": n_ac,
        "n_acinar_soft_rescued": n_ac_soft,
        "n_intestinal_rescued": n_gut,
    }


def compute_immune_evidence(adata):
    """Return immune marker evidence matrix and top-2 margins per cell."""
    classes = list(IMMUNE_MARKERS.keys())
    score_mat = np.column_stack([_compute_marker_score(adata, IMMUNE_MARKERS[c]) for c in classes])

    best_idx = score_mat.argmax(axis=1)
    best_scores = score_mat[np.arange(score_mat.shape[0]), best_idx]

    if score_mat.shape[1] > 1:
        second_scores = np.partition(score_mat, -2, axis=1)[:, -2]
    else:
        second_scores = np.zeros(score_mat.shape[0], dtype=float)
    margins = best_scores - second_scores

    best_labels = np.array([classes[i] for i in best_idx], dtype=object)
    score_df = pd.DataFrame(score_mat, columns=[f"immune_score_{c}" for c in classes], index=adata.obs.index)
    return {
        "classes": classes,
        "score_df": score_df,
        "best_label": best_labels,
        "best_score": best_scores,
        "margin": margins,
    }


def merge_annotations_v3(
    adata,
    celltypist_labels,
    ingest_labels,
    is_healthy=False,
    tumor_margin=0.22,# 0.15
    immune_min_score=0.20,  # Was 0.12
    immune_min_margin=0.07,  # Was 0.03
    immune_reassign_margin=0.09,  # Was 0.05
):
    """Confidence-aware merge with immune guardrails (v3)."""
    final, stats_v2 = merge_annotations_v2(
        adata=adata,
        celltypist_labels=celltypist_labels,
        ingest_labels=ingest_labels,
        is_healthy=is_healthy,
        tumor_margin=tumor_margin,
        require_tumor_evidence_for_non_epithelial_ct=False,
        epithelial_ct_extra_tumor_margin=0.0,
        unassigned_ingest_tumor_rescue=False,
        unassigned_tumor_min_delta=None,
    )

    ct = np.asarray(celltypist_labels, dtype=object)
    ig = np.asarray(ingest_labels, dtype=object)
    final = np.asarray(final, dtype=object)

    ev = compute_immune_evidence(adata)
    score_df = ev["score_df"]
    best_immune = ev["best_label"]
    best_score = ev["best_score"]
    margin = ev["margin"]

    for col in score_df.columns:
        adata.obs[col] = score_df[col].values
    adata.obs["immune_best_label"] = best_immune
    adata.obs["immune_best_score"] = best_score
    adata.obs["immune_score_margin"] = margin

    is_immune_final = np.isin(final, list(IMMUNE_CELL_TYPES))
    is_immune_ingest = np.isin(ig, list(IMMUNE_CELL_TYPES))
    is_nonimmune_ingest = ~is_immune_ingest
    weak_immune = best_score < float(immune_min_score)

    # If an immune call has weak immune evidence, fall back to ingest non-immune class.
    weak_immune_blocked = is_immune_final & weak_immune & is_nonimmune_ingest
    final[weak_immune_blocked] = ig[weak_immune_blocked]

    # Rescue immune label from ingest only when immune evidence is confident.
    confident_immune = (best_score >= float(immune_min_score)) & (margin >= float(immune_min_margin))
    rescue_immune = (~np.isin(final, list(IMMUNE_CELL_TYPES))) & is_immune_ingest & confident_immune
    final[rescue_immune] = ig[rescue_immune]

    # Within-immune subtype correction when marker evidence is clearly stronger.
    score_lookup = {c: np.asarray(score_df[f"immune_score_{c}"].values, dtype=float) for c in IMMUNE_MARKERS}
    subtype_reassigned = np.zeros(adata.n_obs, dtype=bool)
    for c in IMMUNE_CELL_TYPES:
        idx = final == c
        if not np.any(idx):
            continue
        current_score = score_lookup.get(c, np.zeros(adata.n_obs, dtype=float))
        better = (
            idx
            & np.isin(best_immune, list(IMMUNE_CELL_TYPES))
            & ((best_score - current_score) >= float(immune_reassign_margin))
        )
        final[better] = best_immune[better]
        subtype_reassigned |= better

    stats = {
        **stats_v2,
        "n_immune_weak_blocked": int(weak_immune_blocked.sum()),
        "n_immune_rescued": int(rescue_immune.sum()),
        "n_immune_subtype_reassigned": int(subtype_reassigned.sum()),
    }
    return final, stats


# ---------------------------------------------------------------------------
# Streaming v4 (one dataset at a time, memory-efficient)
# ---------------------------------------------------------------------------

STREAMING_V4_REF_NAME_DEFAULT = "DS4_Prime_Cancer_FFPE"


def streaming_v4_ds_ckpt_path(step_dir, name):
    """Path to per-dataset checkpoint ``.h5ad`` under *step_dir*."""
    return Path(step_dir) / f"{name}.h5ad"


def streaming_v4_load_ds_ckpt(step_dir, name):
    """Load per-dataset checkpoint if present, else *None*."""
    p = streaming_v4_ds_ckpt_path(step_dir, name)
    return sc.read_h5ad(p) if p.exists() else None


def streaming_v4_save_ds_ckpt(adata, step_dir, name):
    """Write per-dataset checkpoint; returns path written."""
    p = streaming_v4_ds_ckpt_path(step_dir, name)
    adata.write_h5ad(p)
    return p


def release_adata(adata):
    """Close backed file handles if any, delete object, run GC."""
    try:
        if getattr(adata, "isbacked", False) and getattr(adata, "filename", None) is not None:
            adata.file.close()
    except Exception:
        pass
    del adata
    gc.collect()


def streaming_v4_load_labeled_h5ad(h5ad_dir, name, backed=False):
    """Load final per-dataset labeled ``.h5ad`` from *h5ad_dir*."""
    p = Path(h5ad_dir) / f"{name}.h5ad"
    if not p.exists():
        raise FileNotFoundError(f"Missing labeled h5ad: {p}")
    return sc.read_h5ad(p, backed="r" if backed else None)


def streaming_v4_prepare_reference_ds4(
    datasets,
    cp1_dir,
    cp2_dir,
    cp3_dir,
    ref_dir,
    ds4_cell_type_map,
    min_transcripts,
    max_neg_frac,
    min_cells_per_gene,
    ref_name=STREAMING_V4_REF_NAME_DEFAULT,
):
    """Build or load cached preprocessed DS4 reference for ``sc.tl.ingest``."""
    ref_dir = Path(ref_dir)
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_cache = ref_dir / "DS4_reference_preprocessed.h5ad"

    if ref_cache.exists():
        ref_adata = sc.read_h5ad(ref_cache)
        print(f"Loaded DS4 reference cache: {ref_cache}")
        return ref_name, ref_adata

    print("Preparing DS4 reference cache...")
    info = datasets[ref_name]
    ad = streaming_v4_load_ds_ckpt(cp3_dir, ref_name)
    if ad is None:
        ad = streaming_v4_load_ds_ckpt(cp2_dir, ref_name)
        if ad is None:
            ad = streaming_v4_load_ds_ckpt(cp1_dir, ref_name)
            if ad is None:
                ad = load_xenium_dataset(ref_name, info["outs"])
                streaming_v4_save_ds_ckpt(ad, cp1_dir, ref_name)
            ad = compute_qc_metrics(ad)
            ad, _ = filter_qc(
                ad, min_transcripts, max_neg_frac, min_cells_per_gene
            )
            streaming_v4_save_ds_ckpt(ad, cp2_dir, ref_name)
        ad = preprocess_adata(ad, ref_name)
        streaming_v4_save_ds_ckpt(ad, cp3_dir, ref_name)

    cell_groups = pd.read_csv(str(info["cell_groups_csv"]))
    cell_groups = cell_groups.set_index("cell_id")
    cell_groups.index = cell_groups.index.astype(str)

    shared_cells = ad.obs.index.intersection(cell_groups.index)
    ad.obs["cell_type_fine"] = "Unassigned"
    ad.obs.loc[shared_cells, "cell_type_fine"] = cell_groups.loc[shared_cells, "group"].values
    ad.obs["cell_type_unified"] = (
        ad.obs["cell_type_fine"].map(ds4_cell_type_map).fillna("Unassigned")
    )

    ad.write_h5ad(ref_cache)
    print(f"Saved DS4 reference cache: {ref_cache}")
    return ref_name, ad


def streaming_v4_annotate_export_one_dataset(
    name,
    datasets,
    ref_adata,
    ct_model,
    celltypist_map,
    unified_cell_types,
    cell_type_id_map,
    spatial_dir,
    h5ad_dir,
    cp1_dir,
    cp2_dir,
    cp3_dir,
    cp4_dir,
    min_transcripts,
    max_neg_frac,
    min_cells_per_gene,
    tumor_margin=0.15,
    immune_min_score=0.12,
    immune_min_margin=0.03,
    immune_reassign_margin=0.05,
    he_transform_direction="inverse",
):
    """Process one dataset: QC, preprocess, v3 merge, export CSV + GeoJSON, save h5ad."""
    import json as _json

    info = datasets[name]
    print(f"\n{'='*80}\nProcessing {name}\n{'='*80}")

    ad = streaming_v4_load_ds_ckpt(cp4_dir, name)
    if ad is None:
        ad = streaming_v4_load_ds_ckpt(cp3_dir, name)
        if ad is None:
            ad = streaming_v4_load_ds_ckpt(cp2_dir, name)
            if ad is None:
                ad = streaming_v4_load_ds_ckpt(cp1_dir, name)
                if ad is None:
                    ad = load_xenium_dataset(name, info["outs"])
                    streaming_v4_save_ds_ckpt(ad, cp1_dir, name)
                ad = compute_qc_metrics(ad)
                ad, _ = filter_qc(
                    ad, min_transcripts, max_neg_frac, min_cells_per_gene
                )
                streaming_v4_save_ds_ckpt(ad, cp2_dir, name)

            ad = preprocess_adata(ad, name)
            streaming_v4_save_ds_ckpt(ad, cp3_dir, name)

        is_healthy = info["tissue"] == "healthy"
        ct_labels, ct_raw = run_celltypist_annotation(ad, ct_model, celltypist_map)
        ingest_labels = transfer_labels_ingest(
            ad, ref_adata, label_col="cell_type_unified"
        )

        ad.obs["celltypist_raw"] = ct_raw
        ad.obs["celltypist_mapped"] = ct_labels
        ad.obs["ingest_label"] = ingest_labels

        final, merge_stats = merge_annotations_v3(
            adata=ad,
            celltypist_labels=ct_labels,
            ingest_labels=ingest_labels,
            is_healthy=is_healthy,
            tumor_margin=tumor_margin,
            immune_min_score=immune_min_score,
            immune_min_margin=immune_min_margin,
            immune_reassign_margin=immune_reassign_margin,
        )
        ad.obs["cell_type"] = pd.Categorical(final, categories=unified_cell_types)
        streaming_v4_save_ds_ckpt(ad, cp4_dir, name)

        print(
            f"  v4 merge stats: structural_rescued={merge_stats['n_structural_rescued']:,}, "
            f"tumor_overrides={merge_stats['n_tumor_overrides']:,}, "
            f"epithelial_tumor_blocked={merge_stats['n_epithelial_tumor_blocked']:,}, "
            f"immune_weak_blocked={merge_stats['n_immune_weak_blocked']:,}, "
            f"immune_rescued={merge_stats['n_immune_rescued']:,}, "
            f"immune_subtype_reassigned={merge_stats['n_immune_subtype_reassigned']:,}"
        )
    else:
        print("  Using cached labeled checkpoint.")

    exp_path = Path(info["outs"]) / "experiment.xenium"
    pixel_size = 0.2125
    if exp_path.exists():
        with open(exp_path, encoding="utf-8") as f:
            pixel_size = _json.load(f).get("pixel_size", 0.2125)

    export_spatial_csv(ad, name, spatial_dir, cell_type_id_map)
    export_if_and_he_geojson(
        adata=ad,
        name=name,
        outs_path=info["outs"],
        output_dir=spatial_dir,
        cell_type_id_map=cell_type_id_map,
        pixel_size=pixel_size,
        transform_direction=he_transform_direction,
    )

    h5ad_path = Path(h5ad_dir) / f"{name}.h5ad"
    Path(h5ad_dir).mkdir(parents=True, exist_ok=True)
    ad.write_h5ad(h5ad_path)

    cts = ad.obs["cell_type"].astype(str).value_counts()
    result = {
        "dataset": name,
        "cells": int(ad.n_obs),
        "genes": int(ad.n_vars),
        "tissue": info["tissue"],
        "panel": info["panel"],
        "labeled_frac": float((ad.obs["cell_type"].astype(str) != "Unassigned").mean()),
    }
    for ct in unified_cell_types:
        result[f"n_{ct}"] = int(cts.get(ct, 0))

    release_adata(ad)
    return result


def streaming_v4_run_all_datasets(
    dataset_names,
    datasets,
    ref_adata,
    ct_model,
    celltypist_map,
    unified_cell_types,
    cell_type_id_map,
    spatial_dir,
    h5ad_dir,
    cp1_dir,
    cp2_dir,
    cp3_dir,
    cp4_dir,
    min_transcripts,
    max_neg_frac,
    min_cells_per_gene,
    **annotate_kwargs,
):
    """Run ``streaming_v4_annotate_export_one_dataset`` for every name in *dataset_names*."""
    rows = []
    for name in dataset_names:
        rows.append(
            streaming_v4_annotate_export_one_dataset(
                name=name,
                datasets=datasets,
                ref_adata=ref_adata,
                ct_model=ct_model,
                celltypist_map=celltypist_map,
                unified_cell_types=unified_cell_types,
                cell_type_id_map=cell_type_id_map,
                spatial_dir=spatial_dir,
                h5ad_dir=h5ad_dir,
                cp1_dir=cp1_dir,
                cp2_dir=cp2_dir,
                cp3_dir=cp3_dir,
                cp4_dir=cp4_dir,
                min_transcripts=min_transcripts,
                max_neg_frac=max_neg_frac,
                min_cells_per_gene=min_cells_per_gene,
                **annotate_kwargs,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Streaming v5 (sparse QC + fast ingest + timing)
# ---------------------------------------------------------------------------


def streaming_v5_prepare_reference_ds4(
    datasets,
    cp1_dir,
    cp2_dir,
    cp3_dir,
    ref_dir,
    ds4_cell_type_map,
    min_transcripts,
    max_neg_frac,
    min_cells_per_gene,
    ref_name=STREAMING_V4_REF_NAME_DEFAULT,
):
    """Like v4 reference prep, but raw→QC uses sparse-safe ops (no full densify)."""
    ref_dir = Path(ref_dir)
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_cache = ref_dir / "DS4_reference_preprocessed.h5ad"

    if ref_cache.exists():
        ref_adata = sc.read_h5ad(ref_cache)
        print(f"Loaded DS4 reference cache: {ref_cache}")
        return ref_name, ref_adata

    print("Preparing DS4 reference cache (v5 sparse QC path)...")
    info = datasets[ref_name]
    ad = streaming_v4_load_ds_ckpt(cp3_dir, ref_name)
    if ad is None:
        ad = streaming_v4_load_ds_ckpt(cp2_dir, ref_name)
        if ad is None:
            ad = streaming_v4_load_ds_ckpt(cp1_dir, ref_name)
            if ad is None:
                ad = load_xenium_dataset(ref_name, info["outs"])
                streaming_v4_save_ds_ckpt(ad, cp1_dir, ref_name)
            ad = compute_qc_metrics_sparse(ad)
            ad, _ = filter_qc_sparse(
                ad, min_transcripts, max_neg_frac, min_cells_per_gene
            )
            streaming_v4_save_ds_ckpt(ad, cp2_dir, ref_name)
        ad = preprocess_adata(ad, ref_name)
        streaming_v4_save_ds_ckpt(ad, cp3_dir, ref_name)

    cell_groups = pd.read_csv(str(info["cell_groups_csv"]))
    cell_groups = cell_groups.set_index("cell_id")
    cell_groups.index = cell_groups.index.astype(str)

    shared_cells = ad.obs.index.intersection(cell_groups.index)
    ad.obs["cell_type_fine"] = "Unassigned"
    ad.obs.loc[shared_cells, "cell_type_fine"] = cell_groups.loc[shared_cells, "group"].values
    ad.obs["cell_type_unified"] = (
        ad.obs["cell_type_fine"].map(ds4_cell_type_map).fillna("Unassigned")
    )

    ad.write_h5ad(ref_cache)
    print(f"Saved DS4 reference cache: {ref_cache}")
    return ref_name, ad


def streaming_v5_annotate_export_one_dataset(
    name,
    datasets,
    ref_adata,
    ct_model,
    celltypist_map,
    unified_cell_types,
    cell_type_id_map,
    spatial_dir,
    h5ad_dir,
    cp1_dir,
    cp2_dir,
    cp3_dir,
    cp4_dir,
    min_transcripts,
    max_neg_frac,
    min_cells_per_gene,
    ingest_cache_dir=None,
    tumor_margin=0.15,
    immune_min_score=0.12,
    immune_min_margin=0.03,
    immune_reassign_margin=0.05,
    he_transform_direction="inverse",
):
    """v5: sparse QC, PCA-only ingest + optional disk cache, per-stage timings in result."""
    import json as _json

    info = datasets[name]
    print(f"\n{'='*80}\nProcessing {name} (v5)\n{'='*80}")

    timings = {}
    t_all = time.perf_counter()

    ad = streaming_v4_load_ds_ckpt(cp4_dir, name)
    if ad is None:
        ad = streaming_v4_load_ds_ckpt(cp3_dir, name)
        if ad is None:
            ad = streaming_v4_load_ds_ckpt(cp2_dir, name)
            if ad is None:
                ad = streaming_v4_load_ds_ckpt(cp1_dir, name)
                if ad is None:
                    t0 = time.perf_counter()
                    ad = load_xenium_dataset(name, info["outs"])
                    streaming_v4_save_ds_ckpt(ad, cp1_dir, name)
                    timings["load_raw"] = time.perf_counter() - t0

                t0 = time.perf_counter()
                ad = compute_qc_metrics_sparse(ad)
                ad, _ = filter_qc_sparse(
                    ad, min_transcripts, max_neg_frac, min_cells_per_gene
                )
                timings["qc_sparse"] = time.perf_counter() - t0
                streaming_v4_save_ds_ckpt(ad, cp2_dir, name)

            t0 = time.perf_counter()
            ad = preprocess_adata(ad, name)
            timings["preprocess"] = time.perf_counter() - t0
            streaming_v4_save_ds_ckpt(ad, cp3_dir, name)

        is_healthy = info["tissue"] == "healthy"

        t0 = time.perf_counter()
        ct_labels, ct_raw = run_celltypist_annotation(ad, ct_model, celltypist_map)
        timings["celltypist"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        ingest_labels = transfer_labels_ingest_fast(
            ad,
            ref_adata,
            label_col="cell_type_unified",
            ingest_cache_dir=ingest_cache_dir,
            embedding_method="pca",
        )
        timings["ingest"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        ad.obs["celltypist_raw"] = ct_raw
        ad.obs["celltypist_mapped"] = ct_labels
        ad.obs["ingest_label"] = ingest_labels

        final, merge_stats = merge_annotations_v3(
            adata=ad,
            celltypist_labels=ct_labels,
            ingest_labels=ingest_labels,
            is_healthy=is_healthy,
            tumor_margin=tumor_margin,
            immune_min_score=immune_min_score,
            immune_min_margin=immune_min_margin,
            immune_reassign_margin=immune_reassign_margin,
        )
        ad.obs["cell_type"] = pd.Categorical(final, categories=unified_cell_types)
        timings["merge"] = time.perf_counter() - t0
        streaming_v4_save_ds_ckpt(ad, cp4_dir, name)

        print(
            f"  v5 merge stats: structural_rescued={merge_stats['n_structural_rescued']:,}, "
            f"tumor_overrides={merge_stats['n_tumor_overrides']:,}, "
            f"epithelial_tumor_blocked={merge_stats['n_epithelial_tumor_blocked']:,}, "
            f"immune_weak_blocked={merge_stats['n_immune_weak_blocked']:,}, "
            f"immune_rescued={merge_stats['n_immune_rescued']:,}, "
            f"immune_subtype_reassigned={merge_stats['n_immune_subtype_reassigned']:,}"
        )
    else:
        print("  Using cached labeled checkpoint (v5).")
        timings["note"] = "used_cp4_checkpoint"

    t_exp = time.perf_counter()
    exp_path = Path(info["outs"]) / "experiment.xenium"
    pixel_size = 0.2125
    if exp_path.exists():
        with open(exp_path, encoding="utf-8") as f:
            pixel_size = _json.load(f).get("pixel_size", 0.2125)

    export_spatial_csv(ad, name, spatial_dir, cell_type_id_map)
    export_if_and_he_geojson(
        adata=ad,
        name=name,
        outs_path=info["outs"],
        output_dir=spatial_dir,
        cell_type_id_map=cell_type_id_map,
        pixel_size=pixel_size,
        transform_direction=he_transform_direction,
    )

    h5ad_path = Path(h5ad_dir) / f"{name}.h5ad"
    Path(h5ad_dir).mkdir(parents=True, exist_ok=True)
    ad.write_h5ad(h5ad_path)
    timings["export_io"] = time.perf_counter() - t_exp

    timings["total_wall_s"] = time.perf_counter() - t_all

    cts = ad.obs["cell_type"].astype(str).value_counts()
    result = {
        "dataset": name,
        "cells": int(ad.n_obs),
        "genes": int(ad.n_vars),
        "tissue": info["tissue"],
        "panel": info["panel"],
        "labeled_frac": float((ad.obs["cell_type"].astype(str) != "Unassigned").mean()),
        "timings_sec": timings,
    }
    for ct in unified_cell_types:
        result[f"n_{ct}"] = int(cts.get(ct, 0))

    print(
        "  [v5 timings] "
        + ", ".join(f"{k}={v:.1f}s" if isinstance(v, float) else f"{k}={v!r}" for k, v in timings.items())
    )

    release_adata(ad)
    return result


def streaming_v5_run_all_datasets(
    dataset_names,
    datasets,
    ref_adata,
    ct_model,
    celltypist_map,
    unified_cell_types,
    cell_type_id_map,
    spatial_dir,
    h5ad_dir,
    cp1_dir,
    cp2_dir,
    cp3_dir,
    cp4_dir,
    min_transcripts,
    max_neg_frac,
    min_cells_per_gene,
    ingest_cache_dir=None,
    **annotate_kwargs,
):
    """Run ``streaming_v5_annotate_export_one_dataset`` for every name in *dataset_names*."""
    rows = []
    for name in dataset_names:
        rows.append(
            streaming_v5_annotate_export_one_dataset(
                name=name,
                datasets=datasets,
                ref_adata=ref_adata,
                ct_model=ct_model,
                celltypist_map=celltypist_map,
                unified_cell_types=unified_cell_types,
                cell_type_id_map=cell_type_id_map,
                spatial_dir=spatial_dir,
                h5ad_dir=h5ad_dir,
                cp1_dir=cp1_dir,
                cp2_dir=cp2_dir,
                cp3_dir=cp3_dir,
                cp4_dir=cp4_dir,
                min_transcripts=min_transcripts,
                max_neg_frac=max_neg_frac,
                min_cells_per_gene=min_cells_per_gene,
                ingest_cache_dir=ingest_cache_dir,
                **annotate_kwargs,
            )
        )
    return rows

# ---------------------------------------------------------------------------
# Streaming v6 (stricter thresholds + pericyte merge)
# ---------------------------------------------------------------------------

# v6 default merge parameters (STRICTER than v5)
MERGE_V6_PARAMS_DEFAULT = {
    "tumor_margin": 0.22,           # Was 0.15
    "immune_min_score": 0.20,       # Was 0.12
    "immune_min_margin": 0.07,      # Was 0.03
    "immune_reassign_margin": 0.09, # Was 0.05
}

# Pericyte merge mapping
PERICYTE_MERGE_MAP_DEFAULT = {
    "Pericytes": "Smooth Muscle / Pericytes",
    "Smooth Muscle Cells": "Smooth Muscle / Pericytes",
}


def apply_pericyte_merge(labels, pericyte_merge_map=None):
    """
    Remap pericyte labels to merged class.
    
    Parameters
    ----------
    labels : array-like
        Cell type labels (string).
    pericyte_merge_map : dict, optional
        Mapping from old names to new merged name.
        
    Returns
    -------
    np.ndarray
        Labels with pericytes merged.
    """
    if pericyte_merge_map is None:
        pericyte_merge_map = PERICYTE_MERGE_MAP_DEFAULT
    
    labels = np.asarray(labels, dtype=object)
    for old_name, new_name in pericyte_merge_map.items():
        labels[labels == old_name] = new_name
    return labels


def merge_annotations_v3_strict(
    adata,
    celltypist_labels,
    ingest_labels,
    is_healthy=False,
    tumor_margin=0.22,
    immune_min_score=0.20,
    immune_min_margin=0.07,
    immune_reassign_margin=0.09,
    pericyte_merge_map=None,
    epithelial_tumor_markers=None,
    require_tumor_evidence_for_non_epithelial_ct=True,
    pancreas_tissue_rescue=True,
    acinar_rescue_delta=0.08,
    acinar_rescue_soft_delta=0.045,
    intestinal_rescue_delta=0.06,
    epithelial_ct_extra_tumor_margin=0.06,
    unassigned_ingest_tumor_rescue=True,
    unassigned_tumor_min_delta=0.04,
):
    """
    Confidence-aware merge with immune guardrails (v6 STRICT).

    Pancreas-oriented defaults:
    - ``EPITHELIAL_TUMOR_MARKERS_PANCREAS`` for epithelial vs tumor scoring (when
      ``epithelial_tumor_markers`` is None).
    - ``require_tumor_evidence_for_non_epithelial_ct``: ingest tumor overrides need
      marker evidence for non-epithelial CellTypist labels too.
    - ``epithelial_ct_extra_tumor_margin``: stricter bar for CellTypist epithelial → tumor.
    - ``unassigned_ingest_tumor_rescue`` / ``unassigned_tumor_min_delta``: promote
      ``Unassigned`` to ``Tumor Cells`` when ingest is tumor-like (looser delta).
    - ``apply_pancreas_tumor_tissue_rescue``: acinar / intestinal rescue; optional
      ``acinar_rescue_soft_delta`` second pass for mild acinar signal.
    """
    if pericyte_merge_map is None:
        pericyte_merge_map = PERICYTE_MERGE_MAP_DEFAULT

    markers = epithelial_tumor_markers or EPITHELIAL_TUMOR_MARKERS_PANCREAS

    # Apply pericyte merge to input labels FIRST
    ct = apply_pericyte_merge(celltypist_labels, pericyte_merge_map)
    ig = apply_pericyte_merge(ingest_labels, pericyte_merge_map)

    # Call base v2 merge with remapped labels
    final, stats_v2 = merge_annotations_v2(
        adata=adata,
        celltypist_labels=ct,
        ingest_labels=ig,
        is_healthy=is_healthy,
        tumor_margin=tumor_margin,
        epithelial_tumor_markers=markers,
        require_tumor_evidence_for_non_epithelial_ct=require_tumor_evidence_for_non_epithelial_ct,
        epithelial_ct_extra_tumor_margin=epithelial_ct_extra_tumor_margin,
        unassigned_ingest_tumor_rescue=unassigned_ingest_tumor_rescue,
        unassigned_tumor_min_delta=unassigned_tumor_min_delta,
    )

    final = np.asarray(final, dtype=object)

    # Apply pericyte merge to v2 output (safety pass)
    final = apply_pericyte_merge(final, pericyte_merge_map)

    # ── Immune evidence computation ──
    ev = compute_immune_evidence(adata)
    score_df = ev["score_df"]
    best_immune = ev["best_label"]
    best_score = ev["best_score"]
    margin = ev["margin"]

    for col in score_df.columns:
        adata.obs[col] = score_df[col].values
    adata.obs["immune_best_label"] = best_immune
    adata.obs["immune_best_score"] = best_score
    adata.obs["immune_score_margin"] = margin

    # ── Immune guardrails with STRICTER thresholds ──
    is_immune_final = np.isin(final, list(IMMUNE_CELL_TYPES))
    is_immune_ingest = np.isin(ig, list(IMMUNE_CELL_TYPES))
    is_nonimmune_ingest = ~is_immune_ingest
    weak_immune = best_score < float(immune_min_score)  # STRICTER

    # Weak immune call + non-immune ingest → fall back to ingest
    weak_immune_blocked = is_immune_final & weak_immune & is_nonimmune_ingest
    final[weak_immune_blocked] = ig[weak_immune_blocked]

    # Rescue immune from ingest only when CLEARLY confident (STRICTER)
    confident_immune = (best_score >= float(immune_min_score)) & (margin >= float(immune_min_margin))
    rescue_immune = (~np.isin(final, list(IMMUNE_CELL_TYPES))) & is_immune_ingest & confident_immune
    final[rescue_immune] = ig[rescue_immune]

    # Within-immune subtype correction with STRICTER margin
    score_lookup = {c: np.asarray(score_df[f"immune_score_{c}"].values, dtype=float) for c in IMMUNE_MARKERS}
    subtype_reassigned = np.zeros(adata.n_obs, dtype=bool)
    for c in IMMUNE_CELL_TYPES:
        idx = final == c
        if not np.any(idx):
            continue
        current_score = score_lookup.get(c, np.zeros(adata.n_obs, dtype=float))
        better = (
            idx
            & np.isin(best_immune, list(IMMUNE_CELL_TYPES))
            & ((best_score - current_score) >= float(immune_reassign_margin))  # STRICTER
        )
        final[better] = best_immune[better]
        subtype_reassigned |= better

    # ── Acinar / intestinal (e.g. duodenum) rescue from spurious tumor ──
    final, rescue_stats = apply_pancreas_tumor_tissue_rescue(
        adata,
        final,
        epithelial_tumor_markers=markers,
        acinar_rescue_delta=acinar_rescue_delta,
        acinar_rescue_soft_delta=acinar_rescue_soft_delta,
        intestinal_rescue_delta=intestinal_rescue_delta,
        enabled=pancreas_tissue_rescue,
    )

    # Final pericyte merge pass (ensure no leakage)
    final = apply_pericyte_merge(final, pericyte_merge_map)

    stats = {
        **stats_v2,
        "n_immune_weak_blocked": int(weak_immune_blocked.sum()),
        "n_immune_rescued": int(rescue_immune.sum()),
        "n_immune_subtype_reassigned": int(subtype_reassigned.sum()),
        "n_acinar_rescued": rescue_stats["n_acinar_rescued"],
        "n_acinar_soft_rescued": rescue_stats.get("n_acinar_soft_rescued", 0),
        "n_intestinal_rescued": rescue_stats["n_intestinal_rescued"],
        "n_unassigned_tumor_rescued": stats_v2.get("n_unassigned_tumor_rescued", 0),
        "v6_params": {
            "tumor_margin": tumor_margin,
            "immune_min_score": immune_min_score,
            "immune_min_margin": immune_min_margin,
            "immune_reassign_margin": immune_reassign_margin,
            "require_tumor_evidence_for_non_epithelial_ct": require_tumor_evidence_for_non_epithelial_ct,
            "pancreas_tissue_rescue": pancreas_tissue_rescue,
            "acinar_rescue_delta": acinar_rescue_delta,
            "acinar_rescue_soft_delta": acinar_rescue_soft_delta,
            "intestinal_rescue_delta": intestinal_rescue_delta,
            "epithelial_ct_extra_tumor_margin": epithelial_ct_extra_tumor_margin,
            "unassigned_ingest_tumor_rescue": unassigned_ingest_tumor_rescue,
            "unassigned_tumor_min_delta": unassigned_tumor_min_delta,
        },
    }
    return final, stats


def streaming_v6_prepare_reference_ds4(
    datasets,
    cp1_dir,
    cp2_dir,
    cp3_dir,
    ref_dir,
    ds4_cell_type_map,
    min_transcripts,
    max_neg_frac,
    min_cells_per_gene,
    ref_name,
    pericyte_merge_map=None,
):
    """
    Like v5 reference prep, but applies pericyte merge to reference labels.
    
    Parameters
    ----------
    pericyte_merge_map : dict, optional
        Mapping to merge pericytes into smooth muscle.
    
    Returns
    -------
    ref_name : str
    ref_adata : AnnData
    """
    if pericyte_merge_map is None:
        pericyte_merge_map = PERICYTE_MERGE_MAP_DEFAULT
    
    ref_dir = Path(ref_dir)
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_cache = ref_dir / "DS4_reference_preprocessed_v6.h5ad"

    if ref_cache.exists():
        ref_adata = sc.read_h5ad(ref_cache)
        print(f"Loaded DS4 reference cache (v6): {ref_cache}")
        return ref_name, ref_adata

    print("Preparing DS4 reference cache (v6 with pericyte merge)...")
    info = datasets[ref_name]
    
    # Reuse v5 loading logic
    ad = streaming_v4_load_ds_ckpt(cp3_dir, ref_name)
    if ad is None:
        ad = streaming_v4_load_ds_ckpt(cp2_dir, ref_name)
        if ad is None:
            ad = streaming_v4_load_ds_ckpt(cp1_dir, ref_name)
            if ad is None:
                ad = load_xenium_dataset(ref_name, info["outs"])
                streaming_v4_save_ds_ckpt(ad, cp1_dir, ref_name)
            ad = compute_qc_metrics_sparse(ad)
            ad, _ = filter_qc_sparse(
                ad, min_transcripts, max_neg_frac, min_cells_per_gene
            )
            streaming_v4_save_ds_ckpt(ad, cp2_dir, ref_name)
        ad = preprocess_adata(ad, ref_name)
        streaming_v4_save_ds_ckpt(ad, cp3_dir, ref_name)

    # Load cell groups
    cell_groups = pd.read_csv(str(info["cell_groups_csv"]))
    cell_groups = cell_groups.set_index("cell_id")
    cell_groups.index = cell_groups.index.astype(str)

    shared_cells = ad.obs.index.intersection(cell_groups.index)
    ad.obs["cell_type_fine"] = "Unassigned"
    ad.obs.loc[shared_cells, "cell_type_fine"] = cell_groups.loc[shared_cells, "group"].values
    
    # Map to unified WITH pericyte merge
    unified = ad.obs["cell_type_fine"].map(ds4_cell_type_map).fillna("Unassigned")
    unified = apply_pericyte_merge(unified.values, pericyte_merge_map)
    ad.obs["cell_type_unified"] = pd.Categorical(unified)

    ad.write_h5ad(ref_cache)
    print(f"Saved DS4 reference cache (v6): {ref_cache}")
    
    # Verify no pericytes
    vc = ad.obs["cell_type_unified"].value_counts()
    if "Pericytes" in vc.index:
        print(f"  WARNING: {vc['Pericytes']} Pericytes still present!")
    else:
        print("  ✓ Pericytes successfully merged into Smooth Muscle / Pericytes")
    
    return ref_name, ad


def streaming_v6_annotate_export_one_dataset(
    name,
    datasets,
    ref_adata,
    ct_model,
    celltypist_map,
    unified_cell_types,
    cell_type_id_map,
    spatial_dir,
    h5ad_dir,
    cp1_dir,
    cp2_dir,
    cp3_dir,
    cp4_dir,
    min_transcripts,
    max_neg_frac,
    min_cells_per_gene,
    ingest_cache_dir=None,
    tumor_margin=0.22,
    immune_min_score=0.20,
    immune_min_margin=0.07,
    immune_reassign_margin=0.09,
    pericyte_merge_map=None,
    he_transform_direction="inverse",
    epithelial_tumor_markers=None,
    require_tumor_evidence_for_non_epithelial_ct=True,
    pancreas_tissue_rescue=True,
    acinar_rescue_delta=0.08,
    acinar_rescue_soft_delta=0.045,
    intestinal_rescue_delta=0.06,
    epithelial_ct_extra_tumor_margin=0.06,
    unassigned_ingest_tumor_rescue=True,
    unassigned_tumor_min_delta=0.04,
):
    """
    v6: stricter thresholds + pericyte merge + pancreas-oriented tumor/evidence and tissue rescue.

    Changes from v5:
    - Uses merge_annotations_v3_strict (pancreas marker sets, optional ingest-tumor gating,
      acinar/intestinal rescue from spurious tumor calls).
    - Applies pericyte_merge_map to CellTypist and ingest labels before merge
    - Saves to cp4_dir with '_v6' suffix to avoid overwriting v5 checkpoints
    """
    import json as _json

    if pericyte_merge_map is None:
        pericyte_merge_map = PERICYTE_MERGE_MAP_DEFAULT

    info = datasets[name]
    print(f"\n{'='*80}\nProcessing {name} (v6 STRICT)\n{'='*80}")

    timings = {}
    t_all = time.perf_counter()

    # Check for v6-specific checkpoint
    cp4_path_v6 = Path(cp4_dir) / f"{name}_v6.h5ad"
    ad = None
    if cp4_path_v6.exists():
        ad = sc.read_h5ad(cp4_path_v6)
        print("  Using cached labeled checkpoint (v6).")
        timings["note"] = "used_cp4_v6_checkpoint"
    
    if ad is None:
        # Load through preprocessing pipeline (reuse v5 checkpoints)
        ad = streaming_v4_load_ds_ckpt(cp3_dir, name)
        if ad is None:
            ad = streaming_v4_load_ds_ckpt(cp2_dir, name)
            if ad is None:
                ad = streaming_v4_load_ds_ckpt(cp1_dir, name)
                if ad is None:
                    t0 = time.perf_counter()
                    ad = load_xenium_dataset(name, info["outs"])
                    streaming_v4_save_ds_ckpt(ad, cp1_dir, name)
                    timings["load_raw"] = time.perf_counter() - t0

                t0 = time.perf_counter()
                ad = compute_qc_metrics_sparse(ad)
                ad, _ = filter_qc_sparse(
                    ad, min_transcripts, max_neg_frac, min_cells_per_gene
                )
                timings["qc_sparse"] = time.perf_counter() - t0
                streaming_v4_save_ds_ckpt(ad, cp2_dir, name)

            t0 = time.perf_counter()
            ad = preprocess_adata(ad, name)
            timings["preprocess"] = time.perf_counter() - t0
            streaming_v4_save_ds_ckpt(ad, cp3_dir, name)

        is_healthy = info["tissue"] == "healthy"

        # CellTypist
        t0 = time.perf_counter()
        ct_labels, ct_raw = run_celltypist_annotation(ad, ct_model, celltypist_map)
        # Apply pericyte merge to CellTypist output
        ct_labels = apply_pericyte_merge(ct_labels, pericyte_merge_map)
        timings["celltypist"] = time.perf_counter() - t0

        # Ingest
        t0 = time.perf_counter()
        ingest_labels = transfer_labels_ingest_fast(
            ad,
            ref_adata,
            label_col="cell_type_unified",
            ingest_cache_dir=ingest_cache_dir,
            embedding_method="pca",
        )
        # Apply pericyte merge to ingest output
        ingest_labels = apply_pericyte_merge(ingest_labels, pericyte_merge_map)
        timings["ingest"] = time.perf_counter() - t0

        # v6 STRICT Merge
        t0 = time.perf_counter()
        ad.obs["celltypist_raw"] = ct_raw
        ad.obs["celltypist_mapped"] = pd.Categorical(ct_labels)
        ad.obs["ingest_label"] = pd.Categorical(ingest_labels)

        final, merge_stats = merge_annotations_v3_strict(
            adata=ad,
            celltypist_labels=ct_labels,
            ingest_labels=ingest_labels,
            is_healthy=is_healthy,
            tumor_margin=tumor_margin,
            immune_min_score=immune_min_score,
            immune_min_margin=immune_min_margin,
            immune_reassign_margin=immune_reassign_margin,
            pericyte_merge_map=pericyte_merge_map,
            epithelial_tumor_markers=epithelial_tumor_markers,
            require_tumor_evidence_for_non_epithelial_ct=require_tumor_evidence_for_non_epithelial_ct,
            pancreas_tissue_rescue=pancreas_tissue_rescue,
            acinar_rescue_delta=acinar_rescue_delta,
            acinar_rescue_soft_delta=acinar_rescue_soft_delta,
            intestinal_rescue_delta=intestinal_rescue_delta,
            epithelial_ct_extra_tumor_margin=epithelial_ct_extra_tumor_margin,
            unassigned_ingest_tumor_rescue=unassigned_ingest_tumor_rescue,
            unassigned_tumor_min_delta=unassigned_tumor_min_delta,
        )
        ad.obs["cell_type"] = pd.Categorical(final, categories=unified_cell_types)
        ad.obs["cell_type_id"] = ad.obs["cell_type"].map(cell_type_id_map).astype(int)
        timings["merge"] = time.perf_counter() - t0
        
        # Save v6 checkpoint (separate from v5)
        ad.write_h5ad(cp4_path_v6)

        print(
            f"  v6 merge stats (STRICT): "
            f"structural_rescued={merge_stats.get('n_structural_rescued', 'N/A')}, "
            f"tumor_overrides={merge_stats.get('n_tumor_overrides', 'N/A')}, "
            f"epithelial_tumor_blocked={merge_stats.get('n_epithelial_tumor_blocked', 'N/A')}, "
            f"acinar_rescued={merge_stats.get('n_acinar_rescued', 0):,}, "
            f"acinar_soft={merge_stats.get('n_acinar_soft_rescued', 0):,}, "
            f"intestinal_rescued={merge_stats.get('n_intestinal_rescued', 0):,}, "
            f"unassigned→tumor={merge_stats.get('n_unassigned_tumor_rescued', 0):,}, "
            f"immune_weak_blocked={merge_stats['n_immune_weak_blocked']:,}, "
            f"immune_rescued={merge_stats['n_immune_rescued']:,}, "
            f"immune_subtype_reassigned={merge_stats['n_immune_subtype_reassigned']:,}"
        )
        print(
            f"  v6 params: tumor_margin={tumor_margin}, immune_min_score={immune_min_score}, "
            f"immune_min_margin={immune_min_margin}, immune_reassign_margin={immune_reassign_margin}"
        )

    # ── Export ──
    t_exp = time.perf_counter()
    exp_path = Path(info["outs"]) / "experiment.xenium"
    pixel_size = 0.2125
    if exp_path.exists():
        with open(exp_path, encoding="utf-8") as f:
            pixel_size = _json.load(f).get("pixel_size", 0.2125)

    export_spatial_csv(ad, name, spatial_dir, cell_type_id_map)
    export_if_and_he_geojson(
        adata=ad,
        name=name,
        outs_path=info["outs"],
        output_dir=spatial_dir,
        cell_type_id_map=cell_type_id_map,
        pixel_size=pixel_size,
        transform_direction=he_transform_direction,
    )

    h5ad_path = Path(h5ad_dir) / f"{name}.h5ad"
    Path(h5ad_dir).mkdir(parents=True, exist_ok=True)
    ad.write_h5ad(h5ad_path)
    timings["export_io"] = time.perf_counter() - t_exp

    timings["total_wall_s"] = time.perf_counter() - t_all

    # ── Summary ──
    cts = ad.obs["cell_type"].astype(str).value_counts()
    result = {
        "dataset": name,
        "cells": int(ad.n_obs),
        "genes": int(ad.n_vars),
        "tissue": info["tissue"],
        "panel": info["panel"],
        "labeled_frac": float((ad.obs["cell_type"].astype(str) != "Unassigned").mean()),
        "timings_sec": timings,
    }
    for ct in unified_cell_types:
        result[f"n_{ct}"] = int(cts.get(ct, 0))

    # Verify no pericytes in output
    if "Pericytes" in cts.index:
        print(f"  ⚠️ WARNING: {cts['Pericytes']} Pericytes still in final labels!")
    
    print(
        "  [v6 timings] "
        + ", ".join(f"{k}={v:.1f}s" if isinstance(v, float) else f"{k}={v!r}" for k, v in timings.items())
    )

    release_adata(ad)
    return result


def streaming_v6_run_all_datasets(
    dataset_names,
    datasets,
    ref_adata,
    ct_model,
    celltypist_map,
    unified_cell_types,
    cell_type_id_map,
    spatial_dir,
    h5ad_dir,
    cp1_dir,
    cp2_dir,
    cp3_dir,
    cp4_dir,
    min_transcripts,
    max_neg_frac,
    min_cells_per_gene,
    ingest_cache_dir=None,
    pericyte_merge_map=None,
    **annotate_kwargs,
):
    """
    Run ``streaming_v6_annotate_export_one_dataset`` for every name in *dataset_names*.
    
    Uses stricter default thresholds if not overridden in annotate_kwargs.
    """
    # Apply v6 defaults if not specified
    v6_defaults = {
        "tumor_margin": 0.22,
        "immune_min_score": 0.20,
        "immune_min_margin": 0.07,
        "immune_reassign_margin": 0.09,
        "require_tumor_evidence_for_non_epithelial_ct": True,
        "pancreas_tissue_rescue": True,
        "acinar_rescue_delta": 0.08,
        "acinar_rescue_soft_delta": 0.045,
        "intestinal_rescue_delta": 0.06,
        "epithelial_ct_extra_tumor_margin": 0.06,
        "unassigned_ingest_tumor_rescue": True,
        "unassigned_tumor_min_delta": 0.04,
    }
    for k, v in v6_defaults.items():
        if k not in annotate_kwargs:
            annotate_kwargs[k] = v
    
    if pericyte_merge_map is None:
        pericyte_merge_map = PERICYTE_MERGE_MAP_DEFAULT
    
    rows = []
    for name in dataset_names:
        rows.append(
            streaming_v6_annotate_export_one_dataset(
                name=name,
                datasets=datasets,
                ref_adata=ref_adata,
                ct_model=ct_model,
                celltypist_map=celltypist_map,
                unified_cell_types=unified_cell_types,
                cell_type_id_map=cell_type_id_map,
                spatial_dir=spatial_dir,
                h5ad_dir=h5ad_dir,
                cp1_dir=cp1_dir,
                cp2_dir=cp2_dir,
                cp3_dir=cp3_dir,
                cp4_dir=cp4_dir,
                min_transcripts=min_transcripts,
                max_neg_frac=max_neg_frac,
                min_cells_per_gene=min_cells_per_gene,
                ingest_cache_dir=ingest_cache_dir,
                pericyte_merge_map=pericyte_merge_map,
                **annotate_kwargs,
            )
        )
    return rows
    
# ---------------------------------------------------------------------------
# Marker-Gene Scoring (fallback for V1 panels)
# ---------------------------------------------------------------------------

def marker_gene_scoring(adata, marker_dict, min_markers=2):
    """Score cells by marker-gene sets and assign the highest-scoring type."""
    score_cols, valid_types = [], []
    for ct, markers in marker_dict.items():
        present = [g for g in markers if g in adata.var_names]
        if len(present) < min_markers:
            continue
        score_key = f"score_{ct.replace(' ', '_').replace('/', '_')}"
        sc.tl.score_genes(adata, gene_list=present, score_name=score_key, use_raw=False)
        score_cols.append(score_key)
        valid_types.append(ct)

    if not score_cols:
        return None

    scores = adata.obs[score_cols].values
    best_idx = scores.argmax(axis=1)
    best_scores = scores.max(axis=1)
    labels = np.array([valid_types[i] for i in best_idx])
    labels[best_scores < 0] = "Unassigned"
    return labels


# ---------------------------------------------------------------------------
# Spatial Exports
# ---------------------------------------------------------------------------

def export_spatial_csv(adata, name, output_dir, cell_type_id_map):
    """Write ``cell_types.csv`` with centroid coordinates and cell-type IDs.

    Columns: cell_id, x_centroid, y_centroid, cell_type_id, cell_type_name
    """
    out = Path(output_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(index=adata.obs.index)
    df.index.name = "cell_id"
    df["x_centroid"] = adata.obs["x_centroid"].values if "x_centroid" in adata.obs else 0.0
    df["y_centroid"] = adata.obs["y_centroid"].values if "y_centroid" in adata.obs else 0.0
    df["cell_type_name"] = adata.obs["cell_type"].values
    df["cell_type_id"] = df["cell_type_name"].map(cell_type_id_map).fillna(0).astype(int)

    csv_path = out / "cell_types.csv"
    df.to_csv(csv_path)
    print(f"  CSV saved: {csv_path} ({len(df):,} cells)")
    return csv_path


# RGB 0–255 for QuPath GeoJSON "classification" color; aligned with lab QuPath legend.
# Pancreas-only types (Acinar, Islet) use distinct hues vs. these.
QUPATH_CELL_TYPE_COLORS = {
    "B Cells":                 [70, 130, 180],   # steel blue
    "T Cells":                 [255, 240, 0],   # bright yellow
    "NK Cells":                [0, 215, 255],   # bright cyan / aqua
    "Plasma Cells":            [145, 95, 140],  # mauve / dull purple
    "Macrophages / Monocytes": [255, 0, 200],   # magenta / fuchsia
    "Dendritic Cells":         [135, 206, 250], # light sky blue
    "Fibroblasts":             [50, 220, 90],   # lime / neon green
    "Endothelial Cells":       [235, 35, 35],   # bright red
    "Smooth Muscle Cells":     [72, 190, 175], # seafoam / turquoise
    "Smooth Muscle / Pericytes": [72, 190, 175],  # v6 merged class (same hue as smooth muscle)
    "Pericytes":               [190, 230, 255], # pale light blue
    "Epithelial Cells":        [145, 98, 60],  # medium brown (ductal uses same class)
    "Tumor Cells":             [165, 38, 38],  # dark red / brick
    "Acinar Cells":            [235, 115, 15],  # orange — distinct from brown/red/lime
    "Islet / Endocrine":       [100, 55, 165], # blue-violet — distinct from plasma/mauve
    "Unassigned":              [180, 180, 180],
}


def _sanitize_polygon_ring(coords):
    """Return a valid polygon ring or None if unrecoverable.

    Parameters
    ----------
    coords : list[list[float, float]]
        Polygon ring coordinates (may be invalid/self-intersecting).

    Returns
    -------
    tuple
        (sanitized_coords_or_none, repaired_flag)
    """
    if not coords:
        return None, False

    # Remove consecutive duplicate vertices.
    deduped = [coords[0]]
    for xy in coords[1:]:
        if xy != deduped[-1]:
            deduped.append(xy)

    if len(deduped) < 3:
        return None, False

    # Ensure the ring is closed.
    if deduped[0] != deduped[-1]:
        deduped.append(deduped[0])

    # Must have at least 3 unique points (excluding closure point).
    if len({(xy[0], xy[1]) for xy in deduped[:-1]}) < 3:
        return None, False

    poly = Polygon(deduped)
    if poly.is_valid and not poly.is_empty and poly.area > 0:
        return deduped, False

    # Try a standard repair for self-intersections, bow-ties, etc.
    repaired = poly.buffer(0)
    if repaired.is_empty:
        return None, True

    if isinstance(repaired, MultiPolygon):
        # Keep the dominant area component to preserve a single cell shape.
        repaired = max(repaired.geoms, key=lambda g: g.area)

    if (not repaired.is_valid) or repaired.is_empty or repaired.area <= 0:
        return None, True

    ring = list(repaired.exterior.coords)
    ring = [[round(float(x), 2), round(float(y), 2)] for x, y in ring]
    if len(ring) < 4:
        return None, True
    return ring, True


def _geojson_features_from_vertex_parquet(
    boundaries_path,
    ct_series,
    pixel_size,
    labeled_cells,
    tqdm_desc,
    *,
    boundary_kind,
):
    """Build QuPath-style GeoJSON Feature list from Xenium vertex parquet (cell or nucleus).

    Each feature includes ``properties.cell_id`` and ``properties.boundary_kind`` for
    downstream joins; CellViT++ tiling uses geometry centroids (use nucleus polygons
    for nuclear-centered training points).
    """
    bounds_df = pd.read_parquet(str(boundaries_path))
    bounds_df["cell_id"] = bounds_df["cell_id"].astype(str)
    grouped = bounds_df.groupby("cell_id")
    features = []
    n_total = 0
    n_repaired = 0
    n_skipped = 0
    for cell_id, group in tqdm(grouped, desc=tqdm_desc, leave=False):
        if cell_id not in labeled_cells:
            continue
        n_total += 1
        coords = (group[["vertex_x", "vertex_y"]].values / pixel_size).round(2).tolist()
        coords, was_repaired = _sanitize_polygon_ring(coords)
        if coords is None:
            n_skipped += 1
            continue
        if was_repaired:
            n_repaired += 1
        ct_name = str(ct_series.get(cell_id, "Unassigned"))
        color = QUPATH_CELL_TYPE_COLORS.get(ct_name, [180, 180, 180])

        features.append({
            "type": "Feature",
            "id": "PathCellObject",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {
                "objectType": "annotation",
                "cell_id": cell_id,
                "boundary_kind": boundary_kind,
                "classification": {"name": ct_name, "color": color},
            },
        })
    return features, n_total, n_repaired, n_skipped


def export_geojson(adata, name, outs_path, output_dir, cell_type_id_map,
                   pixel_size=0.2125):
    """Build a QuPath-compatible GeoJSON from cell boundaries + labels.

    Coordinates are converted from Xenium microns to image pixels by dividing
    by *pixel_size* (µm/px, read from ``experiment.xenium``).

    QuPath format: bare JSON array of Feature objects, each with
    ``"id": "PathCellObject"`` and ``"properties": {"objectType": "annotation",
    "classification": {"name": ..., "color": [r, g, b]}}``.

    Feature ``properties`` also include ``cell_id`` and ``boundary_kind`` (``\"cell\"``).

    Falls back to Point features from centroids if ``cell_boundaries.parquet``
    is not available.
    """
    outs_path = Path(outs_path)
    out = Path(output_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    geojson_path = out / "cell_boundaries.geojson"

    ct_series = adata.obs["cell_type"]
    labeled_cells = set(adata.obs.index.astype(str))
    print(f"  Pixel size: {pixel_size} µm/px (dividing coords by this)")

    boundaries_path = outs_path / "cell_boundaries.parquet"
    use_boundaries = boundaries_path.exists()

    if use_boundaries:
        features, n_total, n_repaired, n_skipped = _geojson_features_from_vertex_parquet(
            boundaries_path,
            ct_series,
            pixel_size,
            labeled_cells,
            f"  GeoJSON cell {name}",
            boundary_kind="cell",
        )
        print(
            f"  Polygon sanitize stats (cell): total={n_total:,}, "
            f"repaired={n_repaired:,}, skipped={n_skipped:,}"
        )
    else:
        print(f"  cell_boundaries.parquet not found for {name}, using Point features")
        features = []
        for cell_id in adata.obs.index:
            cell_id = str(cell_id)
            x_c = float(adata.obs.at[cell_id, "x_centroid"]) if "x_centroid" in adata.obs else 0.0
            y_c = float(adata.obs.at[cell_id, "y_centroid"]) if "y_centroid" in adata.obs else 0.0
            x_px = round(x_c / pixel_size, 2)
            y_px = round(y_c / pixel_size, 2)
            ct_name = str(ct_series.get(cell_id, "Unassigned"))
            color = QUPATH_CELL_TYPE_COLORS.get(ct_name, [180, 180, 180])
            features.append({
                "type": "Feature",
                "id": "PathCellObject",
                "geometry": {"type": "Point", "coordinates": [x_px, y_px]},
                "properties": {
                    "objectType": "annotation",
                    "cell_id": cell_id,
                    "boundary_kind": "cell_centroid_fallback",
                    "classification": {"name": ct_name, "color": color},
                },
            })

    with open(geojson_path, "w") as f:
        json.dump(features, f)
    print(f"  GeoJSON saved: {geojson_path} ({len(features):,} features)")
    return geojson_path


def export_nucleus_geojson(
    adata,
    name,
    outs_path,
    output_dir,
    cell_type_id_map,
    pixel_size=0.2125,
):
    """Export nucleus polygons + Xenium labels (IF pixel space), when ``nucleus_boundaries.parquet`` exists.

    10x provides ``nucleus_boundaries.parquet`` alongside ``cell_boundaries.parquet``
    (same ``cell_id`` / vertex columns). Use the H&E-aligned sibling from
    :func:`export_if_and_he_geojson` for CellViT++ on registered H&E.

    Centroids from these polygons approximate the nuclear center better than cell
    polygon centroids or transcript-based ``x_centroid`` when training nucleus-centric models.
    """
    outs_path = Path(outs_path)
    boundaries_path = outs_path / "nucleus_boundaries.parquet"
    if not boundaries_path.exists():
        print(f"  nucleus_boundaries.parquet not found under {outs_path}, skipping nucleus GeoJSON")
        return None

    out = Path(output_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    geojson_path = out / "nucleus_boundaries.geojson"

    ct_series = adata.obs["cell_type"]
    labeled_cells = set(adata.obs.index.astype(str))

    features, n_total, n_repaired, n_skipped = _geojson_features_from_vertex_parquet(
        boundaries_path,
        ct_series,
        pixel_size,
        labeled_cells,
        f"  GeoJSON nucleus {name}",
        boundary_kind="nucleus",
    )
    print(
        f"  Polygon sanitize stats (nucleus): total={n_total:,}, "
        f"repaired={n_repaired:,}, skipped={n_skipped:,}"
    )

    with open(geojson_path, "w") as f:
        json.dump(features, f)
    print(f"  Nucleus GeoJSON saved: {geojson_path} ({len(features):,} features)")
    return geojson_path


def align_geojson_to_he(
    geojson_in_path,
    he_alignment_csv_path,
    geojson_out_path,
    round_decimals=2,
    transform_direction="inverse",
):
    """Apply Xenium H&E affine alignment to an existing GeoJSON.

    Parameters
    ----------
    geojson_in_path : str or Path
        Input GeoJSON path (typically IF/morphology-referenced coordinates).
    he_alignment_csv_path : str or Path
        Path to ``*_he_imagealignment.csv`` (3x3 affine matrix).
    geojson_out_path : str or Path
        Output GeoJSON path aligned to the H&E image space.
    round_decimals : int, default 2
        Coordinate rounding after transformation.
    transform_direction : {"inverse", "forward"}, default "inverse"
        Which matrix direction to apply from ``*_he_imagealignment.csv``.
        For IF->H&E conversion, Xenium alignment CSV is typically H&E->IF,
        so ``"inverse"`` is usually correct.
    """
    geojson_in_path = Path(geojson_in_path)
    he_alignment_csv_path = Path(he_alignment_csv_path)
    geojson_out_path = Path(geojson_out_path)
    geojson_out_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = np.loadtxt(str(he_alignment_csv_path), delimiter=",")
    if matrix.shape != (3, 3):
        raise ValueError(
            f"Expected 3x3 alignment matrix, got {matrix.shape} in {he_alignment_csv_path}"
        )
    if transform_direction not in {"inverse", "forward"}:
        raise ValueError(
            f"transform_direction must be 'inverse' or 'forward', got: {transform_direction}"
        )
    if transform_direction == "inverse":
        matrix = np.linalg.inv(matrix)

    with open(geojson_in_path, encoding="utf-8") as f:
        features = json.load(f)

    if not isinstance(features, list):
        raise ValueError(f"GeoJSON root must be a list for QuPath: {geojson_in_path}")

    def _transform_xy(x, y):
        vec = np.array([float(x), float(y), 1.0], dtype=float)
        out = matrix @ vec
        return [round(float(out[0]), round_decimals), round(float(out[1]), round_decimals)]

    transformed = 0
    for ft in features:
        geom = ft.get("geometry", {})
        gtype = geom.get("type")

        if gtype == "Point":
            x, y = geom.get("coordinates", [0.0, 0.0])[:2]
            geom["coordinates"] = _transform_xy(x, y)
            transformed += 1
        elif gtype == "Polygon":
            rings = geom.get("coordinates", [])
            new_rings = []
            for ring in rings:
                new_ring = [_transform_xy(xy[0], xy[1]) for xy in ring]
                new_rings.append(new_ring)
            geom["coordinates"] = new_rings
            transformed += 1

    with open(geojson_out_path, "w", encoding="utf-8") as f:
        json.dump(features, f)

    print(
        f"  H&E-aligned GeoJSON saved: {geojson_out_path} "
        f"({transformed:,} transformed features; direction={transform_direction})"
    )
    return geojson_out_path


def get_he_alignment_csv_path(outs_path):
    """Infer ``*_he_imagealignment.csv`` path from Xenium ``*_outs`` path."""
    outs_path = Path(outs_path)
    dataset_dir = outs_path.parent
    return dataset_dir / f"{dataset_dir.name}_he_imagealignment.csv"


def export_if_and_he_geojson(
    adata,
    name,
    outs_path,
    output_dir,
    cell_type_id_map,
    pixel_size=0.2125,
    he_alignment_csv_path=None,
    transform_direction="inverse",
):
    """Export IF-space GeoJSON then H&E-aligned GeoJSON for one dataset.

    Writes cell polygons (``cell_boundaries*.geojson``). When
    ``nucleus_boundaries.parquet`` exists under *outs_path*, also writes
    ``nucleus_boundaries.geojson`` and ``nucleus_boundaries_he_aligned.geojson``.
    Use the nucleus H&E file with CellViT++ tiling for nuclear-centered centroids.
    """
    if_geojson = export_geojson(
        adata=adata,
        name=name,
        outs_path=outs_path,
        output_dir=output_dir,
        cell_type_id_map=cell_type_id_map,
        pixel_size=pixel_size,
    )

    if_nucleus_geojson = export_nucleus_geojson(
        adata=adata,
        name=name,
        outs_path=outs_path,
        output_dir=output_dir,
        cell_type_id_map=cell_type_id_map,
        pixel_size=pixel_size,
    )

    if he_alignment_csv_path is None:
        he_alignment_csv_path = get_he_alignment_csv_path(outs_path)
    he_alignment_csv_path = Path(he_alignment_csv_path)

    he_geojson = None
    he_nucleus_geojson = None
    if he_alignment_csv_path.exists():
        he_geojson = Path(output_dir) / name / "cell_boundaries_he_aligned.geojson"
        align_geojson_to_he(
            geojson_in_path=if_geojson,
            he_alignment_csv_path=he_alignment_csv_path,
            geojson_out_path=he_geojson,
            round_decimals=2,
            transform_direction=transform_direction,
        )
        if if_nucleus_geojson is not None:
            he_nucleus_geojson = Path(output_dir) / name / "nucleus_boundaries_he_aligned.geojson"
            align_geojson_to_he(
                geojson_in_path=if_nucleus_geojson,
                he_alignment_csv_path=he_alignment_csv_path,
                geojson_out_path=he_nucleus_geojson,
                round_decimals=2,
                transform_direction=transform_direction,
            )
    else:
        print(f"  H&E alignment CSV not found, skipping H&E GeoJSON: {he_alignment_csv_path}")

    return {
        "if_geojson": Path(if_geojson),
        "he_geojson": Path(he_geojson) if he_geojson is not None else None,
        "if_nucleus_geojson": Path(if_nucleus_geojson) if if_nucleus_geojson is not None else None,
        "he_nucleus_geojson": Path(he_nucleus_geojson) if he_nucleus_geojson is not None else None,
        "he_alignment_csv": he_alignment_csv_path,
    }


def validate_geojson_file(geojson_path, sample_size=5000, seed=0):
    """Validate GeoJSON schema + sampled geometry validity."""
    geojson_path = Path(geojson_path)
    with open(geojson_path, encoding="utf-8") as f:
        data = json.load(f)

    result = {
        "path": geojson_path,
        "root_is_list": isinstance(data, list),
        "total_features": 0,
        "geometry_types": set(),
        "invalid_sample": 0,
        "sample_size": 0,
    }
    if not isinstance(data, list):
        return result

    result["total_features"] = len(data)
    geom_types = set()
    poly_indices = []

    for i, ft in enumerate(data):
        g = ft.get("geometry", {})
        gtype = g.get("type", "<missing>")
        geom_types.add(gtype)
        if gtype == "Polygon":
            poly_indices.append(i)

    result["geometry_types"] = geom_types

    if not poly_indices:
        return result

    rng = np.random.default_rng(seed)
    n = min(sample_size, len(poly_indices))
    picked = rng.choice(poly_indices, size=n, replace=False)

    invalid = 0
    for idx in picked:
        coords = data[int(idx)]["geometry"]["coordinates"][0]
        try:
            poly = Polygon(coords)
            if (not poly.is_valid) or poly.is_empty or poly.area <= 0:
                invalid += 1
        except Exception:
            invalid += 1

    result["sample_size"] = n
    result["invalid_sample"] = int(invalid)
    return result


def validate_if_he_geojson_outputs(dataset_names, spatial_dir, sample_size=5000, include_nucleus=True):
    """Validate IF and H&E GeoJSON outputs for all datasets and print summary.

    When *include_nucleus* is True, also validates ``nucleus_boundaries*.geojson`` if those
    files exist (they are only written when ``nucleus_boundaries.parquet`` is present under outs).
    """
    spatial_dir = Path(spatial_dir)
    summary = {}

    for name in dataset_names:
        ds_dir = spatial_dir / name
        if_path = ds_dir / "cell_boundaries.geojson"
        he_path = ds_dir / "cell_boundaries_he_aligned.geojson"

        if_res = validate_geojson_file(if_path, sample_size=sample_size, seed=0) if if_path.exists() else None
        he_res = validate_geojson_file(he_path, sample_size=sample_size, seed=1) if he_path.exists() else None

        entry = {"if": if_res, "he": he_res}
        nuc_if_res = nuc_he_res = None
        if include_nucleus:
            nuc_if = ds_dir / "nucleus_boundaries.geojson"
            nuc_he = ds_dir / "nucleus_boundaries_he_aligned.geojson"
            nuc_if_res = (
                validate_geojson_file(nuc_if, sample_size=sample_size, seed=2) if nuc_if.exists() else None
            )
            nuc_he_res = (
                validate_geojson_file(nuc_he, sample_size=sample_size, seed=3) if nuc_he.exists() else None
            )
            entry["nucleus_if"] = nuc_if_res
            entry["nucleus_he"] = nuc_he_res

        summary[name] = entry

        def _fmt(res):
            if res is None:
                return "missing"
            geom_types = ",".join(sorted(res["geometry_types"])) if res["geometry_types"] else "none"
            return (
                f"root_list={res['root_is_list']}, n={res['total_features']:,}, "
                f"types={geom_types}, invalid_sample={res['invalid_sample']}/{res['sample_size']}"
            )

        line = f"  {name} | IF: {_fmt(if_res)} | HE: {_fmt(he_res)}"
        if include_nucleus:
            line += f" | Nuc_IF: {_fmt(nuc_if_res)} | Nuc_HE: {_fmt(nuc_he_res)}"
        print(line)

    return summary


# ---------------------------------------------------------------------------
# Publication-Quality Figures
# ---------------------------------------------------------------------------

MARKER_GENES = {
    "B Cells": ["CD19", "MS4A1", "CD79A", "PAX5"],
    "T Cells": ["CD3D", "CD3E", "CD3G", "CD2"],
    "NK Cells": ["NKG7", "GNLY", "KLRD1", "NCAM1"],
    "Plasma Cells": ["JCHAIN", "MZB1", "SDC1", "IGHA1"],
    "Macrophages / Monocytes": ["CD68", "CD14", "CSF1R", "FCGR3A"],
    "Dendritic Cells": ["CLEC4C", "FCER1A", "CD1C", "ITGAX"],
    "Fibroblasts": ["COL1A1", "DCN", "LUM", "FAP"],
    "Endothelial Cells": ["PECAM1", "VWF", "CDH5", "FLT1"],
    "Smooth Muscle Cells": ["ACTA2", "MYH11", "TAGLN", "CNN1"],
    "Pericytes": ["RGS5", "PDGFRB", "MCAM", "CSPG4"],
    "Epithelial Cells": ["EPCAM", "KRT18", "KRT8", "CDH1"],
    "Tumor Cells": ["PAX8", "MKI67", "TOP2A", "WT1", "MUC16"],
}


def _qupath_rgb_to_hex(rgb):
    """Convert QuPath-style [R, G, B] to matplotlib hex (clips to 0–255)."""
    r, g, b = (int(max(0, min(255, x))) for x in rgb[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def _get_pub_palette(cell_types):
    """Matplotlib hex colors matching ``QUPATH_CELL_TYPE_COLORS`` (QuPath / GeoJSON)."""
    palette = {}
    for ct in cell_types:
        rgb = QUPATH_CELL_TYPE_COLORS.get(ct)
        if rgb is not None:
            palette[ct] = _qupath_rgb_to_hex(rgb)
        else:
            palette[ct] = "#b4b4b4"
    return palette


def pub_umap_labeled(adata, name, cell_types, out_dir):
    """UMAP with cell type names placed at cluster centroids (no legend box)."""
    palette = _get_pub_palette(cell_types)
    coords = adata.obsm["X_umap"]
    labels = adata.obs["cell_type"].astype(str)

    fig, ax = plt.subplots(figsize=(10, 9))
    for ct in cell_types:
        mask = labels == ct
        if mask.sum() == 0:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1], s=0.5, c=palette[ct],
                   alpha=0.6, rasterized=True, linewidths=0)

    for ct in cell_types:
        if ct == "Unassigned":
            continue
        mask = labels == ct
        if mask.sum() < 5:
            continue
        cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
        txt = ax.text(cx, cy, ct, fontsize=7, fontweight="bold", ha="center", va="center")
        txt.set_path_effects([
            pe.withStroke(linewidth=2.5, foreground="white"),
        ])

    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    ax.set_title(name, fontsize=14, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    fig.savefig(out_dir / f"pub_umap_labeled_{name}.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def pub_spatial_map(adata, name, cell_types, out_dir):
    """Cell types at real tissue x,y coordinates."""
    palette = _get_pub_palette(cell_types)
    if "spatial" not in adata.obsm:
        print(f"  Skipping spatial map for {name} (no spatial coords)")
        return
    coords = adata.obsm["spatial"]
    labels = adata.obs["cell_type"].astype(str)

    fig, ax = plt.subplots(figsize=(14, 11))
    for ct in cell_types:
        mask = labels == ct
        if mask.sum() == 0:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1], s=0.3, c=palette[ct],
                   label=ct, alpha=0.7, rasterized=True, linewidths=0)

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(f"{name} — Spatial Cell Type Map", fontsize=14, fontweight="bold")
    ax.set_xlabel("X (microns)", fontsize=11)
    ax.set_ylabel("Y (microns)", fontsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend = ax.legend(markerscale=12, fontsize=8, loc="upper left",
                       bbox_to_anchor=(1.01, 1), frameon=True, fancybox=False,
                       edgecolor="#cccccc", title="Cell Type", title_fontsize=9)
    for lh in legend.legend_handles:
        lh.set_alpha(1)

    plt.tight_layout()
    fig.savefig(out_dir / f"pub_spatial_map_{name}.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def pub_dotplot_markers(adatas, cell_types, out_dir):
    """Dot plot of marker genes per cell type across all datasets combined."""
    adata_list = []
    for name, ad in adatas.items():
        a = ad.raw.to_adata() if ad.raw is not None else ad.copy()
        a.obs["cell_type"] = ad.obs["cell_type"].values
        adata_list.append(a)

    combined = sc.concat(adata_list, join="inner")
    observed = combined.obs["cell_type"].astype(str)
    present_cell_types = [ct for ct in cell_types if (observed == ct).any()]
    if not present_cell_types:
        print("  No cell types present in combined data — skipping dot plot")
        return
    combined.obs["cell_type"] = pd.Categorical(
        observed, categories=present_cell_types
    )

    flat_markers = {}
    for ct, genes in MARKER_GENES.items():
        present = [g for g in genes if g in combined.var_names]
        if present:
            flat_markers[ct] = present

    if not flat_markers:
        print("  No marker genes found in data — skipping dot plot")
        return

    fig = sc.pl.dotplot(
        combined,
        var_names=flat_markers,
        groupby="cell_type",
        categories_order=present_cell_types,
        standard_scale="var",
        return_fig=True,
    )
    fig.savefig(out_dir / "pub_dotplot_markers.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def pub_composition_heatmap(adatas, cell_types, out_dir):
    """Cell type composition heatmap across datasets (clustered)."""
    proportions = {}
    for name, ad in adatas.items():
        cts = ad.obs["cell_type"].astype(str).value_counts(normalize=True)
        proportions[name] = cts

    prop_df = pd.DataFrame(proportions).fillna(0).T
    cols = [c for c in cell_types if c in prop_df.columns and c != "Unassigned"]
    prop_df = prop_df[cols]

    g = sns.clustermap(
        prop_df, cmap="YlOrRd", annot=True, fmt=".2f", linewidths=0.5,
        figsize=(12, 6), dendrogram_ratio=(0.1, 0.15),
        cbar_kws={"label": "Proportion", "shrink": 0.6},
        xticklabels=True, yticklabels=True,
    )
    g.ax_heatmap.set_ylabel("")
    g.ax_heatmap.set_xlabel("")
    g.fig.suptitle("Cell Type Composition Across Datasets", fontsize=14,
                   fontweight="bold", y=1.02)
    g.savefig(out_dir / "pub_composition_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def pub_stacked_bar(adatas, cell_types, out_dir):
    """Polished stacked bar chart of cell type proportions."""
    palette = _get_pub_palette(cell_types)
    proportions = {}
    for name, ad in adatas.items():
        cts = ad.obs["cell_type"].astype(str).value_counts(normalize=True)
        proportions[name] = cts

    prop_df = pd.DataFrame(proportions).fillna(0).T
    cols = [c for c in cell_types if c in prop_df.columns]
    prop_df = prop_df[cols]

    fig, ax = plt.subplots(figsize=(12, 6))
    prop_df.plot(
        kind="bar", stacked=True, ax=ax,
        color=[palette.get(c, "#999999") for c in cols],
        width=0.75, edgecolor="white", linewidth=0.5,
    )
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_xlabel("")
    ax.set_title("Cell Type Proportions Across Datasets", fontsize=14, fontweight="bold")
    ax.legend(
        title="Cell Type", bbox_to_anchor=(1.01, 1), loc="upper left",
        fontsize=8, title_fontsize=9, frameon=True, fancybox=False,
        edgecolor="#cccccc",
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(out_dir / "pub_stacked_bar.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def pub_colocalization(adata, name, cell_types, out_dir):
    """Spatial co-localization heatmap using squidpy neighborhood enrichment."""
    if "spatial" not in adata.obsm:
        print(f"  Skipping co-localization for {name} (no spatial coords)")
        return

    ad = adata.copy()
    ad.obs["cell_type"] = pd.Categorical(ad.obs["cell_type"].astype(str), categories=cell_types)

    sq.gr.spatial_neighbors(ad, coord_type="generic", n_neighs=10)
    try:
        sq.gr.nhood_enrichment(ad, cluster_key="cell_type")
    except Exception as e:
        print(f"  Co-localization failed for {name}: {e}")
        return

    zscore = ad.uns["cell_type_nhood_enrichment"]["zscore"]
    cats = list(ad.obs["cell_type"].cat.categories)
    present = [c for c in cats if c in cell_types and c != "Unassigned"]
    idx = [cats.index(c) for c in present if cats.index(c) < zscore.shape[0]]
    zscore_sub = zscore[np.ix_(idx, idx)]

    fig, ax = plt.subplots(figsize=(10, 9))
    vmax = min(np.abs(zscore_sub).max(), 50)
    sns.heatmap(
        zscore_sub, xticklabels=present, yticklabels=present,
        cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
        annot=True, fmt=".1f", linewidths=0.5, ax=ax,
        cbar_kws={"label": "Neighborhood enrichment (z-score)", "shrink": 0.7},
    )
    ax.set_title(f"{name} — Spatial Co-localization", fontsize=14, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / f"pub_colocalization_{name}.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def pub_umap_combined(adatas, cell_types, out_dir):
    """Combined UMAP of all datasets — batch-colored + cell-type-colored."""
    palette = _get_pub_palette(cell_types)

    adata_list = []
    for name, ad in adatas.items():
        a = ad.copy()
        a.obs["dataset"] = name
        adata_list.append(a)

    combined = sc.concat(adata_list, join="inner")
    combined.obs["cell_type"] = pd.Categorical(
        combined.obs["cell_type"].astype(str), categories=cell_types
    )

    n_pcs = min(50, combined.n_vars - 1, combined.n_obs - 1)
    sc.pp.scale(combined, max_value=10)
    sc.tl.pca(combined, n_comps=n_pcs)
    sc.pp.neighbors(combined, n_neighbors=15, n_pcs=n_pcs)
    sc.tl.umap(combined)

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    sc.pl.umap(combined, color="dataset", ax=axes[0], show=False,
               title="Colored by Dataset", size=1, alpha=0.5)
    sc.pl.umap(combined, color="cell_type", ax=axes[1], show=False,
               title="Colored by Cell Type", size=1, alpha=0.5,
               palette=palette)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.suptitle("Combined UMAP — All Datasets", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "pub_umap_combined.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
