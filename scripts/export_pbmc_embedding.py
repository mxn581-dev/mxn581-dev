"""Export a real, public scRNA-seq UMAP as ``x,y,label`` CSV for the hero SVG.

Uses scanpy's processed 10x PBMC 3k dataset: public, small (~24 MB), with a
precomputed UMAP and annotated clusters. scanpy is needed only here, in CI,
so the SVG generator itself stays numpy-only.

Usage:
    python export_pbmc_embedding.py --out data/pbmc.csv --n-cells 800
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def export(out: Path, n_cells: int, seed: int) -> int:
    """Subsample PBMC 3k and write its UMAP coordinates with cell type labels.

    Returns:
        Number of cells written.

    Raises:
        ImportError: If scanpy is not installed.
        ValueError: If n_cells is not positive or the dataset lacks a UMAP.
    """
    if n_cells <= 0:
        raise ValueError(f"n_cells must be positive, got {n_cells}")
    import scanpy as sc  # deferred: heavy import, only this script needs it

    adata = sc.datasets.pbmc3k_processed()
    if "X_umap" not in adata.obsm or "louvain" not in adata.obs:
        raise ValueError("pbmc3k_processed is missing X_umap or louvain labels")

    rng = np.random.default_rng(seed)
    n = min(n_cells, adata.n_obs)
    # Uniform subsampling keeps true cluster proportions; a stratified sample
    # would overstate rare types like megakaryocytes.
    idx = np.sort(rng.choice(adata.n_obs, size=n, replace=False))
    xy = adata.obsm["X_umap"][idx]
    labels = adata.obs["louvain"].astype(str).to_numpy()[idx]

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y", "label"])
        for (x, y), lab in zip(xy, labels):
            w.writerow([f"{x:.4f}", f"{y:.4f}", lab])
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-cells", type=int, default=800)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    try:
        n = export(args.out, args.n_cells, args.seed)
    except (ImportError, ValueError, OSError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(f"wrote {n} cells to {args.out}")


if __name__ == "__main__":
    main()
