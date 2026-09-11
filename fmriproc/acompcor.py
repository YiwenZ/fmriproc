from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import nibabel as nib

from .nuisance import RegressorMatrix


@dataclass(frozen=True)
class ACompCorResult:
    """Return type from compute_acompcor_regressors."""
    matrix: RegressorMatrix
    explained_variance_ratio: list[float]   # per-component, summing to ≤1
    total_variance_explained: float          # sum of the above


def compute_acompcor_regressors(
    bold_path: str,
    mask_path: str,
    n_components: int,
    label: str,
) -> ACompCorResult:
    mask_img = nib.load(mask_path)
    mask_arr = np.asarray(mask_img.dataobj, dtype=np.float32)
    mask_idx = np.where(mask_arr > 0)
    n_voxels = len(mask_idx[0])
    if n_voxels == 0:
        raise ValueError(f"Empty mask provided for acompcor: {mask_path}")

    bold_img = nib.load(bold_path, mmap=True)
    bold_data = np.asarray(bold_img.dataobj)
    if mask_arr.shape != bold_data.shape[:3]:
        raise ValueError(f"Shape mismatch: Mask {mask_arr.shape} vs BOLD {bold_data.shape[:3]}")
    ts_matrix = bold_data[mask_idx].T.astype(np.float32)   # shape (T, n_voxels)
    n_timepoints = ts_matrix.shape[0]

    if n_components > min(n_voxels, n_timepoints):
        raise ValueError(
            f"n_components ({n_components}) exceeds min(n_voxels={n_voxels}, "
            f"n_timepoints={n_timepoints}). Reduce n_components."
        )

    ts_matrix -= ts_matrix.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(ts_matrix, full_matrices=False)
    components = U[:, :n_components]   # shape (T, n_components)

    # Explained variance ratio from singular values.
    variance = S ** 2
    total_variance = float(variance.sum())
    if total_variance > 0:
        evr = [float(variance[i] / total_variance) for i in range(n_components)]
    else:
        evr = [0.0] * n_components

    names = [f"acompcor_{label}_{i:02d}" for i in range(n_components)]
    rows = [list(map(float, row)) for row in components]
    matrix = RegressorMatrix(names=names, rows=rows)
    return ACompCorResult(
        matrix=matrix,
        explained_variance_ratio=evr,
        total_variance_explained=sum(evr),
    )


def write_acompcor_sidecar(
    result: ACompCorResult,
    label: str,
    n_components: int,
    output_path: Path,
) -> None:
    """Write a JSON sidecar with per-component explained variance metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "label": label,
        "n_components": n_components,
        "components": [
            {
                "name": result.matrix.names[i],
                "explained_variance_ratio": result.explained_variance_ratio[i],
            }
            for i in range(n_components)
        ],
        "total_variance_explained": result.total_variance_explained,
    }
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
