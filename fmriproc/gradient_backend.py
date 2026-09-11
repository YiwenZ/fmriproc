"""BrainSpace backend for dataset-level FC gradients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GradientResult:
    gradients: np.ndarray
    lambdas: np.ndarray


def _check_brainspace_available() -> None:
    """Raise a clear error if BrainSpace is unavailable."""
    try:
        import brainspace  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "BrainSpace is required for gradient metrics. Install it with: pip install brainspace"
        ) from exc


def fisherz_to_affinity(matrix: np.ndarray) -> np.ndarray:
    """Convert Fisher-z FC matrix to a non-negative symmetric affinity matrix."""
    arr = np.array(matrix, dtype=float, copy=True)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"FC matrix must be square; got shape {arr.shape}")
    arr = (arr + arr.T) / 2.0
    np.fill_diagonal(arr, 0.0)
    arr[arr < 0] = 0.0
    return arr


def fit_gradient(matrix: np.ndarray, cfg: dict[str, Any]) -> GradientResult:
    """Fit a reference gradient from a group-average FC matrix."""
    _check_brainspace_available()
    from brainspace.gradient import GradientMaps

    gcfg = cfg.get("metrics", {}).get("gradient", {})
    gm = GradientMaps(
        n_components=int(gcfg.get("n_components", 3)),
        approach=gcfg.get("approach", "dm"),
        kernel=gcfg.get("kernel", "normalized_angle"),
        alignment=None,
        random_state=int(gcfg.get("random_state", 42)),
    )
    gm.fit(fisherz_to_affinity(matrix), sparsity=float(gcfg.get("sparsity", 0.9)))
    return GradientResult(gradients=np.asarray(gm.gradients_), lambdas=np.asarray(gm.lambdas_))


def align_to_reference(matrix: np.ndarray, reference_gradients: np.ndarray, cfg: dict[str, Any]) -> GradientResult:
    """Fit a run gradient and align it to reference gradients."""
    _check_brainspace_available()
    from brainspace.gradient import GradientMaps

    gcfg = cfg.get("metrics", {}).get("gradient", {})
    gm = GradientMaps(
        n_components=int(gcfg.get("n_components", 3)),
        approach=gcfg.get("approach", "dm"),
        kernel=gcfg.get("kernel", "normalized_angle"),
        alignment=gcfg.get("alignment", "procrustes"),
        random_state=int(gcfg.get("random_state", 42)),
    )
    gm.fit(
        fisherz_to_affinity(matrix),
        reference=reference_gradients,
        sparsity=float(gcfg.get("sparsity", 0.9)),
    )
    gradients = np.asarray(getattr(gm, "aligned_", gm.gradients_))
    return GradientResult(gradients=gradients, lambdas=np.asarray(gm.lambdas_))
