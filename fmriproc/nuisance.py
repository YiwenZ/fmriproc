from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import nibabel as nib


@dataclass(frozen=True)
class RegressorMatrix:
    names: list[str]
    rows: list[list[float]]


def build_motion_regressors(motion_rows: list[list[float]], model: str) -> RegressorMatrix:
    clean_rows = [[float(value) for value in row[:6]] for row in motion_rows]
    if model == "none":
        return RegressorMatrix(names=[], rows=[[] for _ in clean_rows])
    if model == "basic_6":
        return RegressorMatrix(
            names=[f"motion_{i:02d}" for i in range(1, 7)],
            rows=clean_rows,
        )
    if model != "friston_24":
        raise ValueError(f"Unsupported nuisance motion_model: {model}")

    derivative_rows: list[list[float]] = []
    for idx, row in enumerate(clean_rows):
        if idx == 0:
            derivative_rows.append([0.0] * 6)
        else:
            previous = clean_rows[idx - 1]
            derivative_rows.append([value - prev for value, prev in zip(row, previous)])

    rows: list[list[float]] = []
    for row, deriv in zip(clean_rows, derivative_rows):
        row_sq = [value * value for value in row]
        deriv_sq = [value * value for value in deriv]
        rows.append(row + deriv + row_sq + deriv_sq)

    names = (
        [f"motion_{i:02d}" for i in range(1, 7)]
        + [f"motion_derivative_{i:02d}" for i in range(1, 7)]
        + [f"motion_squared_{i:02d}" for i in range(1, 7)]
        + [f"motion_derivative_squared_{i:02d}" for i in range(1, 7)]
    )
    return RegressorMatrix(names=names, rows=rows)


def write_regressor_tsv(matrix: RegressorMatrix, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(matrix.names)]
    for row in matrix.rows:
        lines.append("\t".join(str(value) for value in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_regressor_text(matrix: RegressorMatrix, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [" ".join(str(value) for value in row) for row in matrix.rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_regressor_matrices(matrices: list[RegressorMatrix]) -> RegressorMatrix:
    """Horizontally concatenate multiple RegressorMatrix objects."""
    if not matrices:
        return RegressorMatrix(names=[], rows=[])
    n_rows = len(matrices[0].rows)
    for idx, mat in enumerate(matrices[1:], start=1):
        if len(mat.rows) != n_rows:
            raise ValueError(
                f"merge_regressor_matrices: row count mismatch at index {idx} "
                f"(expected {n_rows}, got {len(mat.rows)})"
            )
    merged_names: list[str] = []
    for mat in matrices:
        merged_names.extend(mat.names)
    merged_rows: list[list[float]] = [[] for _ in range(n_rows)]
    for mat in matrices:
        for row_idx, row in enumerate(mat.rows):
            merged_rows[row_idx].extend(row)
    return RegressorMatrix(names=merged_names, rows=merged_rows)

SUPPORTED_NUISANCE_MODELS = {"basic_6", "friston_24", "acompcor", "none"}


def resolve_nuisance_model(nuisance_cfg: dict) -> str:
    model = nuisance_cfg.get("model")
    if model is None:
        # Backward compatibility: older configs enabled aCompCor without a model key.
        if nuisance_cfg.get("acompcor", {}).get("enabled", False):
            model = "acompcor"
        else:
            model = nuisance_cfg.get("motion_model", "basic_6")
    model = str(model)
    if model not in SUPPORTED_NUISANCE_MODELS:
        raise ValueError(
            f"Unsupported nuisance model: {model}. "
            f"Expected one of {sorted(SUPPORTED_NUISANCE_MODELS)}."
        )
    return model


def compute_mask_mean_regressor(bold_path: str, mask_path: str, name: str) -> RegressorMatrix:
    mask_img = nib.load(mask_path)
    mask_arr = np.asarray(mask_img.dataobj)
    mask_idx = np.where(mask_arr > 0)
    if len(mask_idx[0]) == 0:
        raise ValueError(f"Empty mask provided for nuisance regressor: {mask_path}")

    bold_img = nib.load(bold_path, mmap=True)
    bold_data = np.asarray(bold_img.dataobj)
    if mask_arr.shape != bold_data.shape[:3]:
        raise ValueError(f"Shape mismatch: Mask {mask_arr.shape} vs BOLD {bold_data.shape[:3]}")

    ts_matrix = bold_data[mask_idx].T.astype(np.float32)
    mean_ts = ts_matrix.mean(axis=1)
    return RegressorMatrix(names=[name], rows=[[float(value)] for value in mean_ts])
