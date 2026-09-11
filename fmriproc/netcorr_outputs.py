from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class StandardizedNetCorrOutputs:
    timeseries_tsv: Path
    fc_correlation_tsv: Path
    fc_fisherz_tsv: Path
    fc_long_tsv: Path
    n_timepoints: int
    n_rois: int
    n_edges: int
    n_constant_rois: int
    n_missing_rois: int


def standardize_netcorr_outputs(
    netts_path: Path,
    netcc_path: Path,
    timeseries_tsv: Path,
    fc_correlation_tsv: Path,
    fc_fisherz_tsv: Path,
    fc_long_tsv: Path,
    subject: str,
    session: str,
    task: str,
    atlas: str,
    label_path: Path | None = None,
) -> StandardizedNetCorrOutputs:
    timeseries = parse_netts(netts_path)
    matrix = parse_netcc(netcc_path)
    n_rois = matrix.shape[0]
    labels = load_roi_labels(label_path, n_rois)

    if timeseries.shape[0] == n_rois and timeseries.shape[1] != n_rois:
        timeseries = timeseries.T
    if timeseries.shape[1] != n_rois:
        raise ValueError(f"Time series ROI count ({timeseries.shape[1]}) does not match FC matrix ({n_rois})")

    z_matrix = fisher_z(matrix)
    write_timeseries_tsv(timeseries_tsv, timeseries, labels)
    write_matrix_tsv(fc_correlation_tsv, matrix, labels)
    write_matrix_tsv(fc_fisherz_tsv, z_matrix, labels)
    n_edges = write_long_tsv(fc_long_tsv, matrix, z_matrix, labels, subject, session, task, atlas)
    n_missing = int(np.isnan(timeseries).any(axis=0).sum())
    n_constant = int(np.array([_is_constant(timeseries[:, idx]) for idx in range(timeseries.shape[1])]).sum())

    return StandardizedNetCorrOutputs(
        timeseries_tsv=timeseries_tsv,
        fc_correlation_tsv=fc_correlation_tsv,
        fc_fisherz_tsv=fc_fisherz_tsv,
        fc_long_tsv=fc_long_tsv,
        n_timepoints=int(timeseries.shape[0]),
        n_rois=n_rois,
        n_edges=n_edges,
        n_constant_rois=n_constant,
        n_missing_rois=n_missing,
    )


def parse_netts(path: Path) -> np.ndarray:
    rows = _numeric_rows(path)
    if not rows:
        raise ValueError(f"No numeric rows found in 3dNetCorr .netts file: {path}")
    return np.array(rows, dtype=float)


def parse_netcc(path: Path) -> np.ndarray:
    rows = _numeric_rows(path)
    if not rows:
        raise ValueError(f"No numeric rows found in 3dNetCorr .netcc file: {path}")
    square_rows = _last_square_rows(rows)
    matrix = np.array(square_rows, dtype=float)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"3dNetCorr .netcc matrix is not square: {path}")
    return matrix


def fisher_z(matrix: np.ndarray) -> np.ndarray:
    clipped = np.clip(matrix.astype(float), -0.999999, 0.999999)
    return np.arctanh(clipped)


def load_roi_labels(label_path: Path | None, n_rois: int) -> list[str]:
    labels: list[str] = []
    if label_path and label_path.exists():
        if label_path.suffix == ".tsv":
            labels = _labels_from_tsv(label_path)
        else:
            labels = _labels_from_lut(label_path)
    if len(labels) < n_rois:
        labels.extend(f"ROI_{idx:03d}" for idx in range(len(labels) + 1, n_rois + 1))
    return labels[:n_rois]


def write_timeseries_tsv(path: Path, timeseries: np.ndarray, labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["timepoint", *labels])
        for idx, row in enumerate(timeseries):
            writer.writerow([idx, *[_format_float(value) for value in row]])


def write_matrix_tsv(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["roi_id", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[_format_float(value) for value in row]])


def write_long_tsv(
    path: Path,
    matrix: np.ndarray,
    z_matrix: np.ndarray,
    labels: list[str],
    subject: str,
    session: str,
    task: str,
    atlas: str,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["subject", "session", "task", "atlas", "roi_i", "roi_j", "r", "z"],
            delimiter="\t",
        )
        writer.writeheader()
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                writer.writerow(
                    {
                        "subject": subject,
                        "session": session,
                        "task": task,
                        "atlas": atlas,
                        "roi_i": labels[i],
                        "roi_j": labels[j],
                        "r": _format_float(matrix[i, j]),
                        "z": _format_float(z_matrix[i, j]),
                    }
                )
                count += 1
    return count


def _numeric_rows(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        parts = line.replace(",", " ").split()
        try:
            rows.append([float(part) for part in parts])
        except ValueError:
            continue
    return rows


def _last_square_rows(rows: list[list[float]]) -> list[list[float]]:
    for size in range(len(rows), 0, -1):
        candidate = rows[-size:]
        if all(len(row) == size for row in candidate):
            return candidate
    raise ValueError("No square matrix block found")


def _labels_from_tsv(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            return []
        name_field = next((field for field in ("label", "name", "roi_name", "network") if field in reader.fieldnames), None)
        index_field = next((field for field in ("index", "id", "roi_id", "value") if field in reader.fieldnames), None)
        labels: list[str] = []
        for idx, row in enumerate(reader, start=1):
            value = row.get(name_field or "", "").strip() if name_field else ""
            labels.append(value or f"ROI_{idx:03d}")
        if index_field:
            return labels
        return labels


def _labels_from_lut(path: Path) -> list[str]:
    labels: list[str] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0].lstrip("-").isdigit():
            labels.append(parts[1])
    return labels


def _is_constant(values: np.ndarray) -> bool:
    clean = values[~np.isnan(values)]
    if clean.size == 0:
        return False
    return bool(np.nanmax(clean) == np.nanmin(clean))


def _format_float(value: float) -> str:
    if math.isnan(float(value)):
        return "nan"
    return f"{float(value):.6f}"
