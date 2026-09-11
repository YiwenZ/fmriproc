"""Atlas ROI summary for regional metric maps."""

from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np

from .atlas_registry import AtlasSpec
from .netcorr_outputs import load_roi_labels


def compute_roi_summary(
    map_path: Path,
    atlas: AtlasSpec,
    metric: str,
    subject: str,
    session: str,
    task: str,
    group_mask_path: Path | None = None,
    min_roi_coverage: float = 0.5,
    resolution: str = "auto",
) -> list[dict]:
    """Compute per-ROI statistics for a regional metric map.

    Returns a list of dicts, one per ROI, with keys: subject, session, task,
    atlas, metric, roi_id, roi_name, mean, sd, n_voxels_valid, n_voxels_atlas,
    coverage.
    """
    map_img = nib.load(str(map_path))
    map_data = np.asanyarray(map_img.dataobj).astype(np.float64)
    if map_data.ndim == 4 and map_data.shape[3] == 1:
        map_data = np.squeeze(map_data, axis=3)

    atlas_path = atlas.resolve_path(resolution)
    atlas_img = nib.load(str(atlas_path))
    atlas_data = np.asanyarray(atlas_img.dataobj).astype(np.int32)

    # Resample atlas if grid mismatch.
    if map_data.shape[:3] != atlas_data.shape[:3]:
        from .netcorr import _maybe_resample_nifti
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            resampled, _ = _maybe_resample_nifti(
                atlas.name, atlas_path, map_path, Path(td), dry_run=False
            )
            atlas_img = nib.load(str(resampled))
            atlas_data = np.asanyarray(atlas_img.dataobj).astype(np.int32)

    # Build ROI mask from atlas (exclude background 0).
    roi_ids = np.unique(atlas_data[atlas_data > 0])
    if roi_ids.size == 0:
        return []

    # Apply group mask.
    if group_mask_path is not None:
        mask_img = nib.load(str(group_mask_path))
        mask_data = np.asanyarray(mask_img.dataobj)
        if mask_data.shape[:3] != map_data.shape[:3]:
            from .netcorr import _maybe_resample_nifti
            import tempfile

            with tempfile.TemporaryDirectory() as td:
                resampled_mask, _ = _maybe_resample_nifti(
                    "group_mask", group_mask_path, map_path, Path(td), dry_run=False
                )
                mask_img = nib.load(str(resampled_mask))
                mask_data = np.asanyarray(mask_img.dataobj)
        valid_mask = (mask_data > 0) if mask_data.ndim == 3 else (mask_data > 0).any(axis=-1)
        if valid_mask.ndim == 3:
            valid_mask = valid_mask.astype(bool)
    else:
        valid_mask = np.ones(map_data.shape[:3], dtype=bool)

    # Apply finite check.
    finite_mask = np.isfinite(map_data)
    if finite_mask.ndim == 4:
        finite_mask = finite_mask.all(axis=-1)

    labels = load_roi_labels(atlas.label_path, int(roi_ids.max()))

    rows: list[dict] = []
    for roi_id in roi_ids:
        roi_bool = atlas_data == roi_id
        valid_voxels = roi_bool & valid_mask & finite_mask
        n_valid = int(np.sum(valid_voxels))
        n_atlas = int(np.sum(roi_bool))
        coverage = n_valid / n_atlas if n_atlas > 0 else 0.0

        if coverage < min_roi_coverage and n_atlas > 0:
            continue

        roi_name = labels[int(roi_id) - 1] if int(roi_id) <= len(labels) else f"ROI_{int(roi_id):03d}"

        if n_valid > 0:
            vals = map_data[valid_voxels]
            mean_val = float(np.mean(vals))
            sd_val = float(np.std(vals))
        else:
            mean_val = float("nan")
            sd_val = float("nan")

        rows.append({
            "subject": subject,
            "session": session,
            "task": task,
            "atlas": atlas.name,
            "metric": metric,
            "roi_id": f"ROI_{int(roi_id):03d}",
            "roi_name": roi_name,
            "mean": f"{mean_val:.6f}",
            "sd": f"{sd_val:.6f}",
            "n_voxels_valid": n_valid,
            "n_voxels_atlas": n_atlas,
            "coverage": f"{coverage:.6f}",
        })

    return rows


_ROI_FIELDNAMES = [
    "subject", "session", "task", "atlas", "metric", "roi_id", "roi_name",
    "mean", "sd", "n_voxels_valid", "n_voxels_atlas", "coverage",
]


def write_roi_summary_tsv(path: Path, rows: list[dict]) -> None:
    """Write per-ROI summary rows as a TSV file.

    Always writes the header row even when *rows* is empty, so downstream
    aggregation and inclusion checks see a predictable file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_ROI_FIELDNAMES, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
