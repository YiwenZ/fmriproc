"""Within-mask z-scoring for regional metric maps."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def zscore_within_mask(map_path: Path, mask_path: Path, output_path: Path) -> Path:
    """Z-score *map_path* voxels within *mask_path*, write to *output_path*.

    Voxels outside the mask and non-finite values are set to zero in the
    output z-map.

    Returns *output_path*.
    """
    map_img = nib.load(str(map_path))
    mask_img = nib.load(str(mask_path))
    map_data = np.asanyarray(map_img.dataobj).astype(np.float64)
    mask_data = np.asanyarray(mask_img.dataobj)

    # Broadcast mask shape to match map data (handles 3D mask on 4D data).
    if mask_data.ndim == 3 and map_data.ndim == 4:
        mask_data = mask_data[..., np.newaxis]
    mask_bool = (mask_data > 0) & np.isfinite(map_data)

    z_data = np.zeros_like(map_data)
    if not np.any(mask_bool):
        _save_nifti(z_data, map_img.affine, map_img.header, output_path)
        return output_path

    within = map_data[mask_bool]
    mean = float(np.mean(within))
    sd = float(np.std(within))

    if sd == 0:
        raise RuntimeError(
            f"Within-mask standard deviation is zero for {map_path}. "
            f"Cannot compute z-scores."
        )

    z_data[mask_bool] = (within - mean) / sd
    _save_nifti(z_data, map_img.affine, map_img.header, output_path)
    return output_path


def _save_nifti(
    data: np.ndarray,
    affine: np.ndarray,
    header: nib.Nifti1Header,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(data, affine, header)
    nib.save(out_img, str(output_path))
