from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np


def _load_img(path: Path) -> nib.Nifti1Image:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
        img = nib.Nifti1Image(data, img.affine, img.header)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D or 4D NIfTI: {path}")
    return img


def _load_3d(path: Path) -> np.ndarray:
    img = _load_img(path)
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    return data


def _load_overlay_img(path: Path) -> nib.Nifti1Image:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = np.mean(data, axis=3).astype("float32")
        img = nib.Nifti1Image(data, img.affine, img.header)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D or 4D NIfTI: {path}")
    return img


def _robust_limits(data: np.ndarray) -> tuple[float, float]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [2, 98])
    if float(lo) == float(hi):
        hi = lo + 1.0
    return float(lo), float(hi)


def _slice_indices(n_slices: int, depth: int) -> list[int]:
    if depth <= 0:
        raise ValueError("Image depth must be positive")
    n = max(1, min(int(n_slices), depth))
    return [int(v) for v in np.linspace(0, depth - 1, n + 2)[1:-1]]


def _to_axial_slice(data: np.ndarray, index: int) -> np.ndarray:
    return np.rot90(data[:, :, index])


def _edge_overlay(data: np.ndarray) -> np.ma.MaskedArray:
    """Return edge-like overlay data with flat background masked out."""
    gy, gx = np.gradient(data.astype("float32"))
    edge = np.sqrt(gx * gx + gy * gy)
    finite = edge[np.isfinite(edge)]
    positive = finite[finite > 0]
    if positive.size == 0:
        return np.ma.masked_array(edge, mask=np.ones(edge.shape, dtype=bool))
    threshold = float(np.max(positive) * 0.05)
    mask = edge <= threshold
    return np.ma.masked_array(edge, mask=mask)


def _foreground_mask(data: np.ndarray) -> np.ndarray:
    finite = data[np.isfinite(data)]
    positive = finite[finite > 0]
    if positive.size == 0:
        return np.zeros(data.shape, dtype=bool)
    threshold = float(np.percentile(positive, 5))
    return data > threshold


def _positive_mask(data: np.ndarray) -> np.ndarray:
    finite = data[np.isfinite(data)]
    positive = finite[finite > 0]
    if positive.size == 0:
        return np.zeros(data.shape, dtype=bool)
    threshold = float(np.percentile(positive, 1))
    return data > threshold


def render_fsl_slicer_overlay_png(
    background_path: Path,
    overlay_path: Path,
    output_path: Path,
) -> None:
    """Render a PhiPipe-style overlay using FSL slicer and pngappend."""
    if shutil.which("slicer") is None or shutil.which("pngappend") is None:
        raise RuntimeError("FSL slicer/pngappend not available for final MNI QC rendering")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # slicer only handles 3D inputs. Compute temporal mean if background is 4D.
    bg_path = background_path
    bg_img = nib.load(str(background_path))
    if bg_img.ndim >= 4:
        bg_data = np.asanyarray(bg_img.dataobj, dtype=np.float32)
        mean_data = bg_data.mean(axis=3, dtype="float32")
        # Clip negative values from spline interpolation overshoot at mask
        # boundaries so slicer auto-scaling does not wash out the brain.
        mean_data = np.clip(mean_data, 0, None)
        mean_img = nib.Nifti1Image(mean_data, bg_img.affine, bg_img.header)
        bg_path = output_path.parent / "tmp_bold_mni_mean.nii.gz"
        nib.save(mean_img, str(bg_path))
        bg_img = nib.load(str(bg_path))

    # Resample overlay to background grid if they differ (e.g. EPI ref vs T1).
    olay_path = overlay_path
    olay_img = nib.load(str(overlay_path))
    if olay_img.shape[:3] != bg_img.shape[:3] or not np.allclose(olay_img.affine, bg_img.affine):
        olay_resampled = resample_from_to(olay_img, bg_img, order=0)
        olay_path = output_path.parent / "tmp_overlay_resampled.nii.gz"
        nib.save(olay_resampled, str(olay_path))

    # Compute brain-only percentiles for display scaling.
    # Using the full min-max range (especially with masked outputs) washes
    # out brain contrast. Exclude near-zero voxels (spline ring, background)
    # and use p2–p98 of the remaining tissue-like voxels.
    bg_data = np.asanyarray(bg_img.dataobj, dtype=np.float32)
    tissue = bg_data[bg_data > (bg_data.max() * 0.01)]
    if tissue.size > 0:
        imin = str(int(np.percentile(tissue, 2)))
        imax = str(int(np.percentile(tissue, 98)))
    else:
        imin, imax = "0", "1"

    slice_specs = [
        ("-x", "0.35", "mytmp_s1.png"),
        ("-x", "0.45", "mytmp_s2.png"),
        ("-x", "0.55", "mytmp_s3.png"),
        ("-x", "0.65", "mytmp_s4.png"),
        ("-y", "0.35", "mytmp_c1.png"),
        ("-y", "0.45", "mytmp_c2.png"),
        ("-y", "0.55", "mytmp_c3.png"),
        ("-y", "0.65", "mytmp_c4.png"),
        ("-z", "0.35", "mytmp_a1.png"),
        ("-z", "0.45", "mytmp_a2.png"),
        ("-z", "0.55", "mytmp_a3.png"),
        ("-z", "0.65", "mytmp_a4.png"),
    ]

    try:
        with tempfile.TemporaryDirectory(prefix="fmri_qc_slicer_") as tmp:
            tmp_dir = Path(tmp)
            slicer_cmd = [
                "slicer",
                str(bg_path),
                str(olay_path),
                "-s",
                "2",
                "-u",
                "-i",
                imin,
                imax,
            ]
            for axis, fraction, filename in slice_specs:
                slicer_cmd.extend([axis, fraction, str(tmp_dir / filename)])
            subprocess.run(slicer_cmd, check=True)

            pngappend_cmd: list[str] = ["pngappend"]
            for idx, (_, _, filename) in enumerate(slice_specs):
                if idx > 0:
                    pngappend_cmd.append("+")
                pngappend_cmd.append(str(tmp_dir / filename))
            pngappend_cmd.append(str(output_path))
            subprocess.run(pngappend_cmd, check=True)
    finally:
        if bg_path != background_path and bg_path.exists():
            bg_path.unlink(missing_ok=True)
        if olay_path != overlay_path and olay_path.exists():
            olay_path.unlink(missing_ok=True)


def render_afni_chauffeur_overlay_png(
    background_path: Path,
    overlay_path: Path,
    output_path: Path,
) -> None:
    """Render final MNI QC with AFNI tools.

    Background is the temporal mean of MNI-space BOLD; overlay is an edge map
    from the positive MNI template voxels.
    """
    required = ("3dTstat", "3dcalc", "@chauffeur_afni")
    if any(shutil.which(tool) is None for tool in required):
        raise RuntimeError("AFNI 3dTstat/3dcalc/@chauffeur_afni not available for final MNI QC rendering")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fmri_qc_afni_") as tmp:
        tmp_dir = Path(tmp)
        mean_path = tmp_dir / "bold_mni_mean.nii.gz"
        edge_path = tmp_dir / "mni_template_edge.nii.gz"
        prefix = tmp_dir / "epi_to_mni_overlay"

        subprocess.run(["3dTstat", "-mean", "-prefix", str(mean_path), str(background_path)], check=True)
        subprocess.run(
            [
                "3dcalc",
                "-a",
                str(overlay_path),
                "-b",
                "a+i",
                "-c",
                "a-i",
                "-d",
                "a+j",
                "-e",
                "a-j",
                "-f",
                "a+k",
                "-g",
                "a-k",
                "-expr",
                "ispositive(a)*amongst(0,b,c,d,e,f,g)",
                "-prefix",
                str(edge_path),
            ],
            check=True,
        )
        subprocess.run(
            [
                "@chauffeur_afni",
                "-ulay",
                str(mean_path),
                "-olay",
                str(edge_path),
                "-prefix",
                str(prefix),
                "-set_xhairs",
                "OFF",
                "-montx",
                "4",
                "-monty",
                "3",
                "-opacity",
                "5",
                "-do_clean",
            ],
            check=True,
        )

        candidates = sorted(tmp_dir.glob("epi_to_mni_overlay*.png"))
        if not candidates:
            raise RuntimeError("AFNI @chauffeur_afni did not produce a PNG")
        candidates[0].replace(output_path)


def render_final_mni_overlay_png(
    background_path: Path,
    overlay_path: Path,
    output_path: Path,
    title: str,
    n_slices: int = 9,
    alpha: float = 0.35,
) -> None:
    """Render final MNI QC snapshot using the best available toolchain.

    Prefers FSL slicer, then AFNI @chauffeur_afni, then Python matplotlib.
    """
    try:
        render_fsl_slicer_overlay_png(background_path, overlay_path, output_path)
        return
    except RuntimeError:
        pass

    try:
        render_afni_chauffeur_overlay_png(background_path, overlay_path, output_path)
        return
    except RuntimeError:
        pass

    render_overlay_png(overlay_path, background_path, output_path, title, n_slices=n_slices, alpha=alpha)


def render_contour_overlay_png(
    background_path: Path,
    contour_path: Path,
    output_path: Path,
    title: str,
    n_slices: int = 9,
    contour_color: str = "yellow",
    contour_linewidth: float = 0.9,
) -> None:
    background_img = _load_overlay_img(background_path)
    contour_img = _load_img(contour_path)

    if background_img.shape != contour_img.shape or not np.allclose(background_img.affine, contour_img.affine):
        contour_img = resample_from_to(contour_img, background_img, order=0)

    background = np.asanyarray(background_img.dataobj, dtype=np.float32)
    contour_data = np.asanyarray(contour_img.dataobj, dtype=np.float32)
    contour_mask = _positive_mask(contour_data).astype("float32")

    vmin, vmax = _robust_limits(background)
    indices = _slice_indices(n_slices, background.shape[2])
    cols = 3
    rows = int(np.ceil(len(indices) / cols))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), squeeze=False)
    fig.suptitle(title)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, idx in zip(axes.ravel(), indices):
        background_slice = _to_axial_slice(background, idx)
        contour_slice = _to_axial_slice(contour_mask, idx)
        ax.imshow(background_slice, cmap="gray", vmin=vmin, vmax=vmax)
        if np.any(contour_slice > 0):
            ax.contour(contour_slice, levels=[0.5], colors=[contour_color], linewidths=contour_linewidth)
        ax.set_title(f"z={idx}", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def render_slice_montage_png(input_path: Path, output_path: Path, title: str, n_slices: int = 9) -> None:
    data = _load_3d(input_path)
    vmin, vmax = _robust_limits(data)
    indices = _slice_indices(n_slices, data.shape[2])
    cols = 3
    rows = int(np.ceil(len(indices) / cols))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), squeeze=False)
    fig.suptitle(title)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, idx in zip(axes.ravel(), indices):
        ax.imshow(_to_axial_slice(data, idx), cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"z={idx}", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def render_overlay_png(
    underlay_path: Path,
    overlay_path: Path,
    output_path: Path,
    title: str,
    n_slices: int = 9,
    alpha: float = 0.35,
    overlay_mask_path: Path | None = None,
) -> None:
    underlay_img = _load_img(underlay_path)
    overlay_img = _load_overlay_img(overlay_path)

    # Apply brain mask to overlay in its native space (before resampling)
    # so non-brain regions do not contribute edges.
    if overlay_mask_path is not None and overlay_mask_path.exists():
        mask_img = _load_img(overlay_mask_path)
        overlay_arr = np.asanyarray(overlay_img.dataobj, dtype=np.float32)
        mask_arr = np.asanyarray(mask_img.dataobj, dtype=np.float32)
        overlay_arr = overlay_arr * (mask_arr > 0.5)
        overlay_img = nib.Nifti1Image(overlay_arr.astype(np.float32), overlay_img.affine, overlay_img.header)

    if underlay_img.shape != overlay_img.shape or not np.allclose(underlay_img.affine, overlay_img.affine):
        overlay_img = resample_from_to(overlay_img, underlay_img, order=1)
    underlay = np.asanyarray(underlay_img.dataobj, dtype=np.float32)
    overlay = np.asanyarray(overlay_img.dataobj, dtype=np.float32)

    umin, umax = _robust_limits(underlay)
    indices = _slice_indices(n_slices, underlay.shape[2])
    cols = 3
    rows = int(np.ceil(len(indices) / cols))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), squeeze=False)
    fig.suptitle(title)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, idx in zip(axes.ravel(), indices):
        underlay_slice = _to_axial_slice(underlay, idx)
        ax.imshow(underlay_slice, cmap="gray", vmin=umin, vmax=umax)
        overlay_edges = _edge_overlay(_to_axial_slice(overlay, idx))
        overlay_edges.mask = np.logical_or(overlay_edges.mask, ~_foreground_mask(underlay_slice))
        if np.ma.count(overlay_edges) > 0:
            edge_values = overlay_edges.compressed()
            vmin, vmax = np.percentile(edge_values, [5, 99])
            if float(vmin) == float(vmax):
                vmax = vmin + 1.0
            ax.imshow(overlay_edges, cmap="hot", vmin=vmin, vmax=vmax, alpha=alpha)
        ax.set_title(f"z={idx}", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
