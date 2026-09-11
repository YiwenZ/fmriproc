from __future__ import annotations

import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .atlas_registry import AtlasSpec
from .paths import MetricPaths


@dataclass(frozen=True)
class NetCorrResult:
    atlas: AtlasSpec
    bold_path: Path
    atlas_path: Path
    roi_path: Path
    prefix: Path
    command: list[str]
    netcc_path: Path
    netts_path: Path
    log_path: Path
    resampled_atlas_path: Path | None = None
    group_mask_path: Path | None = None
    resampled_group_mask_path: Path | None = None
    constrained_roi_path: Path | None = None
    roi_labels_before_group_mask: int | None = None
    roi_labels_after_group_mask: int | None = None


def build_3dnetcorr_command(
    bold_path: Path,
    roi_path: Path,
    prefix: Path,
    push_thru_many_zeros: bool = True,
) -> list[str]:
    command = [
        "3dNetCorr",
    ]
    if push_thru_many_zeros:
        command.append("-push_thru_many_zeros")
    command.extend(
        [
            "-inset",
            str(bold_path),
            "-in_rois",
            str(roi_path),
            "-ts_out",
            "-prefix",
            str(prefix),
        ]
    )
    return command


def run_3dnetcorr(
    bold_path: Path,
    atlas: AtlasSpec,
    metric_paths: MetricPaths,
    cfg: dict[str, Any],
    dry_run: bool = False,
    force: bool = False,
) -> NetCorrResult:
    metric_paths.ensure_dirs()
    raw_dir = metric_paths.raw_tool_dir("afni_3dNetCorr")
    metrics_cfg = cfg.get("metrics", {})
    resolution = metrics_cfg.get("atlases", {}).get("resolution", "auto")
    atlas_path = atlas.resolve_path(resolution)
    roi_path, resampled = _maybe_resample_nifti(atlas.name, atlas_path, bold_path, raw_dir, dry_run)
    group_mask_path = _resolve_group_mask(metric_paths, cfg)
    resampled_group_mask_path: Path | None = None
    constrained_roi_path: Path | None = None
    roi_labels_before_group_mask: int | None = None
    roi_labels_after_group_mask: int | None = None
    if group_mask_path is not None:
        roi_labels_before_group_mask = _count_positive_labels(roi_path) if not dry_run else None
        group_mask_roi, resampled_group_mask_path = _maybe_resample_nifti(
            group_mask_path.stem.replace(".nii", ""),
            group_mask_path,
            bold_path,
            raw_dir,
            dry_run,
        )
        constrained_roi_path = raw_dir / f"{atlas.name}_constrained_by_{group_mask_path.name.removesuffix('.nii.gz')}.nii.gz"
        if dry_run or (constrained_roi_path.exists() and not force):
            roi_path = constrained_roi_path
        else:
            roi_path = _apply_group_mask_to_roi(roi_path, group_mask_roi, constrained_roi_path)
        roi_labels_after_group_mask = _count_positive_labels(roi_path) if not dry_run else None
    prefix = raw_dir / metric_paths.stem(atlas.name)
    netcorr_cfg = metrics_cfg.get("netcorr", {})
    command = build_3dnetcorr_command(
        bold_path,
        roi_path,
        prefix,
        push_thru_many_zeros=netcorr_cfg.get("push_thru_many_zeros", True),
    )
    log_path = prefix.with_suffix(".log")
    result = NetCorrResult(
        atlas=atlas,
        bold_path=bold_path,
        atlas_path=atlas_path,
        roi_path=roi_path,
        prefix=prefix,
        command=command,
        netcc_path=_netcorr_output(prefix, ".netcc"),
        netts_path=_netcorr_output(prefix, ".netts"),
        log_path=log_path,
        resampled_atlas_path=resampled,
        group_mask_path=group_mask_path,
        resampled_group_mask_path=resampled_group_mask_path,
        constrained_roi_path=constrained_roi_path,
        roi_labels_before_group_mask=roi_labels_before_group_mask,
        roi_labels_after_group_mask=roi_labels_after_group_mask,
    )

    if dry_run:
        _write_log(log_path, f"DRY RUN: {_quote(command)}\n")
        return result

    if result.netcc_path.exists() and result.netts_path.exists() and not force:
        _write_log(log_path, f"SKIP existing outputs: {_quote(command)}\n")
        return result

    if shutil.which("3dNetCorr") is None:
        raise RuntimeError("AFNI 3dNetCorr is not available on PATH.")

    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    _write_log(
        log_path,
        "\n".join(
            [
                f"COMMAND: {_quote(command)}",
                f"RETURN_CODE: {completed.returncode}",
                "STDOUT:",
                completed.stdout,
                "STDERR:",
                completed.stderr,
            ]
        ),
    )
    if completed.returncode != 0:
        raise RuntimeError(f"3dNetCorr failed for atlas {atlas.name}; see {log_path}")
    return result


def _maybe_resample_nifti(
    name: str,
    input_path: Path,
    bold_path: Path,
    raw_dir: Path,
    dry_run: bool,
) -> tuple[Path, Path | None]:
    if not _nifti_grid_mismatch(input_path, bold_path):
        return input_path, None
    resampled = raw_dir / f"{name}_resampled_to_bold.nii.gz"
    if dry_run or resampled.exists():
        return resampled, resampled
    if shutil.which("3dresample") is None:
        return input_path, None
    command = ["3dresample", "-master", str(bold_path), "-input", str(input_path), "-prefix", str(resampled), "-rmode", "NN"]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"3dresample failed while aligning {name}: {completed.stderr}")
    return resampled, resampled


def _resolve_group_mask(metric_paths: MetricPaths, cfg: dict[str, Any]) -> Path | None:
    group_cfg = cfg.get("metrics", {}).get("group_mask", {})
    if not group_cfg.get("enabled", True) or not group_cfg.get("apply", True):
        return None
    filename = group_cfg.get("default_constraint", "grp_mask_thr70.nii.gz")
    path = metric_paths.prep_paths.prep_dir / "group_mask" / filename
    if not path.exists():
        raise FileNotFoundError(f"Group mask not found: {path}. Run 'fmriproc group-mask --prep-dir {metric_paths.prep_paths.prep_dir}' first.")
    return path


def _apply_group_mask_to_roi(roi_path: Path, group_mask_path: Path, output_path: Path) -> Path:
    if shutil.which("3dcalc") is not None:
        command = [
            "3dcalc",
            "-a",
            str(roi_path),
            "-b",
            str(group_mask_path),
            "-expr",
            "a*step(b)",
            "-prefix",
            str(output_path),
            "-overwrite",
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"3dcalc failed while applying group mask: {completed.stderr}")
        return output_path

    import nibabel as nib
    import numpy as np

    roi_img = nib.load(str(roi_path))
    mask_img = nib.load(str(group_mask_path))
    roi_data = np.asanyarray(roi_img.dataobj)
    mask_data = np.asanyarray(mask_img.dataobj)
    if roi_data.shape[:3] != mask_data.shape[:3]:
        raise RuntimeError("Cannot apply group mask with Python fallback because atlas and mask grids differ.")
    out = nib.Nifti1Image((roi_data * (mask_data > 0)).astype(roi_data.dtype), roi_img.affine, roi_img.header)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(output_path))
    return output_path


def _count_positive_labels(path: Path) -> int:
    try:
        import nibabel as nib
        import numpy as np

        data = np.asanyarray(nib.load(str(path)).dataobj)
        labels = np.unique(data[data > 0])
        return int(labels.size)
    except Exception:
        return 0


def _nifti_grid_mismatch(atlas_path: Path, bold_path: Path) -> bool:
    try:
        import nibabel as nib

        atlas_img = nib.load(str(atlas_path))
        bold_img = nib.load(str(bold_path))
        atlas_shape = atlas_img.shape[:3]
        bold_shape = bold_img.shape[:3]
        atlas_zooms = tuple(round(float(v), 3) for v in atlas_img.header.get_zooms()[:3])
        bold_zooms = tuple(round(float(v), 3) for v in bold_img.header.get_zooms()[:3])
        return atlas_shape != bold_shape or atlas_zooms != bold_zooms
    except Exception:
        return False


def _write_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _quote(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _netcorr_output(prefix: Path, suffix: str) -> Path:
    return prefix.parent / f"{prefix.name}_000{suffix}"
