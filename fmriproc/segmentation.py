from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


VALID_SEGMENTATION_BACKENDS = {"samseg_if_available", "samseg", "native", "afni", "fsl"}

# SAMSEG output follows FreeSurferColorLUT labels. Keep this as a conservative
# intracranial whitelist; do not use data > 0 because SAMSEG also labels skull,
# soft non-brain tissue, and fluid inside eyes.
WM_LABELS = {2, 7, 10, 13, 16, 28, 41, 46, 49, 52, 60, 77, 85}
CSF_LABELS = {4, 5, 14, 15, 24, 30, 31, 43, 44, 62, 63, 72}
GM_LABELS = {3, 8, 11, 12, 17, 18, 26, 42, 47, 50, 51, 53, 54, 58, 80}
BRAIN_LABELS = WM_LABELS | CSF_LABELS | GM_LABELS


def requested_segmentation_backend(cfg: dict[str, Any]) -> str:
    return str(cfg.get("anat", {}).get("segmentation_backend", "samseg_if_available"))


def resolve_segmentation_backend(
    cfg: dict[str, Any],
    runtime_backend: str,
    run_samseg_path: str | None,
) -> str:
    requested = requested_segmentation_backend(cfg)
    if requested == "samseg_if_available":
        return "samseg" if run_samseg_path else runtime_backend
    if requested == "samseg":
        if not run_samseg_path:
            raise RuntimeError("anat.segmentation_backend='samseg' requires run_samseg on PATH")
        return "samseg"
    if requested == "native":
        return runtime_backend
    if requested in {"afni", "fsl"}:
        if requested != runtime_backend:
            raise RuntimeError(
                f"anat.segmentation_backend='{requested}' requires runtime.backend='{requested}'"
            )
        return requested
    raise ValueError(
        "Invalid anat.segmentation_backend: "
        f"{requested}. Expected one of: {', '.join(sorted(VALID_SEGMENTATION_BACKENDS))}"
    )


def build_run_samseg_command(t1_path: Path, output_dir: Path, threads: int = 1) -> list[str]:
    safe_threads = max(int(threads), 1)
    return [
        "run_samseg",
        "--input",
        str(t1_path),
        "--output",
        str(output_dir),
        "--threads",
        str(safe_threads),
    ]


def write_segmentation_provenance(
    path: Path,
    requested: str,
    used: str,
    source_t1: Path,
    samseg_dir: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "segmentation_backend_requested": requested,
        "segmentation_backend_used": used,
        "source_t1": str(source_t1),
        "samseg_dir": str(samseg_dir) if samseg_dir is not None else "",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def derive_masks_from_samseg(seg_path: Path, mask_dir: Path) -> dict[str, Path]:
    mask_dir.mkdir(parents=True, exist_ok=True)
    img = nib.load(str(seg_path))
    data = np.asanyarray(img.dataobj)

    brain = np.isin(data, sorted(BRAIN_LABELS))
    wm = np.isin(data, sorted(WM_LABELS))
    csf = np.isin(data, sorted(CSF_LABELS))
    gm = np.isin(data, sorted(GM_LABELS))

    outputs = {
        "brain": mask_dir / "t1_brain_mask.nii.gz",
        "wm": mask_dir / "t1_wm_mask.nii.gz",
        "csf": mask_dir / "t1_csf_mask.nii.gz",
        "gm": mask_dir / "t1_gm_mask.nii.gz",
    }
    for key, mask in (("brain", brain), ("wm", wm), ("csf", csf), ("gm", gm)):
        out_img = nib.Nifti1Image(mask.astype(np.uint8), img.affine)
        nib.save(out_img, outputs[key])
    return outputs
