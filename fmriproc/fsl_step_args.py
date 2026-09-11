from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .models import SelectedInputs, StepDefinition
from .paths import PrepPaths


def resolve_fsl_step_args(
    step: StepDefinition,
    paths: PrepPaths,
    inputs: SelectedInputs,
    config: dict[str, Any],
    template_path: str = "",
) -> list[str]:
    """Return positional arguments for FSL bash functions."""
    mode = config.get("runtime", {}).get("mode", "tested")
    bold_config = config.get("bold", {})
    t1_input = str(inputs.t1.path)
    bold_input = str(inputs.bold.path)

    # ── ANAT steps ────────────────────────────────────────────────────────

    if step.step_id == "t1_s01_reorient2std":
        return [t1_input, str(paths.anat_output(step.primary_output)), mode]

    elif step.step_id == "t1_s02_skullstrip_bet":
        return [
            str(paths.anat_output("t1_s01_reorient2std.nii.gz")),
            str(paths.anat_output(step.primary_output)),
            mode,
        ]

    elif step.step_id == "t1_s03_fast_segmentation":
        return [
            str(paths.anat_output("t1_s02_skullstrip_bet.nii.gz")),
            str(paths.anat_dir / "seg"),
            mode,
            str(paths.anat_output(step.primary_output)),
        ]

    elif step.step_id == "t1_s04_masks":
        return [
            str(paths.anat_output("t1_s02_skullstrip_bet.nii.gz")),
            str(paths.anat_dir / "seg"),
            str(paths.masks_dir),
            mode,
            str(paths.anat_output(step.primary_output)),
        ]

    elif step.step_id == "t1_s05_space_mni152_affine":
        return [
            str(paths.anat_output("t1_s02_skullstrip_bet.nii.gz")),
            template_path,
            str(paths.anat_output(step.primary_output)),
            str(paths.reg_output("fsl_anat_to_mni_affine.mat")),
            mode,
        ]

    elif step.step_id == "t1_s06_space_mni152_nonlinear":
        return [
            str(paths.anat_output("t1_s01_reorient2std.nii.gz")),
            str(paths.reg_output("fsl_anat_to_mni_affine.mat")),
            str(paths.anat_output(step.primary_output)),
            str(paths.reg_output("fsl_anat_to_mni_warp.nii.gz")),
            str(paths.reg_output("fsl_anat_to_mni_coef.nii.gz")),
            mode,
        ]

    # ── BOLD steps ────────────────────────────────────────────────────────

    elif step.step_id == "bold_s01_reorient2std":
        return [bold_input, str(paths.bold_output(step.primary_output)), mode]

    elif step.step_id == "bold_s02_drop_dummy":
        dummy = bold_config.get("dummy_scans", 10)
        vol_count = inputs.bold.volume_count or dummy + 1
        length_after = str(max(vol_count - dummy, 1))
        return [
            str(paths.bold_output("bold_s01_reorient2std.nii.gz")),
            str(paths.bold_output(step.primary_output)),
            str(dummy),
            length_after,
            mode,
        ]

    elif step.step_id == "bold_s03_ref":
        return [
            str(paths.bold_output("bold_s02_drop_dummy.nii.gz")),
            str(paths.bold_output(step.primary_output)),
            mode,
        ]

    elif step.step_id == "bold_s04_motion_correction_mcflirt":
        return [
            str(paths.bold_output("bold_s02_drop_dummy.nii.gz")),
            str(paths.bold_output(step.primary_output)),
            str(paths.bold_output("bold_s03_ref.nii.gz")),
            str(paths.qc_output("motion_parameters.par")),
            mode,
        ]

    elif step.step_id == "bold_s05_slice_timing":
        tr = str(bold_config.get("tr", "auto"))
        st_cfg = _get_slice_timing_config(config)
        direction = str(st_cfg.get("fsl_direction", "3"))
        interleaved = str(st_cfg.get("fsl_interleaved", "odd"))
        return [
            str(paths.bold_output("bold_s04_motion_correction_mcflirt.nii.gz")),
            str(paths.bold_output(step.primary_output)),
            tr,
            direction,
            interleaved,
            mode,
        ]

    elif step.step_id == "bold_s06_epi_to_anat_bbr":
        out_prefix = str(paths.reg_dir / "fsl_epi_to_anat_bbr")
        return [
            str(paths.bold_output("bold_s03_ref.nii.gz")),
            str(paths.anat_output("t1_s01_reorient2std.nii.gz")),
            str(paths.anat_output("t1_s02_skullstrip_bet.nii.gz")),
            out_prefix,
            mode,
            str(paths.bold_output(step.primary_output)),
        ]

    elif step.step_id == "bold_s07_masks_to_epi":
        return [
            str(paths.masks_dir),
            str(paths.reg_output("fsl_epi_to_anat_bbr.mat")),
            str(paths.bold_output("bold_s03_ref.nii.gz")),
            str(paths.masks_dir),
            mode,
            str(paths.bold_output(step.primary_output)),
        ]

    elif step.step_id == "bold_s08_nuisance_regression":
        tr = str(bold_config.get("tr", "1.0"))
        highpass = str(bold_config.get("highpass", 0.01))
        lowpass_raw = bold_config.get("lowpass")
        lowpass = str(lowpass_raw) if lowpass_raw is not None else "99999"
        post_motion = _post_motion_fsl(paths, config)
        return [
            str(post_motion),
            str(paths.bold_output(step.primary_output)),
            str(paths.masks_dir / "t1_brain_mask_to_epi.nii.gz"),
            str(paths.masks_dir / "t1_csf_mask_to_epi.nii.gz"),
            str(paths.masks_dir / "t1_wm_mask_to_epi.nii.gz"),
            str(paths.qc_output("nuisance_regressors.txt")),
            str(paths.qc_output("nuisance_regressors.mat")),
            tr,
            highpass,
            lowpass,
            mode,
        ]

    elif step.step_id == "bold_s09_temporal_filtering":
        tr = float(bold_config.get("tr", 2.0))
        highpass = float(bold_config.get("highpass", 0.01))
        lowpass_raw = bold_config.get("lowpass")
        # FSL bptf sigma = 1 / (2 * TR * cutoff).
        # A negative lowpass sigma disables lowpass filtering in fslmaths.
        hp_sigma = str(max(1, round(1.0 / (2.0 * tr * highpass))))
        if lowpass_raw is not None:
            lp_sigma = str(max(1, round(1.0 / (2.0 * tr * float(lowpass_raw)))))
        else:
            lp_sigma = "-1"
        post_motion = _post_motion_fsl(paths, config)
        return [
            str(paths.bold_output("bold_s08_nuisance_regression.nii.gz")),
            str(paths.bold_output(step.primary_output)),
            hp_sigma,
            lp_sigma,
            mode,
        ]

    elif step.step_id == "bold_s10_grand_mean_scaling_10000":
        target = str(bold_config.get("grand_mean", 10000))
        return [
            str(paths.bold_output("bold_s09_temporal_filtering.nii.gz")),
            str(paths.masks_dir / "t1_brain_mask_to_epi.nii.gz"),
            str(paths.bold_output(step.primary_output)),
            target,
            mode,
        ]

    elif step.step_id == "bold_s11_space_mni152":
        tr = str(bold_config.get("tr", "2.0"))
        return [
            str(paths.bold_output("bold_s10_grand_mean_scaling_10000.nii.gz")),
            template_path,
            str(paths.reg_output("fsl_epi_to_anat_bbr.mat")),
            str(paths.reg_output("fsl_anat_to_mni_warp.nii.gz")),
            str(paths.masks_dir / "t1_brain_mask_to_epi.nii.gz"),
            str(paths.bold_output(step.primary_output)),
            tr,
            mode,
        ]

    elif step.step_id == "bold_s12_smooth_sm6":
        fwhm = float(bold_config.get("smooth_fwhm", 6))
        sigma = str(round(fwhm / 2.3548, 4))
        return [
            str(paths.bold_output("bold_s11_space_mni152.nii.gz")),
            str(paths.bold_output(step.primary_output)),
            sigma,
            mode,
        ]

    raise ValueError(f"No FSL argument resolver for step: {step.step_id}")


def _get_slice_timing_config(config: dict[str, Any]) -> dict[str, Any]:
    """Read slice timing config from nested bold.slice_timing or flat bold keys."""
    value = config.get("bold", {}).get("slice_timing", False)
    if isinstance(value, dict):
        return value
    return {}


def _slice_timing_enabled_fsl(config: dict[str, Any]) -> bool:
    value = config.get("bold", {}).get("slice_timing", False)
    if isinstance(value, dict):
        return bool(value.get("enabled", False))
    return bool(value)


def _post_motion_fsl(paths: PrepPaths, config: dict[str, Any]) -> Path:
    if _slice_timing_enabled_fsl(config):
        return paths.bold_output("bold_s05_slice_timing.nii.gz")
    return paths.bold_output("bold_s04_motion_correction_mcflirt.nii.gz")
