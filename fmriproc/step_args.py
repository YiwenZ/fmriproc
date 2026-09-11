from __future__ import annotations

from pathlib import Path
from typing import Any

from .bids_scan import read_bids_json
from .models import SelectedInputs, StepDefinition
from .paths import PrepPaths


def resolve_step_args(
    step: StepDefinition,
    paths: PrepPaths,
    inputs: SelectedInputs,
    config: dict[str, Any],
    template_path: str = "",
    reg_backend: str = "afni",
) -> list[str]:
    """Return positional arguments for the Bash function implementing `step`.

    `reg_backend` is "afni" or "ants" and changes the argument lists for
    registration steps (t1_s04, t1_s05, bold_s06, bold_s10).
    """
    mode = config.get("runtime", {}).get("mode", "tested")
    bold_config = config.get("bold", {})
    t1_input = str(inputs.t1.path)
    bold_input = str(inputs.bold.path)

    # ── Common (non-registration) steps ──────────────────────────────────

    if step.step_id == "t1_s01_deoblique":
        return [t1_input, str(paths.anat_output(step.primary_output)), mode]

    elif step.step_id == "t1_s02_skullstrip":
        return [str(paths.anat_output("t1_s01_deoblique.nii.gz")), str(paths.anat_output(step.primary_output)), mode]

    elif step.step_id == "t1_s03_unifize":
        return [str(paths.anat_output("t1_s02_skullstrip.nii.gz")), str(paths.anat_output(step.primary_output)), mode]

    elif step.step_id == "t1_s06_segmentation":
        return [
            str(paths.anat_output("t1_s03_unifize.nii.gz")),
            str(paths.anat_dir / "Segsy"),
            mode,
            str(paths.anat_output("t1_s06_segmentation.done")),
        ]

    elif step.step_id == "t1_s07_masks":
        return [
            str(paths.anat_output("t1_s03_unifize.nii.gz")),
            str(paths.anat_dir / "Segsy" / "Classes+orig"),
            str(paths.masks_dir),
            mode,
            str(paths.anat_output("t1_s07_masks.done")),
        ]

    elif step.step_id == "bold_s01_drop_dummy":
        dummy = str(bold_config.get("dummy_scans", 10))
        return [bold_input, str(paths.bold_output(step.primary_output)), dummy, mode]

    elif step.step_id == "bold_s02_deoblique":
        return [str(paths.bold_output("bold_s01_drop_dummy.nii.gz")), str(paths.bold_output(step.primary_output)), mode]

    elif step.step_id == "bold_s03_motion_correction":
        return [
            str(paths.bold_output("bold_s02_deoblique.nii.gz")),
            str(paths.bold_output(step.primary_output)),
            str(paths.qc_output("motion_parameters.1D")),
            str(paths.qc_output("motion_mat.1D")),
            mode,
        ]

    elif step.step_id == "bold_s04_slice_timing":
        tr = str(bold_config.get("tr", "auto"))
        pattern = _resolve_slice_timing_pattern(paths, inputs, bold_config)
        return [
            str(paths.bold_output("bold_s03_motion_correction.nii.gz")),
            str(paths.bold_output(step.primary_output)),
            tr,
            pattern,
            mode,
        ]

    elif step.step_id == "bold_s05_epi_ref_unifize":
        bucket = str(paths.work_dir / step.step_id / "epi_base_0.nii.gz")
        return [
            str(_post_motion_bold_input(paths, config)),
            bucket,
            str(paths.bold_output(step.primary_output)),
            mode,
        ]

    elif step.step_id == "bold_s08_nuisance_regression":
        tr = str(bold_config.get("tr", "1.0"))
        highpass = str(bold_config.get("highpass", 0.01))
        lowpass_raw = bold_config.get("lowpass")
        lowpass = str(lowpass_raw) if lowpass_raw is not None else "99999"
        csf_mask = str(paths.masks_dir / "t1_csf_mask_to_epi.nii.gz")
        wm_mask = str(paths.masks_dir / "t1_wm_mask_to_epi.nii.gz")
        nuisance_1d = str(paths.qc_output("nuisance_regressors.1D"))
        mask = str(paths.masks_dir / "t1_brain_mask_to_epi.nii.gz")
        return [
            str(_post_motion_bold_input(paths, config)),
            str(paths.bold_output(step.primary_output)),
            mask,
            csf_mask,
            wm_mask,
            nuisance_1d,
            tr,
            highpass,
            lowpass,
            mode,
        ]

    elif step.step_id == "bold_s09_grand_mean_scaling_10000":
        target = str(bold_config.get("grand_mean", 10000))
        mask = str(paths.masks_dir / "t1_brain_mask_to_epi.nii.gz")
        return [
            str(paths.bold_output("bold_s08_nuisance_regression.nii.gz")),
            mask,
            str(paths.bold_output(step.primary_output)),
            target,
            mode,
        ]

    elif step.step_id == "bold_s11_smooth_sm6":
        fwhm = str(bold_config.get("smooth_fwhm", 6))
        return [
            str(paths.bold_output("bold_s10_space_mni152.nii.gz")),
            str(paths.bold_output(step.primary_output)),
            fwhm,
            mode,
        ]

    # ── Registration steps (ANTs vs AFNI) ────────────────────────────────

    if reg_backend == "ants":
        return _ants_reg_args(step, paths, template_path, mode)
    else:
        return _afni_reg_args(step, paths, template_path, mode)


def _ants_reg_args(step: StepDefinition, paths: PrepPaths, template_path: str, mode: str) -> list[str]:
    """ANTs-style registration arguments."""
    if step.step_id == "t1_s04_space_mni152_affine":
        # ants_t1_to_mni_affine: moving fixed output matrix prefix mode
        return [
            str(paths.anat_output("t1_s03_unifize.nii.gz")),
            template_path,
            str(paths.anat_output(step.primary_output)),
            str(paths.reg_output("ants_t1_to_mni_0GenericAffine.mat")),
            str(paths.reg_dir) + "/ants_t1_to_mni_affine_",
            mode,
        ]

    elif step.step_id == "t1_s05_space_mni152_nonlinear":
        # ants_t1_to_mni_nonlinear: moving fixed output matrix warp prefix mode
        return [
            str(paths.anat_output("t1_s03_unifize.nii.gz")),
            template_path,
            str(paths.anat_output(step.primary_output)),
            str(paths.reg_output("ants_t1_to_mni_0GenericAffine.mat")),
            str(paths.reg_output("ants_t1_to_mni_1Warp.nii.gz")),
            str(paths.reg_dir) + "/ants_t1_to_mni_nonlinear_",
            mode,
        ]

    elif step.step_id == "bold_s06_epi_to_anat":
        # ants_epi_to_t1_affine: epi t1 output matrix prefix mode
        return [
            str(paths.bold_output("bold_s05_epi_ref_unifize.nii.gz")),
            str(paths.anat_output("t1_s03_unifize.nii.gz")),
            str(paths.bold_output("bold_s06_epi_to_anat.nii.gz")),
            str(paths.reg_output("ants_epi_to_t1_0GenericAffine.mat")),
            str(paths.reg_dir) + "/ants_epi_to_t1_affine_",
            mode,
            str(paths.bold_output("bold_s06_epi_to_anat.done")),
        ]

    elif step.step_id == "bold_s07_masks_to_epi":
        # ANTs mask-to-EPI: use 3D EPI ref (bold_s05), not 4D time series (bold_s03).
        epi_to_t1_mat = str(paths.reg_output("ants_epi_to_t1_0GenericAffine.mat"))
        return [
            str(paths.masks_dir),
            epi_to_t1_mat,
            str(paths.bold_output("bold_s05_epi_ref_unifize.nii.gz")),
            str(paths.masks_dir),
            mode,
            str(paths.bold_output("bold_s07_masks_to_epi.done")),
        ]

    elif step.step_id == "bold_s10_space_mni152":
        # ants_apply_epi_to_mni: input template output epi2t1_mat t12mni_warp t12mni_mat epi_mask mode
        return [
            str(paths.bold_output("bold_s09_grand_mean_scaling_10000.nii.gz")),
            template_path,
            str(paths.bold_output(step.primary_output)),
            str(paths.reg_output("ants_epi_to_t1_0GenericAffine.mat")),
            str(paths.reg_output("ants_t1_to_mni_1Warp.nii.gz")),
            str(paths.reg_output("ants_t1_to_mni_0GenericAffine.mat")),
            str(paths.masks_dir / "t1_brain_mask_to_epi.nii.gz"),
            mode,
        ]

    raise ValueError(f"No ANTs argument resolver for step: {step.step_id}")


def _afni_reg_args(step: StepDefinition, paths: PrepPaths, template_path: str, mode: str) -> list[str]:
    """AFNI-style registration arguments (original)."""
    if step.step_id == "t1_s04_space_mni152_affine":
        return [
            str(paths.anat_output("t1_s03_unifize.nii.gz")),
            template_path,
            str(paths.anat_output(step.primary_output)),
            str(paths.reg_output("anat_s01_to_mni152_affine.1D")),
            mode,
        ]

    elif step.step_id == "t1_s05_space_mni152_nonlinear":
        return [
            str(paths.anat_output("t1_s04_space_mni152_affine.nii.gz")),
            template_path,
            str(paths.anat_output(step.primary_output)),
            mode,
            str(paths.reg_output("anat_s02_to_mni152_warp.nii.gz")),
        ]

    elif step.step_id == "bold_s06_epi_to_anat":
        return [
            str(paths.bold_output("bold_s05_epi_ref_unifize.nii.gz")),
            str(paths.anat_output("t1_s03_unifize.nii.gz")),
            str(paths.reg_dir),
            mode,
            str(paths.bold_output("bold_s06_epi_to_anat.done")),
        ]

    elif step.step_id == "bold_s07_masks_to_epi":
        epi_to_anat_mat = str(paths.reg_dir / "bold_s05_epi_ref_unifize_align_mat.aff12.1D")
        return [
            str(paths.masks_dir),
            epi_to_anat_mat,
            str(paths.bold_output("bold_s03_motion_correction.nii.gz")),
            str(paths.masks_dir),
            mode,
            str(paths.bold_output("bold_s07_masks_to_epi.done")),
        ]

    elif step.step_id == "bold_s10_space_mni152":
        warp = str(paths.reg_output("anat_s02_to_mni152_warp.nii.gz"))
        anat_affine = str(paths.reg_output("anat_s01_to_mni152_affine.1D"))
        epi_affine = str(paths.reg_dir / "bold_s05_epi_ref_unifize_align_mat.aff12.1D")
        return [
            str(paths.bold_output("bold_s09_grand_mean_scaling_10000.nii.gz")),
            warp,
            anat_affine,
            epi_affine,
            str(paths.masks_dir / "t1_brain_mask_to_epi.nii.gz"),
            str(paths.bold_output(step.primary_output)),
            mode,
        ]

    raise ValueError(f"No AFNI argument resolver for step: {step.step_id}")


def _slice_timing_config(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("bold", {}).get("slice_timing", False)
    if isinstance(value, dict):
        return value
    return {
        "enabled": bool(value),
        "source": config.get("bold", {}).get("slice_timing_source", "afni_pattern"),
        "afni_pattern": config.get("bold", {}).get("slice_timing_pattern", "alt+z"),
    }


def _slice_timing_enabled(config: dict[str, Any]) -> bool:
    return bool(_slice_timing_config(config).get("enabled", False))


def _post_motion_bold_input(paths: PrepPaths, config: dict[str, Any]) -> Path:
    if _slice_timing_enabled(config):
        return paths.bold_output("bold_s04_slice_timing.nii.gz")
    return paths.bold_output("bold_s03_motion_correction.nii.gz")


def _resolve_slice_timing_pattern(paths: PrepPaths, inputs: SelectedInputs, bold_config: dict[str, Any]) -> str:
    config = {"bold": bold_config}
    timing_config = _slice_timing_config(config)
    source = timing_config.get("source", "afni_pattern")

    if source == "afni_pattern":
        return str(timing_config.get("afni_pattern", "alt+z"))

    if source == "bids_json":
        sidecar = read_bids_json(inputs.bold.path)
        slice_timing = sidecar.get("SliceTiming")
        if not isinstance(slice_timing, list) or not slice_timing:
            raise ValueError(
                f"Slice timing source is bids_json, but no SliceTiming list was found for {inputs.bold.path}"
            )
        timing_file = paths.work_dir / "slice_timing_from_bids.1D"
        timing_file.parent.mkdir(parents=True, exist_ok=True)
        timing_file.write_text(
            "\n".join(str(value) for value in slice_timing) + "\n",
            encoding="utf-8",
        )
        return f"@{timing_file}"

    raise ValueError(f"Unsupported bold.slice_timing.source: {source}")
