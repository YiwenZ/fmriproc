from __future__ import annotations

import shutil
from typing import Any


def check_software(config: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Check that required software is available.

    Returns (required_messages, optional_warnings).
    required_messages → missing tools that should block execution.
    optional_warnings  → informational fallback notes (e.g., ANTs unavailable).
    """
    required: list[str] = []
    warnings: list[str] = []
    backend = config.get("runtime", {}).get("backend", "afni")
    reg_backend = config.get("runtime", {}).get("registration_backend", "ants_if_available")

    if backend == "afni":
        _check_afni(required)
    elif backend == "fsl":
        _check_fsl(required, warnings)

    if reg_backend == "ants":
        _check_ants(required)
    elif reg_backend == "ants_if_available":
        _check_ants_optional(required, warnings)
    elif reg_backend == "fsl":
        _check_fsl(required, warnings)

    _check_samseg(config, required, warnings)

    # afni backend: no ANTs check needed.
    return required, warnings


def _check_afni(messages: list[str]) -> None:
    if shutil.which("3dWarp") is None:
        messages.append("AFNI backend selected but 3dWarp not found on PATH")
    if shutil.which("3dvolreg") is None:
        messages.append("AFNI backend selected but 3dvolreg not found on PATH")


def _check_fsl(messages: list[str], warnings: list[str]) -> None:
    """Check FSL preprocessing toolchain availability."""
    _fsl_tools = [
        "fslreorient2std", "bet", "fast", "fslmaths", "fslroi",
        "mcflirt", "slicetimer", "flirt", "epi_reg", "convert_xfm",
        "fslmeants", "fslstats", "Text2Vest", "fsl_glm",
        "fnirt", "applywarp", "convertwarp", "fslcpgeom",
    ]
    for tool in _fsl_tools:
        if shutil.which(tool) is None:
            messages.append(f"FSL backend selected but {tool} not found on PATH")
    if shutil.which("bc") is None:
        warning = (
            "bc not found on PATH; FSL BET will fall back to DWI-style basic "
            "parameters (-f 0.2 -m) instead of enhanced BET (-f 0.1 -B -R -m)"
        )
        if warning not in warnings:
            warnings.append(warning)


def _check_ants(messages: list[str]) -> None:
    if shutil.which("antsRegistrationSyN.sh") is None:
        messages.append("ANTs registration required but antsRegistrationSyN.sh not found on PATH")
    if shutil.which("antsApplyTransforms") is None:
        messages.append("ANTs registration required but antsApplyTransforms not found on PATH")
    # ANTs-first bold_s10 uses 3dNwarpApply for composite warp application.
    if shutil.which("3dNwarpApply") is None:
        messages.append("ANTs registration pipeline requires 3dNwarpApply (from AFNI) for BOLD→MNI step")


def _check_ants_optional(required: list[str], warnings: list[str]) -> None:
    missing_ants_reg = shutil.which("antsRegistrationSyN.sh") is None
    missing_ants_apply = shutil.which("antsApplyTransforms") is None
    if missing_ants_reg or missing_ants_apply:
        missing = []
        if missing_ants_reg:
            missing.append("antsRegistrationSyN.sh")
        if missing_ants_apply:
            missing.append("antsApplyTransforms")
        warnings.append(
            f"ANTs tools not fully available ({', '.join(missing)} missing); "
            f"will use AFNI registration fallback"
        )


def _check_samseg(config: dict[str, Any], required: list[str], warnings: list[str]) -> None:
    requested = config.get("anat", {}).get("segmentation_backend", "samseg_if_available")
    has_samseg = shutil.which("run_samseg") is not None
    if requested == "samseg" and not has_samseg:
        required.append("run_samseg not found; required by anat.segmentation_backend='samseg'")
    elif requested == "samseg_if_available" and not has_samseg:
        runtime_backend = config.get("runtime", {}).get("backend", "afni")
        warnings.append(f"run_samseg not found; falling back to native {runtime_backend} segmentation")


def fail_on_required(required_messages: list[str]) -> None:
    """Raise SystemExit if any required-software messages are present."""
    if required_messages:
        raise SystemExit("\n".join(required_messages))
