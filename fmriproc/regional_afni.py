"""AFNI runners for regional fMRI metrics: ALFF/fALFF (``3dRSFC``) and ReHo
(``3dReHo``).

Each runner builds an AFNI command, executes it (with dry-run / force support),
writes command-native outputs under ``raw/``, and converts results to NIfTI for
downstream use.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import MetricPaths


@dataclass(frozen=True)
class RSFCResult:
    command: list[str]
    log_path: Path
    raw_prefix: str
    alff_map: Path
    falff_map: Path
    mask_path: Path | None


@dataclass(frozen=True)
class ReHoResult:
    command: list[str]
    log_path: Path
    reho_map: Path
    mask_path: Path | None


# ---------------------------------------------------------------------------
# 3dRSFC – ALFF / fALFF
# ---------------------------------------------------------------------------


def build_3drsfc_command(
    bold_path: Path,
    mask_path: Path,
    raw_prefix: Path,
    tr: float,
    low_hz: float = 0.01,
    high_hz: float = 0.1,
) -> list[str]:
    """Build a ``3dRSFC`` command for ALFF/fALFF computation.

    Returns a list of command tokens.
    """
    return [
        "3dRSFC",
        "-nodetrend",
        "-dt",
        str(tr),
        "-mask",
        str(mask_path),
        "-no_rs_out",
        "-no_rsfa",
        "-notrans",
        "-prefix",
        str(raw_prefix),
        str(low_hz),
        str(high_hz),
        str(bold_path),
    ]


def run_3drsfc(
    bold_path: Path,
    metric_paths: MetricPaths,
    cfg: dict[str, Any],
    group_mask_path: Path | None = None,
    tr: float | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> RSFCResult:
    """Run AFNI ``3dRSFC`` for ALFF/fALFF and convert outputs to NIfTI."""
    from .metrics_inputs import resolve_bold_tr

    metric_paths.ensure_dirs()
    raw_dir = metric_paths.raw_tool_dir("afni_3dRSFC")
    raw_prefix = raw_dir / metric_paths._metric_stem("rsfc")

    if tr is None:
        tr = resolve_bold_tr(metric_paths.prep_paths, cfg, bold_path)

    regional_cfg = cfg.get("metrics", {}).get("regional", {})
    low_hz = float(regional_cfg.get("frequency", {}).get("low_hz", 0.01))
    high_hz = float(regional_cfg.get("frequency", {}).get("high_hz", 0.1))

    if group_mask_path is None:
        mask_path = raw_dir / "automask.nii.gz"
        if not dry_run and (not mask_path.exists() or force):
            subprocess.run(["3dAutomask", "-prefix", str(mask_path), str(bold_path)], check=True)
    else:
        mask_path = group_mask_path
    command = build_3drsfc_command(bold_path, mask_path, raw_prefix, tr, low_hz, high_hz)
    log_path = raw_dir / f"{metric_paths._metric_stem('rsfc')}_3dRSFC.log"

    # Expected NIfTI outputs.
    alff_map = metric_paths.map_output("alff")
    falff_map = metric_paths.map_output("falff")

    if dry_run:
        _write_log(log_path, f"DRY RUN: {_quote(command)}\n")
        return RSFCResult(command=command, log_path=log_path, raw_prefix=str(raw_prefix),
                          alff_map=alff_map, falff_map=falff_map, mask_path=mask_path)

    if alff_map.exists() and falff_map.exists() and not force:
        _write_log(log_path, f"SKIP existing outputs: {_quote(command)}\n")
        return RSFCResult(command=command, log_path=log_path, raw_prefix=str(raw_prefix),
                          alff_map=alff_map, falff_map=falff_map, mask_path=mask_path)

    if shutil.which("3dRSFC") is None:
        raise RuntimeError("AFNI 3dRSFC is not available on PATH.")

    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    _write_log(
        log_path,
        "\n".join([
            f"COMMAND: {_quote(command)}",
            f"RETURN_CODE: {completed.returncode}",
            "STDOUT:", completed.stdout,
            "STDERR:", completed.stderr,
        ]),
    )
    if completed.returncode != 0:
        raise RuntimeError(f"3dRSFC failed; see {log_path}")

    # Convert AFNI HEAD/BRIK → NIfTI.
    _afni_to_nifti(f"{raw_prefix}_ALFF+orig", alff_map)
    _afni_to_nifti(f"{raw_prefix}_fALFF+orig", falff_map)

    return RSFCResult(command=command, log_path=log_path, raw_prefix=str(raw_prefix),
                      alff_map=alff_map, falff_map=falff_map, mask_path=mask_path)


# ---------------------------------------------------------------------------
# 3dReHo
# ---------------------------------------------------------------------------


def build_3dreho_command(
    bold_path: Path,
    mask_path: Path,
    output_map: Path,
    nneigh: int = 27,
) -> list[str]:
    """Build a ``3dReHo`` command."""
    return [
        "3dReHo",
        "-prefix",
        str(output_map),
        "-inset",
        str(bold_path),
        "-mask",
        str(mask_path),
        "-nneigh",
        str(nneigh),
    ]


def run_3dreho(
    bold_path: Path,
    metric_paths: MetricPaths,
    cfg: dict[str, Any],
    group_mask_path: Path | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> ReHoResult:
    """Run AFNI ``3dReHo`` and convert output to NIfTI."""
    metric_paths.ensure_dirs()
    raw_dir = metric_paths.raw_tool_dir("afni_3dReHo")

    regional_cfg = cfg.get("metrics", {}).get("regional", {})
    nneigh = int(regional_cfg.get("reho", {}).get("neighborhood", 27))
    if nneigh not in (7, 19, 27):
        raise ValueError(f"ReHo neighborhood must be 7, 19, or 27; got {nneigh}")

    if group_mask_path is None:
        mask_path = raw_dir / "automask.nii.gz"
        if not dry_run and (not mask_path.exists() or force):
            subprocess.run(["3dAutomask", "-prefix", str(mask_path), str(bold_path)], check=True)
    else:
        mask_path = group_mask_path
    # 3dReHo writes BRIK/HEAD to its -prefix; we convert to .nii.gz below.
    raw_prefix = raw_dir / metric_paths._metric_stem("reho")
    command = build_3dreho_command(bold_path, mask_path, raw_prefix, nneigh)
    log_path = raw_dir / f"{metric_paths._metric_stem('reho')}_3dReHo.log"

    reho_map = metric_paths.map_output("reho")

    if dry_run:
        _write_log(log_path, f"DRY RUN: {_quote(command)}\n")
        return ReHoResult(command=command, log_path=log_path, reho_map=reho_map, mask_path=mask_path)

    if reho_map.exists() and not force:
        _write_log(log_path, f"SKIP existing outputs: {_quote(command)}\n")
        return ReHoResult(command=command, log_path=log_path, reho_map=reho_map, mask_path=mask_path)

    if shutil.which("3dReHo") is None:
        raise RuntimeError("AFNI 3dReHo is not available on PATH.")

    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    _write_log(
        log_path,
        "\n".join([
            f"COMMAND: {_quote(command)}",
            f"RETURN_CODE: {completed.returncode}",
            "STDOUT:", completed.stdout,
            "STDERR:", completed.stderr,
        ]),
    )
    if completed.returncode != 0:
        raise RuntimeError(f"3dReHo failed; see {log_path}")

    _afni_to_nifti(f"{raw_prefix}+orig", reho_map)
    return ReHoResult(command=command, log_path=log_path, reho_map=reho_map, mask_path=mask_path)


# ---------------------------------------------------------------------------
# NIfTI conversion
# ---------------------------------------------------------------------------


def _afni_to_nifti(afni_prefix: str, output_path: Path) -> None:
    """Convert an AFNI HEAD/BRIK pair to NIfTI ``.nii.gz``.

    Prefers ``3dAFNItoNIFTI``; falls back to nibabel when AFNI conversion is
    unavailable and nibabel supports AFNI loading.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("3dAFNItoNIFTI") is not None:
        cmd = ["3dAFNItoNIFTI", "-prefix", str(output_path), afni_prefix]
        completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            # AFNI converter may rename the output; check common pattern.
            alt = output_path.parent / f"{output_path.name.removesuffix('.nii.gz')}.nii"
            if not alt.exists():
                raise RuntimeError(f"3dAFNItoNIFTI failed for {afni_prefix}: {completed.stderr}")
        return

    # nibabel fallback.
    try:
        import nibabel as nib
    except ImportError:
        raise RuntimeError(
            f"Cannot convert {afni_prefix} to NIfTI. "
            f"3dAFNItoNIFTI is not available and nibabel is not installed."
        )

    try:
        img = nib.load(afni_prefix)
        nib.save(img, str(output_path))
    except Exception as exc:
        raise RuntimeError(
            f"NIfTI conversion failed for {afni_prefix}. "
            f"Install AFNI (3dAFNItoNIFTI) or ensure nibabel supports AFNI I/O. "
            f"Error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _quote(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)
