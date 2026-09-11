from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import MetricPaths, PrepPaths


@dataclass(frozen=True)
class MetricsBoldInput:
    path: Path
    backend: str
    smoothed: bool
    reason: str


def resolve_final_mni_bold(paths: PrepPaths, cfg: dict[str, Any], smoothed: bool = False) -> MetricsBoldInput:
    backend = cfg.get("runtime", {}).get("backend", "auto")
    candidates = _candidate_filenames(backend, smoothed)

    for candidate_backend, filename in candidates:
        path = paths.bold_output(filename)
        if path.exists():
            return MetricsBoldInput(
                path=path,
                backend=candidate_backend,
                smoothed=smoothed,
                reason=f"selected {filename} for {candidate_backend} backend",
            )

    expected = ", ".join(filename for _, filename in candidates)
    raise FileNotFoundError(
        f"No final MNI BOLD found for {paths.subject}/{paths.session}/task-{paths.task}. "
        f"Looked in {paths.func_task_dir} for: {expected}"
    )


def _candidate_filenames(backend: str, smoothed: bool) -> list[tuple[str, str]]:
    afni = ("afni", "bold_s11_smooth_sm6.nii.gz" if smoothed else "bold_s10_space_mni152.nii.gz")
    fsl = ("fsl", "bold_s12_smooth_sm6.nii.gz" if smoothed else "bold_s11_space_mni152.nii.gz")
    if backend == "afni":
        return [afni]
    if backend == "fsl":
        return [fsl]
    return [afni, fsl]


# ---------------------------------------------------------------------------
# TR resolution
# ---------------------------------------------------------------------------


def resolve_bold_tr(paths: PrepPaths, cfg: dict[str, Any], bold_path: Path | None = None) -> float:
    """Resolve repetition time (TR) in seconds for ALFF/fALFF computation.

    Resolution order:
    1. Numeric ``bold.tr`` from config.
    2. If ``bold.tr: auto``, try the BIDS JSON sidecar derived from the final
       BOLD filename (e.g. ``bold_s10_space_mni152.nii.gz`` →
       ``bold_s10_space_mni152.json``).
    3. Fall back to any ``*_bold.json`` in the func task directory.
    4. Fail with a clear message.
    """
    tr_cfg = cfg.get("bold", {}).get("tr")
    if tr_cfg is not None and isinstance(tr_cfg, (int, float)) and tr_cfg != "auto":
        return float(tr_cfg)

    # Candidate JSON sidecars, ordered by priority.
    candidates: list[Path] = []

    # Derive sidecar from final BOLD path: foo_bar.nii.gz → foo_bar.json
    if bold_path is not None and bold_path.exists():
        bold_name = bold_path.name
        for suffix in (".nii.gz", ".nii"):
            if bold_name.endswith(suffix):
                json_name = bold_name[: -len(suffix)] + ".json"
                candidates.append(bold_path.with_name(json_name))
                break

    # Also try *_bold.json in the func task dir.
    candidates.extend(sorted(paths.func_task_dir.glob("*_bold.json")))

    # Fallback to original BIDS directory if not copied to Prep dir
    bids_dir_str = cfg.get("dataset", {}).get("bids_dir")
    if bids_dir_str:
        bids_func_dir = Path(bids_dir_str) / paths.subject / paths.session / "func"
        if bids_func_dir.exists():
            candidates.extend(sorted(bids_func_dir.glob(f"*{paths.task}*_bold.json")))

    for candidate in candidates:
        tr = _tr_from_json_sidecar(candidate)
        if tr is not None:
            return tr

    raise RuntimeError(
        f"Cannot resolve TR for {paths.subject}/{paths.session}/task-{paths.task}. "
        f"Set bold.tr explicitly in config (e.g. bold.tr: 2.0), or ensure a "
        f"BIDS JSON sidecar is present in {paths.func_task_dir}."
    )


def _tr_from_json_sidecar(json_path: Path | None) -> float | None:
    if json_path is None or not json_path.exists():
        return None
    try:
        sidecar = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    tr = sidecar.get("RepetitionTime")
    if tr is not None and isinstance(tr, (int, float)):
        return float(tr)
    return None


# ---------------------------------------------------------------------------
# Group-mask resolution for regional metrics
# ---------------------------------------------------------------------------


def resolve_group_mask_for_metrics(
    metric_paths: MetricPaths,
    cfg: dict[str, Any],
    bold_path: Path,
    tool_name: str = "afni_regional",
    dry_run: bool = False,
) -> Path | None:
    """Resolve and optionally resample the dataset group mask.

    Returns the path to a mask aligned to *bold_path*'s grid, or ``None``
    when ``metrics.regional.group_mask.apply`` is ``False``.

    The resampled copy is cached under ``raw/<tool_name>/``.
    """
    regional_cfg = cfg.get("metrics", {}).get("regional", {})
    mask_cfg = regional_cfg.get("group_mask", {})
    if not mask_cfg.get("apply", True):
        return None

    constraint_filename = mask_cfg.get("constraint", "grp_mask_thr70.nii.gz")
    group_mask_path = metric_paths.prep_paths.prep_dir / "group_mask" / constraint_filename
    if not group_mask_path.exists():
        raise FileNotFoundError(
            f"Group mask not found: {group_mask_path}. "
            f"Run 'fmriproc group-mask --prep-dir {metric_paths.prep_paths.prep_dir}' first."
        )

    # Check for grid mismatch; resample if needed.
    from .netcorr import _nifti_grid_mismatch, _maybe_resample_nifti

    raw_dir = metric_paths.raw_tool_dir(tool_name)
    if _nifti_grid_mismatch(group_mask_path, bold_path):
        resampled, _ = _maybe_resample_nifti(
            "group_mask",
            group_mask_path,
            bold_path,
            raw_dir,
            dry_run,
        )
        return resampled
    return group_mask_path
