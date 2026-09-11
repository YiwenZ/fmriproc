from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from .manifest import write_tsv


@dataclass(frozen=True)
class PreprocessSummaryOutputs:
    preprocess_qc_tsv: Path
    assets_tsv: Path
    summary_json: Path


MAIN_FIELDS = [
    "subject",
    "session",
    "task",
    "bold_path",
    "t1_path",
    "has_t1",
    "has_bold",
    "has_sbref",
    "has_fmap_ap",
    "has_fmap_pa",
    "fallback_reason",
    "preprocess_status",
    "last_step",
    "failed_step",
    "failed_reason",
    "n_steps_completed",
    "n_steps_failed",
    "n_steps_skipped",
    "slice_timing_status",
    "segmentation_backend_requested",
    "segmentation_backend_used",
    "segmentation_source",
    "brain_mask_voxels",
    "wm_mask_voxels",
    "csf_mask_voxels",
    "n_volumes_raw",
    "n_dummy_removed",
    "n_volumes_used",
    "mean_fd_power",
    "median_fd_power",
    "max_fd_power",
    "percent_fd_power_gt_0_2",
    "percent_fd_power_gt_0_5",
    "nuisance_model",
    "nuisance_n_columns",
    "final_bold_path",
    "final_bold_exists",
    "final_bold_shape",
    "final_bold_n_volumes",
    "qc_status",
    "qc_flags",
]

ASSET_FIELDS = [
    "subject",
    "session",
    "task",
    "epi_to_t1_overlay_path",
    "epi_to_mni_overlay_path",
    "motion_plot_path",
    "precheck_t1_slices_path",
    "precheck_epi_slices_path",
    "final_bold_path",
    "motion_summary_path",
    "status_jsonl_path",
]


def write_preprocess_summary(prep_dir: Path) -> PreprocessSummaryOutputs:
    prep_dir = Path(prep_dir)
    qc_dir = prep_dir / "derivatives" / "qc"
    rows = [_build_run_row(prep_dir, run) for run in _discover_runs(prep_dir)]
    rows.sort(key=lambda row: (row["subject"], row["session"], row["task"]))

    outputs = PreprocessSummaryOutputs(
        preprocess_qc_tsv=qc_dir / "preprocess_qc.tsv",
        assets_tsv=qc_dir / "preprocess_qc_assets.tsv",
        summary_json=qc_dir / "preprocess_qc_summary.json",
    )
    write_tsv(outputs.preprocess_qc_tsv, MAIN_FIELDS, rows)
    write_tsv(outputs.assets_tsv, ASSET_FIELDS, rows)
    outputs.summary_json.parent.mkdir(parents=True, exist_ok=True)
    outputs.summary_json.write_text(
        json.dumps(_build_summary(rows, prep_dir), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return outputs


def _discover_runs(prep_dir: Path) -> list[tuple[str, str, str]]:
    runs: set[tuple[str, str, str]] = set()
    for task_dir in prep_dir.glob("sub-*/ses-*/qc/task-*"):
        runs.add((task_dir.parents[2].name, task_dir.parents[1].name, task_dir.name.removeprefix("task-")))
    for task_dir in prep_dir.glob("sub-*/ses-*/logs/task-*"):
        runs.add((task_dir.parents[2].name, task_dir.parents[1].name, task_dir.name.removeprefix("task-")))
    return sorted(runs)


def _build_run_row(prep_dir: Path, run: tuple[str, str, str]) -> dict[str, Any]:
    subject, session, task = run
    session_dir = prep_dir / subject / session
    qc_dir = session_dir / "qc" / f"task-{task}"
    log_dir = session_dir / "logs" / f"task-{task}"
    func_dir = session_dir / "func" / f"task-{task}"
    masks_dir = session_dir / "masks"

    selected = _read_first_tsv(qc_dir / "selected_inputs.tsv")
    input_qc = _read_first_tsv(qc_dir / "input_qc.tsv")
    motion = _read_first_tsv(qc_dir / "motion_summary.tsv")
    statuses = _read_status(log_dir / "status.jsonl")
    nuisance = _read_json(qc_dir / "nuisance_metadata.json")
    segmentation = _resolve_segmentation(session_dir, qc_dir)
    final_bold = _resolve_final_bold(func_dir)
    final_shape, final_n_volumes = _nifti_shape(final_bold)

    failed = [record for record in statuses if record.get("status") == "failed"]
    skipped = [record for record in statuses if str(record.get("status", "")).startswith("skipped")]
    completed = [record for record in statuses if record.get("status") == "completed"]
    slice_timing = next(
        (record.get("status", "") for record in statuses if "slice_timing" in str(record.get("step_id", ""))),
        "",
    )

    row: dict[str, Any] = {
        "subject": subject,
        "session": session,
        "task": task,
        "bold_path": selected.get("bold_path", ""),
        "t1_path": selected.get("t1_path", ""),
        "has_t1": input_qc.get("has_t1", ""),
        "has_bold": input_qc.get("has_bold", ""),
        "has_sbref": input_qc.get("has_sbref", ""),
        "has_fmap_ap": input_qc.get("has_fmap_ap", ""),
        "has_fmap_pa": input_qc.get("has_fmap_pa", ""),
        "fallback_reason": selected.get("fallback_reason", ""),
        "preprocess_status": _preprocess_status(statuses, final_bold, failed),
        "last_step": statuses[-1].get("step_id", "") if statuses else "",
        "failed_step": failed[0].get("step_id", "") if failed else "",
        "failed_reason": failed[0].get("reason", "") if failed else "",
        "n_steps_completed": len(completed),
        "n_steps_failed": len(failed),
        "n_steps_skipped": len(skipped),
        "slice_timing_status": slice_timing,
        "segmentation_backend_requested": segmentation["requested"],
        "segmentation_backend_used": segmentation["used"],
        "segmentation_source": segmentation["source"],
        "brain_mask_voxels": _mask_voxels(masks_dir / "t1_brain_mask.nii.gz"),
        "wm_mask_voxels": _mask_voxels(masks_dir / "t1_wm_mask.nii.gz"),
        "csf_mask_voxels": _mask_voxels(masks_dir / "t1_csf_mask.nii.gz"),
        "n_volumes_raw": motion.get("n_volumes_raw", ""),
        "n_dummy_removed": motion.get("dummy_removed", ""),
        "n_volumes_used": motion.get("n_volumes_used", ""),
        "mean_fd_power": motion.get("mean_fd_power", ""),
        "median_fd_power": motion.get("median_fd_power", ""),
        "max_fd_power": motion.get("max_fd_power", ""),
        "percent_fd_power_gt_0_2": motion.get("percent_fd_power_gt_0_2", ""),
        "percent_fd_power_gt_0_5": motion.get("percent_fd_power_gt_0_5", ""),
        "nuisance_model": nuisance.get("model", ""),
        "nuisance_n_columns": nuisance.get("n_columns", ""),
        "final_bold_path": str(final_bold) if final_bold else "",
        "final_bold_exists": bool(final_bold),
        "final_bold_shape": final_shape,
        "final_bold_n_volumes": final_n_volumes,
        "epi_to_t1_overlay_path": _qc_asset(qc_dir, "epi_to_t1_overlay.png"),
        "epi_to_mni_overlay_path": _qc_asset(qc_dir, "epi_to_mni_overlay.png"),
        "motion_plot_path": _qc_asset(qc_dir, "motion_parameters.png"),
        "precheck_t1_slices_path": _qc_asset(qc_dir, "precheck_t1_slices.png"),
        "precheck_epi_slices_path": _qc_asset(qc_dir, "precheck_epi_slices.png"),
        "motion_summary_path": _existing_path(qc_dir / "motion_summary.tsv"),
        "status_jsonl_path": _existing_path(log_dir / "status.jsonl"),
    }
    flags = _qc_flags(row, masks_dir, qc_dir, _expects_mni_overlay(statuses, final_bold))
    row["qc_flags"] = ";".join(flags)
    fail_flags = {"failed_step", "missing_final_bold", "missing_masks", "high_fd_outlier_ratio"}
    row["qc_status"] = "fail" if fail_flags.intersection(flags) else "warn" if flags else "pass"
    return row


def _existing_path(path: Path) -> str:
    return str(path) if path.exists() else ""


def _qc_asset(qc_dir: Path, filename: str) -> str:
    for path in [qc_dir / filename, qc_dir / "snapshots" / filename]:
        if path.exists():
            return str(path)
    return ""


def _read_first_tsv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return rows[0] if rows else {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_status(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _resolve_segmentation(session_dir: Path, qc_dir: Path) -> dict[str, str]:
    provenance = qc_dir / "segmentation_backend.json"
    if provenance.exists():
        data = _read_json(provenance)
        return {
            "requested": str(data.get("segmentation_backend_requested", "")),
            "used": str(data.get("segmentation_backend_used", "")),
            "source": "provenance",
        }
    if (session_dir / "anat" / "samseg" / "seg.mgz").exists():
        return {"requested": "", "used": "samseg", "source": "inferred"}
    if (session_dir / "anat" / "Segsy").exists() or (session_dir / "anat" / "Segsy" / "Classes+orig.HEAD").exists():
        return {"requested": "", "used": "afni_3dSeg", "source": "inferred"}
    if (session_dir / "anat" / "seg" / "t1_seg_pve_0.nii.gz").exists():
        return {"requested": "", "used": "fsl_fast", "source": "inferred"}
    return {"requested": "", "used": "legacy_unknown", "source": "inferred"}


def _resolve_final_bold(func_dir: Path) -> Path | None:
    for name in ["bold_s11_smooth_sm6.nii.gz", "bold_s10_space_mni152.nii.gz", "bold_s08_grand_mean_scaled.nii.gz"]:
        path = func_dir / name
        if path.exists():
            return path
    return None


def _nifti_shape(path: Path | None) -> tuple[str, str]:
    if path is None:
        return "", ""
    img = nib.load(str(path))
    shape = img.shape
    return "x".join(str(value) for value in shape), str(shape[3] if len(shape) == 4 else "")


def _mask_voxels(path: Path) -> str:
    if not path.exists():
        return ""
    data = np.asanyarray(nib.load(str(path)).dataobj)
    return str(int(np.count_nonzero(data)))


def _preprocess_status(statuses: list[dict[str, Any]], final_bold: Path | None, failed: list[dict[str, Any]]) -> str:
    if failed:
        return "failed"
    if final_bold:
        return "completed"
    if statuses:
        return "partial"
    return "missing"


def _expects_mni_overlay(statuses: list[dict[str, Any]], final_bold: Path | None) -> bool:
    if final_bold and "mni152" in final_bold.name.lower():
        return True
    return any(
        record.get("status") == "completed" and "mni" in str(record.get("step_id", "")).lower()
        for record in statuses
    )


def _qc_flags(row: dict[str, Any], masks_dir: Path, qc_dir: Path, expects_mni_overlay: bool) -> list[str]:
    flags: list[str] = []
    if row["failed_step"]:
        flags.append("failed_step")
    if not row["final_bold_exists"]:
        flags.append("missing_final_bold")
    if not all((masks_dir / name).exists() for name in ["t1_brain_mask.nii.gz", "t1_wm_mask.nii.gz", "t1_csf_mask.nii.gz"]):
        flags.append("missing_masks")
    if _float(row["percent_fd_power_gt_0_5"]) > 20:
        flags.append("high_fd_outlier_ratio")
    if _float(row["mean_fd_power"]) > 0.5:
        flags.append("high_mean_fd")
    if not row["motion_summary_path"]:
        flags.append("missing_motion_summary")
    if not row["nuisance_model"]:
        flags.append("missing_nuisance_metadata")
    if not _qc_asset(qc_dir, "epi_to_t1_overlay.png"):
        flags.append("missing_epi_to_t1_overlay")
    if expects_mni_overlay and not _qc_asset(qc_dir, "epi_to_mni_overlay.png"):
        flags.append("missing_epi_to_mni_overlay")
    if row["segmentation_backend_used"] == "legacy_unknown":
        flags.append("segmentation_backend_unknown")
    return flags


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _build_summary(rows: list[dict[str, Any]], prep_dir: Path) -> dict[str, Any]:
    statuses = Counter(row["qc_status"] for row in rows)
    backends = Counter(row["segmentation_backend_used"] for row in rows)
    return {
        "n_runs_total": len(rows),
        "n_runs_pass": statuses.get("pass", 0),
        "n_runs_warn": statuses.get("warn", 0),
        "n_runs_fail": statuses.get("fail", 0),
        "metrics_missing_rois": _metrics_missing_roi_rows(prep_dir),
        "segmentation_backends": dict(sorted(backends.items())),
    }


def _metrics_missing_roi_rows(prep_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = set(prep_dir.glob("sub-*/ses-*/metrics/task-*/qc/*_metrics-qc.tsv"))
    paths.update(prep_dir.glob("sub-*/ses-*/metrics/task-*/qc/*_metrics_qc.tsv"))
    for path in sorted(paths):
        row = _read_first_tsv(path)
        n_missing = int(_float(row.get("n_missing_rois", 0)))
        if n_missing <= 0:
            continue
        rows.append(
            {
                "subject": row.get("subject", path.parents[4].name),
                "session": row.get("session", path.parents[3].name),
                "task": row.get("task", path.parents[1].name.removeprefix("task-")),
                "atlas": row.get("atlas", ""),
                "n_missing_rois": n_missing,
            }
        )
    return rows
