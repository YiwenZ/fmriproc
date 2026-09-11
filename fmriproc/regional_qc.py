"""QC and metadata sidecars for regional metric runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .manifest import write_tsv
from .paths import MetricPaths


def write_regional_qc(
    metric_paths: MetricPaths,
    requested_metrics: list[str],
    atlas_names: list[str],
    bold_path: Path,
    group_mask_path: Path | None,
    rsfc_result: Any | None,
    reho_result: Any | None,
    roi_paths: dict[tuple[str, str], Path],
    cfg: dict[str, Any],
) -> tuple[Path, Path]:
    """Write ``regional_metrics_qc.tsv`` and ``regional_metrics.json``.

    Returns ``(qc_tsv_path, metadata_json_path)``.
    """
    qc_path = metric_paths.regional_qc_tsv_output()
    metadata_path = metric_paths.regional_metadata_json_output()

    motion_summary = _read_motion_summary(metric_paths.prep_paths.qc_output("motion_summary.tsv"))

    # Count low-coverage ROIs from ROI TSVs.
    min_cov = float(cfg.get("metrics", {}).get("regional", {}).get("roi_summary", {}).get("min_roi_coverage", 0.5))
    n_low_coverage = 0
    total_rois = 0
    coverage_sum = 0.0
    for (atlas, metric), path in roi_paths.items():
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                total_rois += 1
                cov = float(row.get("coverage", 0))
                coverage_sum += cov
                if cov < min_cov:
                    n_low_coverage += 1
    mean_coverage = coverage_sum / total_rois if total_rois > 0 else 0.0

    row: dict[str, Any] = {
        "subject": metric_paths.prep_paths.subject,
        "session": metric_paths.prep_paths.session,
        "task": metric_paths.prep_paths.task,
        "input_bold": str(bold_path),
        "group_mask": str(group_mask_path or ""),
        "metrics_requested": ",".join(requested_metrics),
        "atlases_requested": ",".join(atlas_names),
        "alff_map": str(rsfc_result.alff_map) if rsfc_result else "",
        "falff_map": str(rsfc_result.falff_map) if rsfc_result else "",
        "reho_map": str(reho_result.reho_map) if reho_result else "",
        "n_low_coverage_rois": n_low_coverage,
        "mean_roi_coverage": f"{mean_coverage:.6f}",
        **motion_summary,
    }
    write_tsv(qc_path, list(row.keys()), [row])

    metadata = {
        "subject": metric_paths.prep_paths.subject,
        "session": metric_paths.prep_paths.session,
        "task": metric_paths.prep_paths.task,
        "calculation_space": "MNI",
        "software_backend": "afni",
        "metrics_requested": requested_metrics,
        "atlases_requested": atlas_names,
        "commands": {
            "3dRSFC": rsfc_result.command if rsfc_result else [],
            "3dReHo": reho_result.command if reho_result else [],
        },
        "parameters": cfg.get("metrics", {}).get("regional", {}),
        "paths": {
            "input_bold": str(bold_path),
            "group_mask": str(group_mask_path or ""),
            "alff_map": str(rsfc_result.alff_map) if rsfc_result else "",
            "falff_map": str(rsfc_result.falff_map) if rsfc_result else "",
            "reho_map": str(reho_result.reho_map) if reho_result else "",
            "roi_summaries": {f"{atlas}_{metric}": str(p) for (atlas, metric), p in roi_paths.items()},
        },
        "summary": {
            "n_low_coverage_rois": n_low_coverage,
            "mean_roi_coverage": f"{mean_coverage:.6f}",
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return qc_path, metadata_path


def _read_motion_summary(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        try:
            row = next(reader)
        except StopIteration:
            return {}
    excluded = {"subject", "session", "task"}
    return {f"motion_{key}": value for key, value in row.items() if key not in excluded}
