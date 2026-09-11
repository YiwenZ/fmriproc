from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .manifest import write_tsv
from .netcorr import NetCorrResult
from .netcorr_outputs import StandardizedNetCorrOutputs
from .paths import MetricPaths


def write_metrics_qc(
    metric_paths: MetricPaths,
    atlas_name: str,
    netcorr: NetCorrResult,
    standardized: StandardizedNetCorrOutputs,
    cfg: dict[str, Any],
) -> tuple[Path, Path]:
    qc_path = metric_paths.qc_tsv_output(atlas_name)
    metadata_path = metric_paths.metadata_json_output(atlas_name)
    motion_summary = _read_motion_summary(metric_paths.prep_paths.qc_output("motion_summary.tsv"))

    row = {
        "subject": metric_paths.prep_paths.subject,
        "session": metric_paths.prep_paths.session,
        "task": metric_paths.prep_paths.task,
        "atlas": atlas_name,
        "input_bold": str(netcorr.bold_path),
        "atlas_path": str(netcorr.atlas_path),
        "resampled_atlas_path": str(netcorr.resampled_atlas_path or ""),
        "group_mask_path": str(netcorr.group_mask_path or ""),
        "resampled_group_mask_path": str(netcorr.resampled_group_mask_path or ""),
        "constrained_roi_path": str(netcorr.constrained_roi_path or ""),
        "roi_labels_before_group_mask": netcorr.roi_labels_before_group_mask,
        "roi_labels_after_group_mask": netcorr.roi_labels_after_group_mask,
        "n_timepoints": standardized.n_timepoints,
        "n_rois": standardized.n_rois,
        "n_edges": standardized.n_edges,
        "n_missing_rois": standardized.n_missing_rois,
        "n_constant_rois": standardized.n_constant_rois,
        **motion_summary,
    }
    write_tsv(qc_path, list(row.keys()), [row])

    metadata = {
        "subject": metric_paths.prep_paths.subject,
        "session": metric_paths.prep_paths.session,
        "task": metric_paths.prep_paths.task,
        "atlas": atlas_name,
        "software_backend": "afni_3dNetCorr",
        "command": netcorr.command,
        "parameters": cfg.get("metrics", {}),
        "paths": {
            "input_bold": str(netcorr.bold_path),
            "atlas": str(netcorr.atlas_path),
            "roi_input": str(netcorr.roi_path),
            "resampled_atlas": str(netcorr.resampled_atlas_path or ""),
            "group_mask": str(netcorr.group_mask_path or ""),
            "resampled_group_mask": str(netcorr.resampled_group_mask_path or ""),
            "constrained_roi": str(netcorr.constrained_roi_path or ""),
            "raw_netcc": str(netcorr.netcc_path),
            "raw_netts": str(netcorr.netts_path),
            "log": str(netcorr.log_path),
            "timeseries_tsv": str(standardized.timeseries_tsv),
            "fc_correlation_tsv": str(standardized.fc_correlation_tsv),
            "fc_fisherz_tsv": str(standardized.fc_fisherz_tsv),
            "fc_long_tsv": str(standardized.fc_long_tsv),
            "qc_tsv": str(qc_path),
        },
        "summary": {
            "n_timepoints": standardized.n_timepoints,
            "n_rois": standardized.n_rois,
            "n_edges": standardized.n_edges,
            "n_missing_rois": standardized.n_missing_rois,
            "n_constant_rois": standardized.n_constant_rois,
            "roi_labels_before_group_mask": netcorr.roi_labels_before_group_mask,
            "roi_labels_after_group_mask": netcorr.roi_labels_after_group_mask,
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
