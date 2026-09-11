"""Dataset-level metrics aggregation.

Discovers completed subject/session/task runs, applies inclusion rules,
optionally executes single-run FC metrics, and writes analysis-ready group
tables under ``Prep/derivatives/metrics/``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .atlas_registry import AtlasSpec
from .motion_qc_utils import evaluate_task_motion
from .paths import DatasetMetricPaths, MetricPaths, PrepPaths


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DatasetRun:
    """A single subject-session-task run discovered under ``Prep/``."""

    subject: str
    session: str
    task: str
    prep_paths: PrepPaths
    metric_paths: MetricPaths
    status: str = "pending"
    reason: str = ""
    final_bold_path: Path | None = None
    motion_summary_path: Path | None = None
    mean_fd: float | None = None
    fd_outlier_ratio: float | None = None
    group_mask_path: Path | None = None
    atlases_requested: list[str] = field(default_factory=list)
    metrics_outputs_present: list[str] = field(default_factory=list)
    metrics_outputs_missing: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_runs(
    prep_dir: Path,
    tasks: list[str] | None = None,
    subjects: list[str] | None = None,
    sessions: list[str] | None = None,
) -> list[DatasetRun]:
    """Scan ``Prep/sub-*/ses-*/func/task-*`` and return one ``DatasetRun`` per
    discovered subject-session-task combination.

    Parameters
    ----------
    prep_dir:
        Root of the preprocessing output tree.
    tasks:
        If non-empty, only include runs whose task name (without ``task-``
        prefix) appears in this list.
    subjects:
        If non-empty, only include runs whose subject appears in this list.
    sessions:
        If non-empty, only include runs whose session appears in this list.
    """
    runs: list[DatasetRun] = []
    task_set = set(tasks or [])
    subject_set = set(subjects or [])
    session_set = set(sessions or [])

    for func_dir in sorted(prep_dir.glob("sub-*/ses-*/func/task-*")):
        if not func_dir.is_dir():
            continue
        task_dir = func_dir
        session_dir = task_dir.parent.parent  # ses-*
        subject_dir = session_dir.parent  # sub-*

        subject = subject_dir.name
        session = session_dir.name
        task = task_dir.name.removeprefix("task-")

        if subject_set and subject not in subject_set:
            continue
        if session_set and session not in session_set:
            continue
        if task_set and task not in task_set:
            continue

        prep_paths = PrepPaths(prep_dir, subject, session, task)
        metric_paths = MetricPaths(prep_paths)
        runs.append(
            DatasetRun(
                subject=subject,
                session=session,
                task=task,
                prep_paths=prep_paths,
                metric_paths=metric_paths,
            )
        )

    return runs


# ---------------------------------------------------------------------------
# Inclusion evaluation
# ---------------------------------------------------------------------------


def evaluate_run_inclusion(
    run: DatasetRun,
    cfg: dict[str, Any],
    atlases: list[AtlasSpec],
    selected_metrics: list[str] | None = None,
    allow_missing_outputs: bool = False,
) -> None:
    """Evaluate inclusion for a single *run* and mutate its status/reason fields.

    Checks are applied in order: final BOLD existence → motion QC →
    group-mask existence → metrics output presence.
    """
    dataset_cfg = cfg.get("metrics", {}).get("dataset", {})
    inclusion_cfg = dataset_cfg.get("inclusion", {})
    require_group_mask = inclusion_cfg.get("require_group_mask", True)
    require_motion_qc = inclusion_cfg.get("require_motion_qc", True)

    # 1. Resolve final MNI BOLD.
    from .metrics_inputs import resolve_final_mni_bold

    try:
        bold_input = resolve_final_mni_bold(run.prep_paths, cfg, smoothed=False)
        run.final_bold_path = bold_input.path
    except FileNotFoundError:
        run.status = "excluded"
        run.reason = "missing_final_bold"
        return

    # 2. Motion QC.
    if require_motion_qc:
        motion_cfg: dict[str, Any] = inclusion_cfg.get("motion", {})
        if not motion_cfg:
            # Fall back to group_mask selection motion defaults.
            motion_cfg = cfg.get("metrics", {}).get("group_mask", {}).get("selection", {}).get("motion", {})
        selection_cfg = {
            "require_motion_qc": True,
            "motion": motion_cfg,
        }
        motion_decision = evaluate_task_motion(run.prep_paths.session_dir, run.task, selection_cfg)
        run.motion_summary_path = motion_decision.get("motion_summary_path")
        run.mean_fd = motion_decision.get("mean_fd")
        run.fd_outlier_ratio = motion_decision.get("fd_outlier_ratio")
        if not motion_decision["include"]:
            run.status = "excluded"
            run.reason = str(motion_decision["reason"])
            return

    # 3. Group mask.
    if require_group_mask:
        group_mask_filename = cfg.get("metrics", {}).get("group_mask", {}).get("default_constraint", "grp_mask_thr70.nii.gz")
        group_mask_path = prep_dir_from_cfg(cfg, run.prep_paths.prep_dir) / "group_mask" / group_mask_filename
        if not group_mask_path.exists():
            run.status = "excluded"
            run.reason = "missing_group_mask"
            return
        run.group_mask_path = group_mask_path

    # 4. Check metrics outputs for all selected families. Gradient is a
    # dataset-level family, so inclusion checks its FC prerequisites rather
    # than requiring gradient outputs to already exist.
    metrics_to_check = selected_metrics if selected_metrics else ["fc"]
    existing_metric_outputs = [metric for metric in metrics_to_check if metric not in {"combat", "gradient"}]
    if "combat" in metrics_to_check and "fc" not in existing_metric_outputs:
        existing_metric_outputs.append("fc")
    run.atlases_requested = [atlas.name for atlas in atlases]
    present: list[str] = []
    missing: list[str] = []
    for atlas in atlases:
        expected = expected_metric_outputs(run.metric_paths, atlas.name, existing_metric_outputs)
        for label, path in expected:
            if path.exists():
                present.append(label)
            else:
                missing.append(label)
        if "gradient" in metrics_to_check:
            from .gradient_outputs import check_gradient_fc_prerequisites

            gradient_missing = check_gradient_fc_prerequisites(
                run.metric_paths, atlas.name, cfg, force_raw=("combat" in metrics_to_check)
            )
            if gradient_missing:
                missing.extend(f"gradient_fc_prerequisite:{item}" for item in gradient_missing)
            else:
                present.append(f"gradient_fc_prerequisites:{atlas.name}")
    run.metrics_outputs_present = present
    run.metrics_outputs_missing = missing

    if missing:
        run_existing = dataset_cfg.get("include_existing_metrics_only", True)
        run_missing_flag = allow_missing_outputs or dataset_cfg.get("run_missing", False)
        if run_existing and not run_missing_flag:
            run.status = "excluded"
            run.reason = "missing_metrics_outputs"
            return

    run.status = "included"
    run.reason = "ok"


def prep_dir_from_cfg(cfg: dict[str, Any], fallback: Path) -> Path:
    """Return the prep directory used for dataset-level paths.

    The dataset config may specify a different prep dir; otherwise use
    *fallback*.
    """
    configured = cfg.get("metrics", {}).get("dataset", {}).get("prep_dir")
    if configured:
        return Path(configured)
    return fallback


def expected_metric_outputs(
    metric_paths: MetricPaths,
    atlas: str,
    selected_metrics: list[str],
) -> list[tuple[str, Path]]:
    """Return ``(label, path)`` pairs for all expected outputs across selected
    metric families."""
    expected: list[tuple[str, Path]] = []
    if "fc" in selected_metrics:
        expected.extend([
            ("timeseries_tsv", metric_paths.timeseries_output(atlas)),
            ("fc_correlation_tsv", metric_paths.fc_correlation_output(atlas)),
            ("fc_fisherz_tsv", metric_paths.fc_fisherz_output(atlas)),
            ("fc_long_tsv", metric_paths.fc_long_output(atlas)),
            ("fc_qc_tsv", metric_paths.qc_tsv_output(atlas)),
            ("fc_metadata_json", metric_paths.metadata_json_output(atlas)),
        ])
    for metric in ("alff", "falff", "reho"):
        if metric in selected_metrics:
            expected.append((f"{metric}_map", metric_paths.map_output(metric)))
            expected.append((f"{metric}_roi_tsv", metric_paths.roi_summary_output(atlas, metric)))
    regional_set = set(selected_metrics) & {"alff", "falff", "reho"}
    if regional_set:
        expected.append(("regional_qc_tsv", metric_paths.regional_qc_tsv_output()))
        expected.append(("regional_metadata_json", metric_paths.regional_metadata_json_output()))
    if "graph" in selected_metrics:
        from .graph_outputs import expected_graph_outputs

        expected.extend(expected_graph_outputs(metric_paths, atlas))
    if "gradient" in selected_metrics:
        from .gradient_outputs import expected_gradient_outputs

        expected.extend(expected_gradient_outputs(metric_paths, atlas))
    return expected


def _expected_output_paths(metric_paths: MetricPaths, atlas: str) -> list[tuple[str, Path]]:
    """Legacy wrapper for fc-only. Prefer :func:`expected_metric_outputs`."""
    return expected_metric_outputs(metric_paths, atlas, ["fc"])


# ---------------------------------------------------------------------------
# Per-run metrics execution (--run-missing mode)
# ---------------------------------------------------------------------------


def run_metrics_for_run(
    run: DatasetRun,
    atlas: AtlasSpec,
    cfg: dict[str, Any],
    dry_run: bool = False,
    force: bool = False,
) -> list[Path]:
    """Execute single-run metrics for one *run* / *atlas* pair.

    Returns the list of generated standardized output paths.
    """
    from .metrics_qc import write_metrics_qc
    from .netcorr import run_3dnetcorr
    from .netcorr_outputs import standardize_netcorr_outputs

    assert run.final_bold_path is not None
    run.metric_paths.ensure_dirs()

    result = run_3dnetcorr(
        bold_path=run.final_bold_path,
        atlas=atlas,
        metric_paths=run.metric_paths,
        cfg=cfg,
        dry_run=dry_run,
        force=force,
    )

    if dry_run:
        return [result.netcc_path, result.netts_path, result.log_path]

    standardized = standardize_netcorr_outputs(
        netts_path=result.netts_path,
        netcc_path=result.netcc_path,
        timeseries_tsv=run.metric_paths.timeseries_output(atlas.name),
        fc_correlation_tsv=run.metric_paths.fc_correlation_output(atlas.name),
        fc_fisherz_tsv=run.metric_paths.fc_fisherz_output(atlas.name),
        fc_long_tsv=run.metric_paths.fc_long_output(atlas.name),
        subject=run.subject,
        session=run.session,
        task=run.task,
        atlas=atlas.name,
        label_path=atlas.label_path,
    )
    qc_tsv, metadata_json = write_metrics_qc(run.metric_paths, atlas.name, result, standardized, cfg)
    return [
        standardized.timeseries_tsv,
        standardized.fc_correlation_tsv,
        standardized.fc_fisherz_tsv,
        standardized.fc_long_tsv,
        qc_tsv,
        metadata_json,
    ]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_fc_long(
    dataset_paths: DatasetMetricPaths,
    atlases: list[AtlasSpec],
    included_runs: list[DatasetRun],
) -> list[Path]:
    """Concatenate per-run ``*_fc-long.tsv`` files into dataset-level
    ``all_fc_long_atlas-<atlas>.tsv`` files.

    Returns the list of written output paths.
    """
    dataset_paths.ensure_dirs()
    written: list[Path] = []

    for atlas in atlases:
        output = dataset_paths.fc_long_tsv(atlas.name)
        _concatenate_tsv(
            output,
            [
                run.metric_paths.fc_long_output(atlas.name)
                for run in included_runs
                if run.metric_paths.fc_long_output(atlas.name).exists()
            ],
            sort_keys=("subject", "session", "task", "atlas", "roi_i", "roi_j"),
        )
        written.append(output)

    return written


def aggregate_metrics_qc(
    dataset_paths: DatasetMetricPaths,
    atlases: list[AtlasSpec],
    included_runs: list[DatasetRun],
) -> Path:
    """Concatenate per-run ``*_metrics-qc.tsv`` files into
    ``participants_metrics_qc.tsv``.
    """
    dataset_paths.ensure_dirs()
    output = dataset_paths.participants_metrics_qc_tsv
    _concatenate_tsv(
        output,
        [
            run.metric_paths.qc_tsv_output(atlas.name)
            for run in included_runs
            for atlas in atlases
            if run.metric_paths.qc_tsv_output(atlas.name).exists()
        ],
        sort_keys=("subject", "session", "task", "atlas"),
    )
    return output


def generate_connectivity_summary(
    dataset_paths: DatasetMetricPaths,
    atlases: list[AtlasSpec],
    included_runs: list[DatasetRun],
) -> Path:
    """Generate ``all_connectivity_summary.tsv`` with per-run/per-atlas
    edge-level summaries (n_edges, mean_z, sd_z, etc.).
    """
    dataset_paths.ensure_dirs()
    output = dataset_paths.connectivity_summary_tsv
    rows: list[dict[str, Any]] = []

    for run in included_runs:
        for atlas in atlases:
            long_path = run.metric_paths.fc_long_output(atlas.name)
            if not long_path.exists():
                continue
            z_values: list[float] = []
            n_edges = 0
            with long_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    n_edges += 1
                    try:
                        z_values.append(float(row["z"]))
                    except (ValueError, KeyError):
                        continue

            qc_path = run.metric_paths.qc_tsv_output(atlas.name)
            n_missing = 0
            n_constant = 0
            if qc_path.exists():
                with qc_path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle, delimiter="\t")
                    qc_row = next(reader, {})
                    n_missing = int(qc_row.get("n_missing_rois", 0))
                    n_constant = int(qc_row.get("n_constant_rois", 0))

            mean_z = sum(z_values) / len(z_values) if z_values else 0.0
            sd_z = (
                (sum((v - mean_z) ** 2 for v in z_values) / len(z_values)) ** 0.5
                if z_values
                else 0.0
            )

            rows.append(
                {
                    "subject": run.subject,
                    "session": run.session,
                    "task": run.task,
                    "atlas": atlas.name,
                    "n_edges": n_edges,
                    "mean_z": f"{mean_z:.6f}",
                    "sd_z": f"{sd_z:.6f}",
                    "n_missing_rois": n_missing,
                    "n_constant_rois": n_constant,
                    "mean_fd": _fmt_optional(run.mean_fd),
                    "fd_outlier_ratio": _fmt_optional(run.fd_outlier_ratio),
                }
            )

    if rows:
        _write_dict_tsv(output, list(rows[0].keys()), rows)
    else:
        _write_empty_with_headers(output, [
            "subject", "session", "task", "atlas", "n_edges",
            "mean_z", "sd_z", "n_missing_rois", "n_constant_rois",
            "mean_fd", "fd_outlier_ratio",
        ])
    return output


def generate_roi_timeseries_qc(
    dataset_paths: DatasetMetricPaths,
    atlases: list[AtlasSpec],
    included_runs: list[DatasetRun],
) -> Path:
    """Generate ``all_roi_timeseries_qc.tsv`` — initially one row per
    run/atlas with available ROI/time-series QC fields.
    """
    dataset_paths.ensure_dirs()
    output = dataset_paths.roi_timeseries_qc_tsv
    rows: list[dict[str, Any]] = []

    for run in included_runs:
        for atlas in atlases:
            qc_path = run.metric_paths.qc_tsv_output(atlas.name)
            if not qc_path.exists():
                continue
            with qc_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                qc_row = next(reader, {})
            rows.append(
                {
                    "subject": run.subject,
                    "session": run.session,
                    "task": run.task,
                    "atlas": atlas.name,
                    "n_timepoints": qc_row.get("n_timepoints", ""),
                    "n_rois": qc_row.get("n_rois", ""),
                    "n_missing_rois": qc_row.get("n_missing_rois", ""),
                    "n_constant_rois": qc_row.get("n_constant_rois", ""),
                    "mean_fd": _fmt_optional(run.mean_fd),
                    "fd_outlier_ratio": _fmt_optional(run.fd_outlier_ratio),
                }
            )

    if rows:
        _write_dict_tsv(output, list(rows[0].keys()), rows)
    else:
        _write_empty_with_headers(output, [
            "subject", "session", "task", "atlas", "n_timepoints",
            "n_rois", "n_missing_rois", "n_constant_rois",
            "mean_fd", "fd_outlier_ratio",
        ])
    return output


# ---------------------------------------------------------------------------
# Inclusion table
# ---------------------------------------------------------------------------


def write_inclusion_table(path: Path, runs: list[DatasetRun]) -> None:
    """Write ``participants_inclusion.tsv``."""
    fieldnames = [
        "subject",
        "session",
        "task",
        "status",
        "reason",
        "final_bold_path",
        "motion_summary_path",
        "mean_fd",
        "fd_outlier_ratio",
        "group_mask_path",
        "atlases_requested",
        "metrics_outputs_present",
        "metrics_outputs_missing",
    ]
    rows = [
        {
            "subject": run.subject,
            "session": run.session,
            "task": run.task,
            "status": run.status,
            "reason": run.reason,
            "final_bold_path": str(run.final_bold_path or ""),
            "motion_summary_path": str(run.motion_summary_path or ""),
            "mean_fd": _fmt_optional(run.mean_fd),
            "fd_outlier_ratio": _fmt_optional(run.fd_outlier_ratio),
            "group_mask_path": str(run.group_mask_path or ""),
            "atlases_requested": ",".join(run.atlases_requested),
            "metrics_outputs_present": ",".join(run.metrics_outputs_present),
            "metrics_outputs_missing": ",".join(run.metrics_outputs_missing),
        }
        for run in runs
    ]
    _write_dict_tsv(path, fieldnames, rows)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def write_manifest(
    path: Path,
    prep_dir: Path,
    cfg: dict[str, Any],
    atlases: list[AtlasSpec],
    tasks: list[str] | None,
    subjects: list[str] | None,
    sessions: list[str] | None,
    runs: list[DatasetRun],
    generated_paths: list[Path],
    command_args: dict[str, Any] | None = None,
) -> None:
    """Write ``metrics_dataset_manifest.json``."""
    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for run in runs:
        status_counts[run.status] = status_counts.get(run.status, 0) + 1
        if run.reason:
            reason_counts[run.reason] = reason_counts.get(run.reason, 0) + 1

    manifest: dict[str, Any] = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "prep_dir": str(prep_dir),
        "command_args": command_args or {},
        "config": {
            "metrics": cfg.get("metrics", {}),
        },
        "requested": {
            "atlases": [atlas.name for atlas in atlases],
            "tasks": tasks or [],
            "subjects": subjects or [],
            "sessions": sessions or [],
        },
        "runs": {
            "discovered": len(runs),
            "included": status_counts.get("included", 0),
            "excluded": status_counts.get("excluded", 0),
            "pending": status_counts.get("pending", 0),
            "by_reason": reason_counts,
        },
        "generated_files": [str(p) for p in generated_paths],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_dataset_metrics(
    prep_dir: Path,
    cfg: dict[str, Any],
    atlases: list[AtlasSpec],
    tasks: list[str] | None = None,
    subjects: list[str] | None = None,
    sessions: list[str] | None = None,
    selected_metrics: list[str] | None = None,
    run_missing: bool = False,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Orchestrate dataset-level metrics discovery, inclusion, and aggregation.

    Returns a dict with keys ``runs``, ``included_runs``, and
    ``generated_paths`` for use by the CLI.
    """
    dataset_cfg = cfg.get("metrics", {}).get("dataset", {})
    effective_run_missing = run_missing or dataset_cfg.get("run_missing", False)
    metrics_to_check = list(selected_metrics if selected_metrics else ["fc"])
    # 3dRSFC always computes ALFF and fALFF together; expand the check list.
    if "alff" in metrics_to_check and "falff" not in metrics_to_check:
        metrics_to_check.append("falff")
    if "falff" in metrics_to_check and "alff" not in metrics_to_check:
        metrics_to_check.append("alff")
    if "combat" in metrics_to_check and "fc" not in metrics_to_check:
        metrics_to_check.append("fc")

    # 1. Discover runs.
    all_runs = discover_runs(prep_dir, tasks=tasks, subjects=subjects, sessions=sessions)

    # 2. Evaluate inclusion.
    for run in all_runs:
        evaluate_run_inclusion(run, cfg, atlases, selected_metrics=metrics_to_check,
                               allow_missing_outputs=effective_run_missing)

    # Recompute included_runs after --run-missing may have changed statuses.
    included_runs = [run for run in all_runs if run.status == "included"]
    generated_paths: list[Path] = []

    if dry_run and effective_run_missing:
        for run in included_runs:
            if _any_outputs_missing(run, atlases, metrics_to_check, cfg):
                generated_paths.append(run.metric_paths.metrics_task_dir)

    # 4. Aggregate.
    dataset_paths = DatasetMetricPaths(prep_dir)
    dataset_paths.ensure_dirs()

    if included_runs and not dry_run:
        # 3.1 FC/Regional missing per-run metrics.
        if effective_run_missing:
            base_metrics = [m for m in metrics_to_check if m in {"fc", "alff", "falff", "reho"}]
            if base_metrics:
                for run in included_runs:
                    try:
                        _run_missing_for_families(run, atlases, base_metrics, cfg, force)
                    except Exception as e:
                        import logging
                        logging.exception(f"Failed to run base missing metrics for {run.subject} {run.session} {run.task}: {e}")
                        run.status = "excluded"
                        run.reason = "metrics_run_failed"

        # FC aggregation.
        if "fc" in metrics_to_check:
            fc_paths = aggregate_fc_long(dataset_paths, atlases, included_runs)
            generated_paths.extend(fc_paths)

            qc_path = aggregate_metrics_qc(dataset_paths, atlases, included_runs)
            generated_paths.append(qc_path)

            summary_path = generate_connectivity_summary(dataset_paths, atlases, included_runs)
            generated_paths.append(summary_path)

            rts_qc_path = generate_roi_timeseries_qc(dataset_paths, atlases, included_runs)
            generated_paths.append(rts_qc_path)

        # Regional aggregation.
        regional_set = set(metrics_to_check) & {"alff", "falff", "reho"}
        if regional_set:
            regional_long = _aggregate_regional_long(dataset_paths, atlases, list(regional_set), included_runs)
            if regional_long:
                generated_paths.append(regional_long)

            regional_qc = _aggregate_regional_qc(dataset_paths, included_runs)
            if regional_qc:
                generated_paths.append(regional_qc)

        # 3.2 ComBat harmonization
        if "combat" in metrics_to_check:
            combat_cfg = cfg.get("metrics", {}).get("combat", {})
            if not combat_cfg.get("enabled", False):
                raise RuntimeError("Selected metric 'combat' requires metrics.combat.enabled: true.")
            from .fc_combat import run_fc_combat

            generated_paths.extend(
                run_fc_combat(
                    prep_dir=prep_dir,
                    cfg=cfg,
                    included_runs=included_runs,
                    atlas_names=[atlas.name for atlas in atlases],
                )
            )

        # 3.3 Graph/Gradient missing per-run metrics.
        if effective_run_missing:
            derived_metrics = [m for m in metrics_to_check if m in {"graph", "gradient"}]
            if derived_metrics:
                for run in included_runs:
                    if run.status != "included":
                        continue
                    try:
                        _run_missing_for_families(run, atlases, derived_metrics, cfg, force)
                    except Exception as e:
                        import logging
                        logging.exception(f"Failed to run derived missing metrics for {run.subject} {run.session} {run.task}: {e}")
                        run.status = "excluded"
                        run.reason = "metrics_run_failed"

        # 3.4 Re-check all expected outputs if run-missing happened.
        if effective_run_missing:
            for run in included_runs:
                if run.status != "included":
                    continue
                _refresh_output_status(run, atlases, metrics_to_check, cfg)
                if run.status == "included" and run.metrics_outputs_missing:
                    run.status = "excluded"
                    run.reason = "missing_metrics_outputs_after_run"
            included_runs = [run for run in all_runs if run.status == "included"]

        # Graph aggregation.
        if "graph" in metrics_to_check:
            from .graph_outputs import (
                aggregate_graph_summary,
                aggregate_graph_thresholds,
                check_graph_method_consistency,
            )

            issues = check_graph_method_consistency(included_runs, atlases)
            if issues:
                raise RuntimeError(
                    "Inconsistent graph methods across runs. "
                    "Cannot pool graph metrics with different backends, community "
                    "methods, densities, random-network counts, or seed strategies.\n"
                    + "\n".join(issues)
                )

            graph_sum = aggregate_graph_summary(dataset_paths, included_runs, atlases)
            if graph_sum:
                generated_paths.append(graph_sum)

            graph_thr = aggregate_graph_thresholds(dataset_paths, included_runs, atlases)
            if graph_thr:
                generated_paths.append(graph_thr)

        # Gradient aggregation and alignment.
        if "gradient" in metrics_to_check:
            from .gradient_dataset import run_dataset_gradients

            gradient_paths = run_dataset_gradients(prep_dir, cfg, atlases, included_runs)
            generated_paths.extend(gradient_paths)

    inclusion_path = dataset_paths.inclusion_tsv
    write_inclusion_table(inclusion_path, all_runs)
    generated_paths.append(inclusion_path)

    manifest_path = dataset_paths.manifest_json
    write_manifest(
        manifest_path,
        prep_dir,
        cfg,
        atlases,
        tasks,
        subjects,
        sessions,
        all_runs,
        generated_paths,
        command_args={
            "tasks": tasks,
            "subjects": subjects,
            "sessions": sessions,
            "run_missing": run_missing,
            "dry_run": dry_run,
            "force": force,
        },
    )
    generated_paths.append(manifest_path)

    return {
        "runs": all_runs,
        "included_runs": included_runs,
        "generated_paths": generated_paths,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _missing_atlas_names_from_outputs(run: DatasetRun, atlases: list[AtlasSpec]) -> list[str]:
    """Return atlas names for which any standardized output is missing."""
    missing: list[str] = []
    for atlas in atlases:
        expected = _expected_output_paths(run.metric_paths, atlas.name)
        if any(not path.exists() for _, path in expected):
            missing.append(atlas.name)
    return missing


def _any_outputs_missing(run: DatasetRun, atlases: list[AtlasSpec], selected_metrics: list[str], cfg: dict[str, Any]) -> bool:
    """Return True if any expected metric output is missing for *run*."""
    metrics_to_check = [metric for metric in selected_metrics if metric != "gradient"]
    for atlas in atlases:
        expected = expected_metric_outputs(run.metric_paths, atlas.name, metrics_to_check)
        if any(not p.exists() for _, p in expected):
            return True
        if "gradient" in selected_metrics:
            from .gradient_outputs import check_gradient_fc_prerequisites

            if check_gradient_fc_prerequisites(run.metric_paths, atlas.name, cfg=cfg):
                return True
    return False


def _refresh_output_status(run: DatasetRun, atlases: list[AtlasSpec], selected_metrics: list[str], cfg: dict[str, Any]) -> None:
    """Recompute ``metrics_outputs_present`` / ``metrics_outputs_missing``."""
    metrics_to_check = [metric for metric in selected_metrics if metric != "gradient"]
    present: list[str] = []
    missing: list[str] = []
    for atlas in atlases:
        for label, p in expected_metric_outputs(run.metric_paths, atlas.name, metrics_to_check):
            if p.exists():
                present.append(label)
            else:
                missing.append(label)
        if "gradient" in selected_metrics:
            from .gradient_outputs import check_gradient_fc_prerequisites

            gradient_missing = check_gradient_fc_prerequisites(run.metric_paths, atlas.name, cfg)
            if gradient_missing:
                missing.extend(f"gradient_fc_prerequisite:{item}" for item in gradient_missing)
            else:
                present.append(f"gradient_fc_prerequisites:{atlas.name}")
    run.metrics_outputs_present = present
    run.metrics_outputs_missing = missing


def _run_missing_for_families(
    run: DatasetRun,
    atlases: list[AtlasSpec],
    selected_metrics: list[str],
    cfg: dict[str, Any],
    force: bool,
) -> None:
    """Run missing per-run metrics for *run*, dispatching by metric family."""
    from .metrics_inputs import resolve_final_mni_bold

    bold_input = resolve_final_mni_bold(run.prep_paths, cfg, smoothed=False)

    # ── FC ──
    if "fc" in selected_metrics:
        for atlas in atlases:
            expected = _expected_output_paths(run.metric_paths, atlas.name)
            if all(p.exists() for _, p in expected):
                continue
            run_metrics_for_run(run, atlas, cfg, dry_run=False, force=force)

    # ── Regional ──
    regional_set = set(selected_metrics) & {"alff", "falff", "reho"}
    if regional_set:
        # Check all expected regional outputs (maps, ROI TSVs, QC, metadata).
        any_missing = False
        for atlas in atlases:
            for label, p in expected_metric_outputs(run.metric_paths, atlas.name,
                                                    list(regional_set)):
                if not p.exists():
                    any_missing = True
                    break
            if any_missing:
                break
        if any_missing:
            from .regional_metrics import run_regional_metrics

            run_regional_metrics(
                bold_path=bold_input.path,
                metric_paths=run.metric_paths,
                atlases=atlases,
                selected_metrics=selected_metrics,
                cfg=cfg,
                dry_run=False,
                force=force,
            )

    # ── Graph ──
    if "graph" in selected_metrics:
        from .graph_metrics import run_graph_metrics

        run_graph_metrics(
            metric_paths=run.metric_paths,
            atlases=atlases,
            cfg=cfg,
            dry_run=False,
            force=force,
        )


def _aggregate_regional_long(
    dataset_paths: DatasetMetricPaths,
    atlases: list[AtlasSpec],
    regional_metrics: list[str],
    included_runs: list[DatasetRun],
) -> Path | None:
    """Concatenate per-atlas/per-metric ROI TSVs into
    ``all_regional_metrics_long.tsv``."""
    path = dataset_paths.derivatives_metrics_dir / "all_regional_metrics_long.tsv"
    inputs: list[Path] = []
    for run in included_runs:
        for atlas in atlases:
            for metric in regional_metrics:
                p = run.metric_paths.roi_summary_output(atlas.name, metric)
                if p.exists():
                    inputs.append(p)
    if inputs:
        _concatenate_tsv(
            path, inputs,
            sort_keys=("subject", "session", "task", "atlas", "metric", "roi_id"),
        )
        return path
    _write_empty_with_headers(path, [
        "subject", "session", "task", "atlas", "metric", "roi_id",
        "roi_name", "mean", "sd", "n_voxels_valid", "n_voxels_atlas", "coverage",
    ])
    return path


def _aggregate_regional_qc(
    dataset_paths: DatasetMetricPaths,
    included_runs: list[DatasetRun],
) -> Path | None:
    """Concatenate per-run ``regional_metrics_qc.tsv`` into
    ``all_regional_metrics_qc.tsv``."""
    path = dataset_paths.derivatives_metrics_dir / "all_regional_metrics_qc.tsv"
    inputs: list[Path] = []
    for run in included_runs:
        p = run.metric_paths.regional_qc_tsv_output()
        if p.exists():
            inputs.append(p)
    if inputs:
        _concatenate_tsv(path, inputs, sort_keys=("subject", "session", "task"))
        return path
    return None


def _concatenate_tsv(
    output: Path,
    input_paths: list[Path],
    sort_keys: tuple[str, ...] = (),
) -> None:
    """Concatenate TSV files with headers, preserving the header only once."""
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None

    for path in input_paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            for row in reader:
                rows.append(dict(row))

    if fieldnames is None:
        return

    if sort_keys:
        rows.sort(key=lambda r: tuple(r.get(k, "") for k in sort_keys))

    _write_dict_tsv(output, fieldnames, rows)


def _write_dict_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_empty_with_headers(path: Path, fieldnames: list[str]) -> None:
    """Write a TSV with headers only (no data rows)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()


def _fmt_optional(value: Any) -> str:
    if value is None:
        return ""
    return str(value)
