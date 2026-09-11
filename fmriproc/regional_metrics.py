"""Single-run regional metrics orchestrator.

Dispatches selected metric families (``alff``, ``falff``, ``reho``) to the
appropriate AFNI runners, then runs z-scoring and ROI summaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .atlas_registry import AtlasSpec
from .paths import MetricPaths


def run_regional_metrics(
    bold_path: Path,
    metric_paths: MetricPaths,
    atlases: list[AtlasSpec],
    selected_metrics: list[str],
    cfg: dict[str, Any],
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, list[Path]]:
    """Execute regional metrics for a single subject/session/task.

    Returns a dict mapping ``"output_paths"`` → list of generated paths.
    """
    from .metrics_inputs import resolve_group_mask_for_metrics
    from .regional_afni import run_3dreho, run_3drsfc

    metric_paths.ensure_dirs()
    generated: list[Path] = []

    # Resolve group mask.
    group_mask = resolve_group_mask_for_metrics(
        metric_paths, cfg, bold_path, dry_run=dry_run
    )

    rsfc_result = None
    reho_result = None
    roi_paths: dict[tuple[str, str], Path] = {}

    regional_metrics_set = set(selected_metrics) & {"alff", "falff", "reho"}
    # 3dRSFC always computes ALFF and fALFF together; if either is selected,
    # both maps exist and should get ROI summaries.
    if regional_metrics_set & {"alff", "falff"}:
        regional_metrics_set.update({"alff", "falff"})
    do_rsfc = bool({"alff", "falff"} & regional_metrics_set)
    do_reho = "reho" in regional_metrics_set

    # --- 3dRSFC (ALFF / fALFF) ---
    if do_rsfc and not dry_run:
        rsfc_result = run_3drsfc(
            bold_path=bold_path,
            metric_paths=metric_paths,
            cfg=cfg,
            group_mask_path=group_mask,
            dry_run=dry_run,
            force=force,
        )
        generated.extend([rsfc_result.alff_map, rsfc_result.falff_map])
        generated.append(rsfc_result.log_path)

        # Z-score
        if group_mask is not None and cfg.get("metrics", {}).get("regional", {}).get("standardize", {}).get("zscore_within_mask", True):
            from .regional_maps import zscore_within_mask

            for metric, map_path in [("alff", rsfc_result.alff_map), ("falff", rsfc_result.falff_map)]:
                z_path = metric_paths.z_map_output(metric)
                zscore_within_mask(map_path, group_mask, z_path)
                generated.append(z_path)

    elif do_rsfc and dry_run:
        generated.append(metric_paths.map_output("alff"))
        generated.append(metric_paths.map_output("falff"))

    # --- 3dReHo ---
    if do_reho and not dry_run:
        reho_result = run_3dreho(
            bold_path=bold_path,
            metric_paths=metric_paths,
            cfg=cfg,
            group_mask_path=group_mask,
            dry_run=dry_run,
            force=force,
        )
        generated.append(reho_result.reho_map)
        generated.append(reho_result.log_path)

        # Z-score
        if group_mask is not None and cfg.get("metrics", {}).get("regional", {}).get("standardize", {}).get("zscore_within_mask", True):
            from .regional_maps import zscore_within_mask

            z_path = metric_paths.z_map_output("reho")
            zscore_within_mask(reho_result.reho_map, group_mask, z_path)
            generated.append(z_path)

    elif do_reho and dry_run:
        generated.append(metric_paths.map_output("reho"))

    # --- ROI summaries ---
    min_coverage = float(cfg.get("metrics", {}).get("regional", {}).get("roi_summary", {}).get("min_roi_coverage", 0.5))

    for metric in regional_metrics_set:
        map_path = metric_paths.map_output(metric)
        if not map_path.exists():
            continue
        from .regional_roi import compute_roi_summary, write_roi_summary_tsv

        for atlas in atlases:
            rows = compute_roi_summary(
                map_path=map_path,
                atlas=atlas,
                metric=metric,
                subject=metric_paths.prep_paths.subject,
                session=metric_paths.prep_paths.session,
                task=metric_paths.prep_paths.task,
                group_mask_path=group_mask,
                min_roi_coverage=min_coverage,
            )
            roi_path = metric_paths.roi_summary_output(atlas.name, metric)
            write_roi_summary_tsv(roi_path, rows)
            roi_paths[(atlas.name, metric)] = roi_path
            generated.append(roi_path)

    # --- QC / metadata ---
    from .regional_qc import write_regional_qc

    atlas_names = [a.name for a in atlases]
    qc_path, meta_path = write_regional_qc(
        metric_paths=metric_paths,
        requested_metrics=selected_metrics,
        atlas_names=atlas_names,
        bold_path=bold_path,
        group_mask_path=group_mask,
        rsfc_result=rsfc_result,
        reho_result=reho_result,
        roi_paths=roi_paths,
        cfg=cfg,
    )
    generated.extend([qc_path, meta_path])

    return {"output_paths": generated}
