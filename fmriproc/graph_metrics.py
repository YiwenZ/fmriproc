"""Single-run graph-metrics orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .atlas_registry import AtlasSpec
from .paths import MetricPaths


def run_graph_metrics(
    metric_paths: MetricPaths,
    atlases: list[AtlasSpec],
    cfg: dict[str, Any],
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, list[Path]]:
    """Execute graph metrics for one subject-session-task.

    Returns ``{"output_paths": [...]}``.
    """
    from .graph_outputs import check_fc_prerequisites

    metric_paths.ensure_dirs()
    generated: list[Path] = []

    for atlas in atlases:
        # 1. Check FC prerequisites.
        missing = check_fc_prerequisites(metric_paths, atlas.name, cfg)
        if missing:
            raise RuntimeError(
                f"Graph metrics require complete FC outputs for atlas {atlas.name}. "
                f"Missing: {'; '.join(missing)}. "
                f"Run 'fmriproc features --metrics fc' first."
            )

        thresholds_path = metric_paths.graph_thresholds_output(atlas.name)
        summary_path = metric_paths.graph_summary_output(atlas.name)
        meta_path = metric_paths.graph_metadata_json_output(atlas.name)
        raw_dir = metric_paths.graph_raw_dir()

        if dry_run:
            generated.extend([
                thresholds_path, summary_path, meta_path,
                metric_paths.graph_raw_log(atlas.name),
            ])
            continue

        from .fc_input_resolver import fc_variant_from_config
        
        current_fc_variant = fc_variant_from_config(cfg)
        can_skip = False
        if thresholds_path.exists() and summary_path.exists() and meta_path.exists() and not force:
            try:
                import json
                existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if existing_meta.get("fc_variant", "raw") == current_fc_variant:
                    can_skip = True
            except Exception:
                pass

        if can_skip:
            generated.extend([thresholds_path, summary_path, meta_path])
            continue

        from .graph_backend_igraph import compute_graph_metrics

        from .fc_input_resolver import resolve_fc_input_paths

        fc_input = resolve_fc_input_paths(metric_paths, atlas.name, cfg)
        fc_fisherz = fc_input.matrix_path
        fc_meta = fc_input.metadata_path

        log_path = metric_paths.graph_raw_log(atlas.name)

        compute_graph_metrics(
            fc_fisherz_path=fc_fisherz,
            fc_metadata_json_path=fc_meta,
            atlas=atlas.name,
            subject=metric_paths.prep_paths.subject,
            session=metric_paths.prep_paths.session,
            task=metric_paths.prep_paths.task,
            thresholds_path=thresholds_path,
            summary_path=summary_path,
            metadata_out_path=meta_path,
            raw_dir=raw_dir,
            log_path=log_path,
            cfg=cfg,
        )
        generated.extend([
            thresholds_path, summary_path, meta_path,
            metric_paths.graph_raw_log(atlas.name),
        ])

    return {"output_paths": generated}
