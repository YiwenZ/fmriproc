"""Graph-metric output helpers: prerequisite checks, expected-output lists,
and dataset-level aggregation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .paths import DatasetMetricPaths, MetricPaths


def check_fc_prerequisites(metric_paths: MetricPaths, atlas: str, cfg: dict | None = None) -> list[str]:
    """Verify that FC outputs required by graph metrics exist.

    Returns a list of missing-item descriptions (empty = all present).
    """
    from .fc_input_resolver import check_fc_input_prerequisites

    return check_fc_input_prerequisites(metric_paths, atlas, cfg or {})


def expected_graph_outputs(metric_paths: MetricPaths, atlas: str) -> list[tuple[str, Path]]:
    """Return ``(label, path)`` pairs for expected graph outputs."""
    return [
        ("graph_thresholds_tsv", metric_paths.graph_thresholds_output(atlas)),
        ("graph_summary_tsv", metric_paths.graph_summary_output(atlas)),
        ("graph_metadata_json", metric_paths.graph_metadata_json_output(atlas)),
    ]


# ---------------------------------------------------------------------------
# Dataset aggregation
# ---------------------------------------------------------------------------


def aggregate_graph_summary(
    dataset_paths: DatasetMetricPaths,
    included_runs: list,
    atlases: list,
) -> Path | None:
    """Concatenate per-run graph summary TSVs into
    ``all_graph_metrics_summary.tsv``."""
    path = dataset_paths.derivatives_metrics_dir / "all_graph_metrics_summary.tsv"
    inputs = [
        run.metric_paths.graph_summary_output(atlas.name)
        for run in included_runs
        for atlas in atlases
        if run.metric_paths.graph_summary_output(atlas.name).exists()
    ]
    if inputs:
        _concatenate_tsv(path, inputs, sort_keys=("subject", "session", "task", "atlas"))
        return path
    return None


def aggregate_graph_thresholds(
    dataset_paths: DatasetMetricPaths,
    included_runs: list,
    atlases: list,
) -> Path | None:
    """Concatenate per-run graph threshold TSVs into
    ``all_graph_metrics_thresholds.tsv``."""
    path = dataset_paths.derivatives_metrics_dir / "all_graph_metrics_thresholds.tsv"
    inputs = [
        run.metric_paths.graph_thresholds_output(atlas.name)
        for run in included_runs
        for atlas in atlases
        if run.metric_paths.graph_thresholds_output(atlas.name).exists()
    ]
    if inputs:
        _concatenate_tsv(path, inputs,
                         sort_keys=("subject", "session", "task", "atlas", "density"))
        return path
    return None


def check_graph_method_consistency(included_runs: list, atlases: list) -> list[str]:
    """Verify that all included runs share the same graph method parameters.

    Returns a list of inconsistency descriptions (empty = all consistent).
    """
    ref: dict | None = None
    issues: list[str] = []
    for run in included_runs:
        for atlas in atlases:
            meta_path = run.metric_paths.graph_metadata_json_output(atlas.name)
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            current = {
                "backend": meta.get("backend"),
                "community_method": meta.get("community_method"),
                "densities": tuple(meta.get("densities", [])),
                "random_networks": meta.get("random_networks"),
                "seed_strategy": meta.get("seed_strategy"),
            }
            if ref is None:
                ref = current
            elif current != ref:
                issues.append(
                    f"Method mismatch for {run.subject}/{run.session}/task-{run.task}/{atlas.name}: "
                    f"got {current}, expected {ref}"
                )
    return issues


def _concatenate_tsv(output: Path, input_paths: list[Path], sort_keys: tuple[str, ...] = ()) -> None:
    """Concatenate TSV files with a single header."""
    from .metrics_dataset import _write_empty_with_headers

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] | None = None
    rows: list[dict[str, str]] = []
    for path in input_paths:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            for row in reader:
                rows.append(dict(row))
    if fieldnames is None:
        return
    if sort_keys:
        rows.sort(key=lambda r: tuple(r.get(k, "") for k in sort_keys))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
