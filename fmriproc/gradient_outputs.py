"""Gradient-metric output helpers and condition-group utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .paths import DatasetMetricPaths, MetricPaths


def condition_group_for_task(task: str, cfg: dict[str, Any]) -> str:
    """Map a BIDS task name to an analysis condition group."""
    ref = cfg.get("metrics", {}).get("gradient", {}).get("reference", {})
    groups = ref.get("condition_groups", {}) or {}
    for group, tasks in groups.items():
        if task in tasks:
            return str(group)
    policy = ref.get("unmapped_task_policy", "fail")
    if policy == "own_group":
        return task
    raise ValueError(f"Gradient task {task!r} is not mapped to any condition group.")


def check_gradient_fc_prerequisites(metric_paths: MetricPaths, atlas: str, cfg: dict[str, Any] | None = None, force_raw: bool = False) -> list[str]:
    """Return missing FC artifacts required by gradient metrics."""
    from .fc_input_resolver import check_fc_input_prerequisites

    return check_fc_input_prerequisites(metric_paths, atlas, cfg or {}, force_raw=force_raw)


def expected_gradient_outputs(metric_paths: MetricPaths, atlas: str) -> list[tuple[str, Path]]:
    """Return expected run-level gradient outputs."""
    return [
        ("gradient_roi_tsv", metric_paths.gradient_roi_output(atlas)),
        ("gradient_summary_tsv", metric_paths.gradient_run_summary_output(atlas)),
        ("gradient_network_tsv", metric_paths.gradient_network_output(atlas)),
        ("gradient_metadata_json", metric_paths.gradient_metadata_json_output(atlas)),
    ]


def check_gradient_method_consistency(included_runs: list, atlases: list) -> list[str]:
    """Verify that run-level gradient metadata use the same method settings."""
    ref: dict[str, Any] | None = None
    issues: list[str] = []
    keys = ("backend", "n_components", "approach", "kernel", "alignment")
    for run in included_runs:
        for atlas in atlases:
            meta_path = run.metric_paths.gradient_metadata_json_output(atlas.name)
            if not meta_path.exists():
                issues.append(
                    f"missing gradient metadata for "
                    f"{run.subject}/{run.session}/task-{run.task}/{atlas.name}: {meta_path}"
                )
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            current = {key: meta.get(key) for key in keys}
            if ref is None:
                ref = current
            elif current != ref:
                differing = sorted(key for key in keys if current.get(key) != ref.get(key))
                issues.append(
                    f"Gradient method mismatch for "
                    f"{run.subject}/{run.session}/task-{run.task}/{atlas.name}: "
                    f"{', '.join(differing)} got {current}, expected {ref}"
                )
    return issues


def concatenate_tsv(output: Path, inputs: list[Path], sort_keys: tuple[str, ...] = ()) -> Path | None:
    """Concatenate TSV files with one header."""
    if not inputs:
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] | None = None
    rows: list[dict[str, str]] = []
    for path in inputs:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            for row in reader:
                rows.append(dict(row))
    if fieldnames is None:
        return None
    if sort_keys:
        rows.sort(key=lambda row: tuple(row.get(key, "") for key in sort_keys))
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return output


def aggregate_gradient_outputs(dataset_paths: DatasetMetricPaths, runs: list, atlases: list) -> list[Path]:
    """Aggregate run-level gradient TSVs into dataset-level files."""
    dataset_paths.ensure_dirs()
    generated: list[Path] = []
    specs = [
        (
            dataset_paths.gradient_roi_tsv,
            [run.metric_paths.gradient_roi_output(atlas.name) for run in runs for atlas in atlases],
            ("subject", "session", "task", "atlas", "roi_id", "gradient"),
        ),
        (
            dataset_paths.gradient_summary_tsv,
            [run.metric_paths.gradient_run_summary_output(atlas.name) for run in runs for atlas in atlases],
            ("subject", "session", "task", "atlas", "gradient"),
        ),
        (
            dataset_paths.gradient_network_tsv,
            [run.metric_paths.gradient_network_output(atlas.name) for run in runs for atlas in atlases],
            ("subject", "session", "task", "atlas", "network", "gradient"),
        ),
    ]
    for output, candidates, sort_keys in specs:
        existing = [path for path in candidates if path.exists()]
        written = concatenate_tsv(output, existing, sort_keys=sort_keys)
        if written:
            generated.append(written)
    return generated
