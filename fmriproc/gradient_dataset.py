"""Dataset-level FC gradient metrics.

This module builds condition-group reference gradients from included FC
matrices, aligns each run to its reference, and writes run-level plus
dataset-level gradient outputs.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .atlas_registry import AtlasSpec
from .gradient_backend import align_to_reference, fit_gradient
from .gradient_outputs import (
    aggregate_gradient_outputs,
    check_gradient_fc_prerequisites,
    check_gradient_method_consistency,
    condition_group_for_task,
)
from .paths import DatasetMetricPaths


@dataclass(frozen=True)
class _RunMatrix:
    """Wrapper to hold an individual run's FC matrix for gradient decomposition."""
    run: Any
    atlas: AtlasSpec
    condition_group: str
    labels: list[str]
    matrix: np.ndarray
    fc_source: Path
    fc_metadata_source: Path


def run_dataset_gradients(
    prep_dir: Path,
    cfg: dict[str, Any],
    atlases: list[AtlasSpec],
    included_runs: list,
) -> list[Path]:
    """Compute dataset-level gradient metrics for *included_runs*.

    FC Fisher-z matrices must already exist. Missing FC inputs are reported as
    an error instead of being generated here, because gradients are a dependent
    dataset-level metric family.
    """
    dataset_paths = DatasetMetricPaths(prep_dir)
    dataset_paths.ensure_dirs()

    grouped: dict[tuple[str, str], list[_RunMatrix]] = defaultdict(list)
    missing_inputs: list[str] = []
    for run in included_runs:
        condition_group = condition_group_for_task(run.task, cfg)
        for atlas in atlases:
            missing = check_gradient_fc_prerequisites(run.metric_paths, atlas.name, cfg)
            if missing:
                missing_inputs.extend(
                    f"{run.subject}/{run.session}/task-{run.task}/{atlas.name}: {item}"
                    for item in missing
                )
                continue
            from .fc_input_resolver import resolve_fc_input_paths

            fc_input = resolve_fc_input_paths(run.metric_paths, atlas.name, cfg)
            labels, matrix = _read_matrix_tsv(fc_input.matrix_path)
            grouped[(atlas.name, condition_group)].append(
                _RunMatrix(
                    run=run,
                    atlas=atlas,
                    condition_group=condition_group,
                    labels=labels,
                    matrix=matrix,
                    fc_source=fc_input.matrix_path,
                    fc_metadata_source=fc_input.metadata_path,
                )
            )

    if missing_inputs:
        raise RuntimeError(
            "FC has not been generated correctly for gradient metrics. "
            "Please check FC outputs before running gradient.\n"
            + "\n".join(missing_inputs)
        )

    if not grouped:
        return []

    gcfg = cfg.get("metrics", {}).get("gradient", {})
    ref_cfg = gcfg.get("reference", {})
    min_runs = int(ref_cfg.get("min_runs_per_group", 2))
    n_components = int(gcfg.get("n_components", 3))

    reference_rows: list[dict[str, Any]] = []
    generated_run_outputs: list[Path] = []

    for (atlas_name, condition_group), items in sorted(grouped.items()):
        if len(items) < min_runs:
            raise RuntimeError(
                f"Gradient reference group {condition_group!r} for atlas {atlas_name!r} "
                f"has {len(items)} run(s), fewer than min_runs_per_group={min_runs}."
            )
        _validate_compatible_matrices(items)
        reference_matrix = items[0].matrix.astype(np.float64, copy=True)
        for item in items[1:]:
            reference_matrix += item.matrix
        reference_matrix /= len(items)
        reference = fit_gradient(reference_matrix, cfg)
        reference_gradients = np.asarray(reference.gradients, dtype=float)[:, :n_components]
        reference_lambdas = _as_1d(reference.lambdas)
        reference_lambda_sum = _positive_sum(reference_lambdas)

        for component in range(reference_gradients.shape[1]):
            reference_rows.append(
                {
                    "atlas": atlas_name,
                    "condition_group": condition_group,
                    "reference_group": condition_group,
                    "reference_n_runs": len(items),
                    "gradient": component + 1,
                    "eigenvalue": _fmt(_value_at(reference_lambdas, component)),
                    "explained_variance_ratio": _fmt(_ratio(_value_at(reference_lambdas, component), reference_lambda_sum)),
                    "n_rois": reference_gradients.shape[0],
                }
            )

        for item in items:
            aligned = align_to_reference(item.matrix, reference_gradients, cfg)
            gradients = np.asarray(aligned.gradients, dtype=float)[:, :reference_gradients.shape[1]]
            lambdas = _as_1d(aligned.lambdas)

            _write_run_outputs(
                item=item,
                gradients=gradients,
                lambdas=lambdas,
                reference_gradients=reference_gradients,
                reference_group=condition_group,
                reference_n_runs=len(items),
                cfg=cfg,
            )
            generated_run_outputs.extend(
                [
                    item.run.metric_paths.gradient_roi_output(atlas_name),
                    item.run.metric_paths.gradient_run_summary_output(atlas_name),
                    item.run.metric_paths.gradient_network_output(atlas_name),
                    item.run.metric_paths.gradient_metadata_json_output(atlas_name),
                ]
            )

    consistency_issues = check_gradient_method_consistency(included_runs, atlases)
    if consistency_issues:
        raise RuntimeError(
            "Inconsistent gradient methods across runs. "
            "Cannot pool gradient metrics with different backend, component, "
            "approach, kernel, or alignment settings.\n"
            + "\n".join(consistency_issues)
        )

    generated = aggregate_gradient_outputs(dataset_paths, included_runs, atlases)
    _write_tsv(dataset_paths.gradient_reference_tsv, reference_rows, [
        "atlas",
        "condition_group",
        "reference_group",
        "reference_n_runs",
        "gradient",
        "eigenvalue",
        "explained_variance_ratio",
        "n_rois",
    ])
    generated.append(dataset_paths.gradient_reference_tsv)
    generated.extend(path for path in generated_run_outputs if path.exists())
    return generated


def _read_matrix_tsv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty FC matrix TSV: {path}") from exc
        labels = header[1:]
        rows: list[list[float]] = []
        row_labels: list[str] = []
        for row in reader:
            if not row:
                continue
            row_labels.append(row[0])
            rows.append([float(value) for value in row[1:]])
    matrix = np.asarray(rows, dtype=float)
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError(f"FC matrix is not square or does not match header: {path}")
    if row_labels and row_labels != labels:
        labels = row_labels
    return labels, matrix


def _validate_compatible_matrices(items: list[_RunMatrix]) -> None:
    first = items[0]
    for item in items[1:]:
        if item.labels != first.labels:
            raise ValueError(
                f"ROI label mismatch in gradient group {item.condition_group!r} "
                f"for atlas {item.atlas.name!r}."
            )
        if item.matrix.shape != first.matrix.shape:
            raise ValueError(
                f"FC shape mismatch in gradient group {item.condition_group!r} "
                f"for atlas {item.atlas.name!r}."
            )


def _write_run_outputs(
    item: _RunMatrix,
    gradients: np.ndarray,
    lambdas: np.ndarray,
    reference_gradients: np.ndarray,
    reference_group: str,
    reference_n_runs: int,
    cfg: dict[str, Any],
) -> None:
    run = item.run
    atlas = item.atlas.name
    roi_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    network_rows: list[dict[str, Any]] = []

    networks = [_network_from_roi_name(label) for label in item.labels]
    lambda_sum = _positive_sum(lambdas)

    for component in range(gradients.shape[1]):
        values = gradients[:, component]
        reference_values = reference_gradients[:, component]
        gradient_index = component + 1
        for roi_idx, (label, value) in enumerate(zip(item.labels, values), start=1):
            roi_rows.append(
                _base_row(run, atlas, item.condition_group)
                | {
                    "reference_group": reference_group,
                    "roi_id": roi_idx,
                    "roi_name": label,
                    "network": networks[roi_idx - 1],
                    "gradient": gradient_index,
                    "gradient_value": _fmt(value),
                }
            )

        summary_rows.append(
            _base_row(run, atlas, item.condition_group)
            | {
                "reference_group": reference_group,
                "reference_n_runs": reference_n_runs,
                "gradient": gradient_index,
                "eigenvalue": _fmt(_value_at(lambdas, component)),
                "explained_variance_ratio": _fmt(_ratio(_value_at(lambdas, component), lambda_sum)),
                "gradient_mean": _fmt(float(np.nanmean(values))),
                "gradient_sd": _fmt(float(np.nanstd(values))),
                "gradient_range": _fmt(float(np.nanmax(values) - np.nanmin(values))),
                "gradient_similarity_to_reference": _fmt(_corr(values, reference_values)),
                "gradient_distance_to_reference": _fmt(float(np.linalg.norm(values - reference_values))),
            }
        )

        network_rows.extend(
            _network_rows(
                run=run,
                atlas=atlas,
                condition_group=item.condition_group,
                reference_group=reference_group,
                gradient_index=gradient_index,
                values=values,
                networks=networks,
                cfg=cfg,
            )
        )

    _write_tsv(
        run.metric_paths.gradient_roi_output(atlas),
        roi_rows,
        [
            "subject",
            "session",
            "task",
            "condition_group",
            "atlas",
            "reference_group",
            "roi_id",
            "roi_name",
            "network",
            "gradient",
            "gradient_value",
        ],
    )
    _write_tsv(
        run.metric_paths.gradient_run_summary_output(atlas),
        summary_rows,
        [
            "subject",
            "session",
            "task",
            "condition_group",
            "atlas",
            "reference_group",
            "reference_n_runs",
            "gradient",
            "eigenvalue",
            "explained_variance_ratio",
            "gradient_mean",
            "gradient_sd",
            "gradient_range",
            "gradient_similarity_to_reference",
            "gradient_distance_to_reference",
        ],
    )
    _write_tsv(
        run.metric_paths.gradient_network_output(atlas),
        network_rows,
        [
            "subject",
            "session",
            "task",
            "condition_group",
            "atlas",
            "reference_group",
            "metric",
            "network",
            "comparison_network",
            "gradient",
            "centroid",
            "dispersion",
            "separation",
            "n_rois",
        ],
    )
    _write_metadata_json(item, reference_group, reference_n_runs, cfg)


def _network_rows(
    run: Any,
    atlas: str,
    condition_group: str,
    reference_group: str,
    gradient_index: int,
    values: np.ndarray,
    networks: list[str],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    min_rois = int(
        cfg.get("metrics", {})
        .get("gradient", {})
        .get("network_summary", {})
        .get("min_rois_per_network", 2)
    )
    centroids: dict[str, float] = {}
    counts: dict[str, int] = {}
    for network in sorted(set(networks)):
        idx = [i for i, name in enumerate(networks) if name == network]
        if len(idx) < min_rois:
            continue
        network_values = values[idx]
        centroid = float(np.nanmean(network_values))
        centroids[network] = centroid
        counts[network] = len(idx)
        dispersion = float(np.nanstd(network_values)) if len(network_values) > 1 else 0.0
        rows.append(
            _base_row(run, atlas, condition_group)
            | {
                "reference_group": reference_group,
                "metric": "within_network_dispersion",
                "network": network,
                "comparison_network": "",
                "gradient": gradient_index,
                "centroid": _fmt(centroid),
                "dispersion": _fmt(dispersion),
                "separation": "",
                "n_rois": len(idx),
            }
        )
    network_names = sorted(centroids)
    for pos, network in enumerate(network_names):
        for comparison in network_names[pos + 1:]:
            rows.append(
                _base_row(run, atlas, condition_group)
                | {
                    "reference_group": reference_group,
                    "metric": "between_network_separation",
                    "network": network,
                    "comparison_network": comparison,
                    "gradient": gradient_index,
                    "centroid": "",
                    "dispersion": "",
                    "separation": _fmt(abs(centroids[network] - centroids[comparison])),
                    "n_rois": f"{counts[network]},{counts[comparison]}",
                }
            )
    return rows


def _write_metadata_json(
    item: _RunMatrix,
    reference_group: str,
    reference_n_runs: int,
    cfg: dict[str, Any],
) -> None:
    gcfg = cfg.get("metrics", {}).get("gradient", {})
    meta = {
        "backend": "brainspace",
        "n_components": int(gcfg.get("n_components", 3)),
        "approach": gcfg.get("approach", "dm"),
        "kernel": gcfg.get("kernel", "normalized_angle"),
        "alignment": gcfg.get("alignment", "procrustes"),
        "subject": item.run.subject,
        "session": item.run.session,
        "task": item.run.task,
        "condition_group": item.condition_group,
        "atlas": item.atlas.name,
        "reference_group": reference_group,
        "reference_n_runs": reference_n_runs,
        "fc_source": str(item.fc_source),
        "fc_metadata_source": str(item.fc_metadata_source),
    }
    path = item.run.metric_paths.gradient_metadata_json_output(item.atlas.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def _base_row(run: Any, atlas: str, condition_group: str) -> dict[str, Any]:
    return {
        "subject": run.subject,
        "session": run.session,
        "task": run.task,
        "condition_group": condition_group,
        "atlas": atlas,
    }


def _network_from_roi_name(label: str) -> str:
    parts = [part for part in label.replace("-", "_").split("_") if part]
    if len(parts) >= 3 and parts[0].endswith("Networks") and parts[1] in {"LH", "RH"}:
        return parts[2]
    if len(parts) >= 3 and parts[0] in {"L", "R", "LH", "RH"} and parts[-1].isdigit():
        return parts[1]

    normalized_tokens = {part.lower() for part in parts}
    fallback = {
        "visual": "Vis",
        "vis": "Vis",
        "somatomotor": "SomMot",
        "sommot": "SomMot",
        "dorsalattention": "DorsAttn",
        "dorsattn": "DorsAttn",
        "ventralattention": "SalVentAttn",
        "salventattn": "SalVentAttn",
        "salience": "SalVentAttn",
        "limbic": "Limbic",
        "control": "Cont",
        "cont": "Cont",
        "default": "Default",
        "temporoparietal": "TempPar",
        "temppar": "TempPar",
    }
    for token in normalized_tokens:
        if token in fallback:
            return fallback[token]
    return "unknown"


def _write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _as_1d(values: Any) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)
    return np.ravel(np.asarray(values, dtype=float))


def _value_at(values: np.ndarray, index: int) -> float:
    if index >= len(values):
        return float("nan")
    return float(values[index])


def _positive_sum(values: np.ndarray) -> float:
    positive = values[np.isfinite(values) & (values > 0)]
    return float(np.sum(positive)) if positive.size else float("nan")


def _ratio(value: float, denominator: float) -> float:
    if math.isnan(float(value)) or math.isnan(float(denominator)) or denominator == 0:
        return float("nan")
    return float(value) / float(denominator)


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or right.size < 2:
        return float("nan")
    if np.nanstd(left) == 0 or np.nanstd(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _fmt(value: float) -> str:
    value = float(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"
