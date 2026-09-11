"""Dataset-level ComBat harmonization for Fisher-z FC matrices."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone

from .paths import DatasetMetricPaths

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MatchedRunCovariates:
    run: Any
    covariates: dict[str, Any]


def _combat_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("metrics", {}).get("combat", {})


def resolve_combat_covariate_columns(cfg: dict[str, Any]) -> dict[str, list[str]]:
    combat_cfg = _combat_cfg(cfg)
    batch_column = str(combat_cfg.get("batch_column", "Site"))

    continuous = list(combat_cfg.get("required_covariates", {}).get("continuous", ["Age", "MeanFD"]))
    continuous += list(combat_cfg.get("optional_covariates", {}).get("continuous", []))
    if "MeanFD" not in continuous:
        continuous.append("MeanFD")

    categorical = list(combat_cfg.get("required_covariates", {}).get("categorical", ["Sex"]))
    categorical += list(combat_cfg.get("optional_covariates", {}).get("categorical", []))
    if batch_column in categorical:
        raise ValueError(f"Batch column '{batch_column}' must not be duplicated in categorical covariates.")

    return {
        "batch": [batch_column],
        "categorical": categorical,
        "continuous": continuous,
    }


def load_combat_covariates(cfg: dict[str, Any]) -> pd.DataFrame:
    combat_cfg = _combat_cfg(cfg)
    path_value = combat_cfg.get("covariates_csv")
    if not path_value:
        raise ValueError("metrics.combat.covariates_csv is required when ComBat is enabled.")
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"ComBat covariates CSV not found: {path}")
    covars = pd.read_csv(path)

    resolved = resolve_combat_covariate_columns(cfg)

    required = {
        str(combat_cfg.get("id_column", "ID")),
        *resolved["batch"],
        *resolved["continuous"],
        *resolved["categorical"],
    }
    missing = sorted(column for column in required if column not in covars.columns)
    if missing:
        raise ValueError(f"ComBat covariates CSV is missing required column(s): {', '.join(missing)}")
    return covars


def validate_combat_config(cfg: dict[str, Any], covars: pd.DataFrame | None = None) -> None:
    if cfg.get("metrics", {}).get("input", {}).get("fc_variant") == "combat":
        if not _combat_cfg(cfg).get("enabled", False):
            raise ValueError("ComBat must be enabled when fc_variant is 'combat'.")

    combat_cfg = _combat_cfg(cfg)
    match_keys = combat_cfg.get("match_keys", ["ID", "session", "task"])
    if not set(match_keys).issubset({"ID", "session", "task"}):
        raise ValueError(f"Invalid match_keys: {match_keys}. Must be a subset of {{'ID', 'session', 'task'}}.")

    if covars is not None:
        allowed_runtime_columns = {"atlas"}
        for column in combat_cfg.get("group_by", ["session", "modal", "atlas"]):
            if column not in covars.columns and column not in allowed_runtime_columns:
                raise ValueError(f"ComBat group_by column not found: {column}")
        for column in combat_cfg.get("match_keys", ["ID", "session", "task"]):
            if column != "ID" and column not in covars.columns:
                raise ValueError(f"ComBat match_keys column not found in covariates CSV: {column}")


def _run_key(run: Any, match_keys: list[str]) -> tuple[str, ...]:
    values: list[str] = []
    for key in match_keys:
        if key == "ID":
            values.append(str(run.subject))
        elif key == "session":
            values.append(str(run.session))
        elif key == "task":
            values.append(str(run.task))
        else:
            raise ValueError(f"Unsupported run match key: {key}")
    return tuple(values)


def match_covariates_to_runs(
    runs: list[Any],
    covars: pd.DataFrame,
    cfg: dict[str, Any],
) -> list[MatchedRunCovariates]:
    combat_cfg = _combat_cfg(cfg)
    match_keys = list(combat_cfg.get("match_keys", ["ID", "session", "task"]))
    csv_keys = [str(combat_cfg.get("id_column", "ID")) if key == "ID" else key for key in match_keys]

    duplicated = covars.duplicated(subset=csv_keys, keep=False)
    if duplicated.any():
        duplicate_values = covars.loc[duplicated, csv_keys].drop_duplicates().to_dict("records")
        raise ValueError(f"Duplicate covariate rows for match key(s): {duplicate_values}")

    covar_index = {
        tuple(str(row[column]) for column in csv_keys): row.to_dict()
        for _, row in covars.iterrows()
    }

    matched: list[MatchedRunCovariates] = []
    missing: list[str] = []
    for run in runs:
        key = _run_key(run, match_keys)
        row = covar_index.get(key)
        if row is None:
            missing.append("/".join(key))
            continue
        matched.append(MatchedRunCovariates(run=run, covariates=row))

    if missing:
        raise ValueError(f"Missing ComBat covariate rows for run key(s): {', '.join(missing)}")
    return matched


def matrix_to_edge_vector(matrix: np.ndarray, labels: list[str]) -> tuple[np.ndarray, list[str]]:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("FC matrix must be square.")
    if matrix.shape[0] != len(labels):
        raise ValueError("FC matrix shape does not match ROI labels.")
    i_upper, j_upper = np.triu_indices(matrix.shape[0], k=1)
    edge_names = [f"{labels[i]}--{labels[j]}" for i, j in zip(i_upper, j_upper, strict=True)]
    return matrix[i_upper, j_upper].astype(float), edge_names


def edge_vector_to_matrix(edge_vector: np.ndarray, n_rois: int) -> np.ndarray:
    vector = np.asarray(edge_vector, dtype=float)
    expected = n_rois * (n_rois - 1) // 2
    if vector.shape[0] != expected:
        raise ValueError(f"Expected {expected} edges for {n_rois} ROIs, got {vector.shape[0]}.")
    matrix = np.zeros((n_rois, n_rois), dtype=float)
    i_upper, j_upper = np.triu_indices(n_rois, k=1)
    matrix[i_upper, j_upper] = vector
    matrix[j_upper, i_upper] = vector
    return matrix


CombatBackend = Callable[[np.ndarray, pd.DataFrame, str, list[str], list[str]], np.ndarray]


def run_combat_backend(
    data_matrix: np.ndarray,
    covars: pd.DataFrame,
    batch_col: str,
    continuous_cols: list[str],
    categorical_cols: list[str],
    backend: CombatBackend | None = None,
) -> np.ndarray:
    """Run ComBat on an edges-by-samples matrix and return adjusted data."""
    if categorical_cols and batch_col in categorical_cols:
        raise ValueError(f"Batch column '{batch_col}' must not be duplicated in categorical covariates.")
        
    if data_matrix.ndim != 2 or data_matrix.shape[1] != len(covars):
        raise ValueError("data_matrix must be 2D (features x samples), with samples matching len(covars)")

    if backend is not None:
        return np.asarray(backend(data_matrix, covars, batch_col, continuous_cols, categorical_cols), dtype=float)

    try:
        from neuroCombat import neuroCombat
    except ImportError as exc:
        raise RuntimeError(
            "metrics.combat.backend='neurocombat' requires the neuroCombat Python package. "
            "Install it in the pipeline environment or disable metrics.combat.enabled."
        ) from exc

    result = neuroCombat(
        dat=np.asarray(data_matrix, dtype=float),
        covars=covars,
        batch_col=batch_col,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
    )
    if isinstance(result, dict) and "data" in result:
        return np.asarray(result["data"], dtype=float)
    raise RuntimeError("neuroCombat did not return a result dictionary with a 'data' field.")


def _read_matrix_tsv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
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
        raise ValueError(f"FC matrix row labels do not match column labels: {path}")
    return labels, matrix


def _write_matrix_tsv(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([""] + labels)
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label] + [f"{float(value):.10g}" for value in row])


def _group_key(covariates: dict[str, Any], atlas: str, group_by: list[str]) -> tuple[tuple[str, str], ...]:
    values: list[tuple[str, str]] = []
    for column in group_by:
        if column == "atlas":
            values.append(("atlas", atlas))
        else:
            values.append((column, str(covariates[column])))
    return tuple(values)


def _group_status(covars: pd.DataFrame, cfg: dict[str, Any]) -> tuple[str, str]:
    combat_cfg = _combat_cfg(cfg)
    batch_col = str(combat_cfg.get("batch_column", "Site"))
    if len(covars) < int(combat_cfg.get("min_samples_per_group", 8)):
        return "skipped", "insufficient_samples"
    batch_counts = covars[batch_col].value_counts()
    if len(batch_counts) < int(combat_cfg.get("min_batches_per_group", 2)):
        return "skipped", "insufficient_batches"
    if (batch_counts < int(combat_cfg.get("min_samples_per_batch", 2))).any():
        return "skipped", "insufficient_samples_per_batch"
    return "completed", "ok"


def run_fc_combat(
    prep_dir: Path,
    cfg: dict[str, Any],
    included_runs: list[Any],
    atlas_names: list[str],
    backend: CombatBackend | None = None,
) -> list[Path]:
    validate_combat_config(cfg)
    combat_cfg = _combat_cfg(cfg)
    if not combat_cfg.get("enabled", False):
        return []

    covars = load_combat_covariates(cfg)
    validate_combat_config(cfg, covars)
    matched = match_covariates_to_runs(included_runs, covars, cfg)

    group_by = list(combat_cfg.get("group_by", ["session", "modal", "atlas"]))
    resolved = resolve_combat_covariate_columns(cfg)
    batch_col = resolved["batch"][0]
    continuous_cols = resolved["continuous"]
    categorical_cols = resolved["categorical"]

    grouped: dict[tuple[tuple[str, str], ...], list[dict[str, Any]]] = defaultdict(list)
    for matched_item in matched:
        for atlas in atlas_names:
            fc_path = matched_item.run.metric_paths.fc_fisherz_output(atlas)
            raw_meta_path = matched_item.run.metric_paths.metadata_json_output(atlas)
            if not fc_path.exists() or not raw_meta_path.exists():
                raise RuntimeError(
                    f"ComBat requires complete raw FC outputs before harmonization: "
                    f"{matched_item.run.subject}/{matched_item.run.session}/task-{matched_item.run.task}/{atlas}"
                )
            labels, matrix = _read_matrix_tsv(fc_path)
            edge_vector, edge_names = matrix_to_edge_vector(matrix, labels)
            grouped[_group_key(matched_item.covariates, atlas, group_by)].append({
                "run": matched_item.run,
                "atlas": atlas,
                "labels": labels,
                "edge_vector": edge_vector,
                "edge_names": edge_names,
                "covariates": matched_item.covariates,
            })

    dataset_paths = DatasetMetricPaths(prep_dir)
    dataset_paths.ensure_dirs()
    group_rows: list[dict[str, Any]] = []
    manifest_groups: list[dict[str, Any]] = []
    generated: list[Path] = []

    for key, items in sorted(grouped.items()):
        key_dict = dict(key)
        covar_df = pd.DataFrame([item["covariates"] for item in items])
        status, reason = _group_status(covar_df, cfg)
        group_rows.append({
            **key_dict,
            "n_runs": len(items),
            "n_batches": covar_df[batch_col].nunique(),
            "status": status,
            "reason": reason,
        })
        manifest_groups.append({
            "group_key": key_dict,
            "n_runs": len(items),
            "n_batches": int(covar_df[batch_col].nunique()),
            "status": status,
            "reason": reason,
        })
        if status != "completed":
            continue

        first_labels = items[0]["labels"]
        first_edges = items[0]["edge_names"]
        for item in items[1:]:
            if item["labels"] != first_labels or item["edge_names"] != first_edges:
                raise ValueError(f"ROI label mismatch within ComBat group: {key_dict}")

        data_matrix = np.vstack([item["edge_vector"] for item in items]).T
        adjusted = run_combat_backend(
            data_matrix,
            covar_df,
            batch_col=batch_col,
            continuous_cols=continuous_cols,
            categorical_cols=categorical_cols,
            backend=backend,
        )
        for sample_index, item in enumerate(items):
            matrix = edge_vector_to_matrix(adjusted[:, sample_index], len(first_labels))
            out_path = item["run"].metric_paths.fc_fisherz_combat_output(item["atlas"])
            meta_path = item["run"].metric_paths.fc_combat_metadata_json_output(item["atlas"])
            _write_matrix_tsv(out_path, first_labels, matrix)
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps({
                "generated": datetime.now(timezone.utc).isoformat(),
                "fc_variant": "combat",
                "source_fc": str(item["run"].metric_paths.fc_fisherz_output(item["atlas"])),
                "backend": combat_cfg.get("backend", "neurocombat"),
                "batch_column": batch_col,
                "continuous_covariates": continuous_cols,
                "categorical_covariates": categorical_cols,
                "group_by": group_by,
                "group_key": key_dict,
            }, indent=2, sort_keys=True), encoding="utf-8")
            generated.extend([out_path, meta_path])

    _write_group_summary(dataset_paths.combat_groups_tsv, group_rows)
    dataset_paths.combat_manifest_json.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(),
        "status": "ready",
        "groups": manifest_groups,
        "covariates_csv": str(combat_cfg.get("covariates_csv")),
        "group_by": group_by,
    }, indent=2, sort_keys=True), encoding="utf-8")
    return [dataset_paths.combat_manifest_json, dataset_paths.combat_groups_tsv] + generated


def _write_group_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) or ["status", "reason"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
