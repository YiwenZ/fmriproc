"""Resolve raw or harmonized FC inputs for FC-derived metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import MetricPaths


@dataclass(frozen=True)
class FCInputPaths:
    variant: str
    matrix_path: Path
    metadata_path: Path
    qc_path: Path | None


def fc_variant_from_config(cfg: dict[str, Any]) -> str:
    metrics_cfg = cfg.get("metrics") or {}
    input_cfg = metrics_cfg.get("input") or {}
    variant = str(input_cfg.get("fc_variant", "raw"))
    if variant not in {"raw", "combat"}:
        raise ValueError("metrics.input.fc_variant must be 'raw' or 'combat'.")
    combat_cfg = metrics_cfg.get("combat") or {}
    if variant == "combat" and not combat_cfg.get("enabled", False):
        raise ValueError("fc_variant is 'combat' but metrics.combat.enabled is not true.")
    return variant


def resolve_fc_input_paths(metric_paths: MetricPaths, atlas: str, cfg: dict[str, Any], force_raw: bool = False) -> FCInputPaths:
    variant = fc_variant_from_config(cfg) if not force_raw else "raw"
    if variant == "raw":
        return FCInputPaths(
            variant="raw",
            matrix_path=metric_paths.fc_fisherz_output(atlas),
            metadata_path=metric_paths.metadata_json_output(atlas),
            qc_path=metric_paths.qc_tsv_output(atlas),
        )
    return FCInputPaths(
        variant="combat",
        matrix_path=metric_paths.fc_fisherz_combat_output(atlas),
        metadata_path=metric_paths.fc_combat_metadata_json_output(atlas),
        qc_path=None,
    )


def check_fc_input_prerequisites(metric_paths: MetricPaths, atlas: str, cfg: dict[str, Any], force_raw: bool = False) -> list[str]:
    resolved = resolve_fc_input_paths(metric_paths, atlas, cfg, force_raw=force_raw)
    missing: list[str] = []
    if not resolved.matrix_path.exists():
        missing.append(f"missing {resolved.variant} Fisher-z matrix: {resolved.matrix_path}")
    if not resolved.metadata_path.exists():
        missing.append(f"missing {resolved.variant} FC metadata JSON: {resolved.metadata_path}")
    if resolved.qc_path is not None and not resolved.qc_path.exists():
        missing.append(f"missing raw FC QC TSV: {resolved.qc_path}")
    return missing
