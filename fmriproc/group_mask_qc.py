from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from .paths import GroupMaskPaths


def build_slicer_command(mask_path: Path, output_png: Path) -> list[str]:
    return ["slicer", str(mask_path), "-a", str(output_png)]


def render_group_mask_qc(paths: GroupMaskPaths, cfg: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if shutil.which("slicer") is None:
        warnings.append("FSL slicer not found; skipped group-mask QC snapshot.")
        _append_warnings(paths, warnings)
        return warnings

    output = paths.qc_snapshot
    output.parent.mkdir(parents=True, exist_ok=True)
    command = build_slicer_command(paths.thr100_mask, output)
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        warnings.append(f"slicer failed while rendering group-mask QC snapshot: {completed.stderr.strip()}")
    _append_warnings(paths, warnings)
    return warnings


def _append_warnings(paths: GroupMaskPaths, warnings: list[str]) -> None:
    if not warnings:
        return
    paths.log_path.parent.mkdir(parents=True, exist_ok=True)
    with paths.log_path.open("a", encoding="utf-8") as handle:
        for warning in warnings:
            handle.write(f"[WARN] {warning}\n")
