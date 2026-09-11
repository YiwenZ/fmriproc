"""Shared helpers for reading and evaluating motion QC summaries.

Used by both group_mask and metrics_dataset modules to apply consistent
inclusion rules based on framewise displacement (FD) metrics.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def read_first_tsv_row(path: Path) -> dict[str, str]:
    """Return the first data row of a TSV file as a string-keyed dict."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return next(reader, {})


def first_float(row: dict[str, str], keys: tuple[str, ...]) -> float | None:
    """Return the first value in *row* that matches one of *keys* as a float."""
    for key in keys:
        value = row.get(key)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return None


def extract_mean_fd(row: dict[str, str]) -> float | None:
    """Extract mean FD from a motion summary row.

    Handles AFNI-style field names (``mean_fd``, ``mean_fd_power``) and
    FSL-style names (``mean_fd_mm``).
    """
    return first_float(row, ("mean_fd", "mean_fd_power", "mean_fd_mm"))


def extract_fd_outlier_ratio(row: dict[str, str], motion_cfg: dict[str, Any]) -> float | None:
    """Extract FD outlier ratio from a motion summary row.

    Accepts an explicit ``fd_outlier_ratio`` or ``outlier_ratio`` field.
    Falls back to computing the ratio from percent or count columns using
    the configured ``fd_outlier_threshold`` (default 0.5).
    """
    explicit = first_float(row, ("fd_outlier_ratio", "outlier_ratio"))
    if explicit is not None:
        return explicit

    threshold = float(motion_cfg.get("fd_outlier_threshold", 0.5))
    if threshold == 0.5:
        percent_keys = ("percent_fd_power_gt_0_5", "percent_fd_gt_0_5")
        count_keys = ("n_fd_power_gt_0_5", "n_fd_gt_0_5")
    else:
        percent_keys = ("percent_fd_power_gt_0_2", "percent_fd_gt_0_2")
        count_keys = ("n_fd_power_gt_0_2", "n_fd_gt_0_2")

    percent = first_float(row, percent_keys)
    if percent is not None:
        return percent / 100.0

    n_outliers = first_float(row, count_keys)
    n_volumes = first_float(row, ("n_volumes_used", "n_volumes"))
    if n_outliers is not None and n_volumes and n_volumes > 0:
        return n_outliers / n_volumes
    return None


def evaluate_session_motion(session_dir: Path, selection_cfg: dict[str, Any]) -> dict[str, Any]:
    """Determine whether *session_dir* passes dataset-level motion QC.

    Returns a dict with keys ``include``, ``reason``, ``task``, and
    optionally ``motion_summary_path``, ``mean_fd``, ``fd_outlier_ratio``.

    Parameters
    ----------
    session_dir:
        ``Prep/sub-*/ses-*/`` directory.
    selection_cfg:
        The ``metrics.group_mask.selection`` (or
        ``metrics.dataset.inclusion``) config dict.
    """
    if not selection_cfg.get("require_motion_qc", True):
        return {"include": True, "reason": "", "task": "motion_qc_not_required"}

    motion_cfg = selection_cfg.get("motion", {})
    mean_fd_max = float(motion_cfg.get("mean_fd_max", 0.5))
    outlier_ratio_max = float(motion_cfg.get("fd_outlier_ratio_max", 0.2))

    summaries = sorted((session_dir / "qc").glob("task-*/motion_summary.tsv"))
    if not summaries:
        return {"include": False, "reason": "missing_motion_summary", "task": ""}

    failures: list[str] = []
    for summary_path in summaries:
        row = read_first_tsv_row(summary_path)
        task = summary_path.parent.name.removeprefix("task-")
        mean_fd = extract_mean_fd(row)
        ratio = extract_fd_outlier_ratio(row, motion_cfg)
        if mean_fd is None:
            failures.append(f"{task}:missing_mean_fd")
            continue
        if ratio is None:
            failures.append(f"{task}:missing_fd_outlier_ratio")
            continue
        if mean_fd < mean_fd_max and ratio < outlier_ratio_max:
            return {
                "include": True,
                "reason": "",
                "task": task,
                "motion_summary_path": summary_path,
                "mean_fd": mean_fd,
                "fd_outlier_ratio": ratio,
            }
        failures.append(f"{task}:mean_fd={mean_fd:.6g},fd_outlier_ratio={ratio:.6g}")

    return {"include": False, "reason": "motion_qc_failed:" + ";".join(failures), "task": ""}


def evaluate_task_motion(
    session_dir: Path, task: str, selection_cfg: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate motion QC for a single *task* within *session_dir*.

    Unlike :func:`evaluate_session_motion`, which accepts a session when any
    task passes, this helper checks only the given task's
    ``qc/task-<task>/motion_summary.tsv``.

    Returns a dict with the same shape as :func:`evaluate_session_motion`.
    """
    if not selection_cfg.get("require_motion_qc", True):
        return {"include": True, "reason": "", "task": task}

    motion_cfg = selection_cfg.get("motion", {})
    mean_fd_max = float(motion_cfg.get("mean_fd_max", 0.5))
    outlier_ratio_max = float(motion_cfg.get("fd_outlier_ratio_max", 0.2))

    summary_path = session_dir / "qc" / f"task-{task}" / "motion_summary.tsv"
    if not summary_path.exists():
        return {"include": False, "reason": "missing_motion_summary", "task": task}

    row = read_first_tsv_row(summary_path)
    mean_fd = extract_mean_fd(row)
    ratio = extract_fd_outlier_ratio(row, motion_cfg)

    if mean_fd is None:
        return {"include": False, "reason": f"{task}:missing_mean_fd", "task": task}
    if ratio is None:
        return {"include": False, "reason": f"{task}:missing_fd_outlier_ratio", "task": task}

    if mean_fd < mean_fd_max and ratio < outlier_ratio_max:
        return {
            "include": True,
            "reason": "",
            "task": task,
            "motion_summary_path": summary_path,
            "mean_fd": mean_fd,
            "fd_outlier_ratio": ratio,
        }

    return {
        "include": False,
        "reason": f"motion_qc_failed:{task}:mean_fd={mean_fd:.6g},fd_outlier_ratio={ratio:.6g}",
        "task": task,
    }
