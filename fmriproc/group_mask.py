from __future__ import annotations

import csv
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from .motion_qc_utils import evaluate_session_motion, extract_mean_fd, extract_fd_outlier_ratio, first_float, read_first_tsv_row
from .paths import GroupMaskPaths


@dataclass(frozen=True)
class GroupMaskInput:
    subject: str
    session: str
    task: str
    selected_t1_mni: Path | None
    derived_mask_path: Path | None
    status: str
    exclusion_reason: str = ""
    motion_summary_path: Path | None = None
    mean_fd: float | None = None
    fd_outlier_ratio: float | None = None


@dataclass(frozen=True)
class GroupMaskResult:
    paths: GroupMaskPaths
    valid_inputs: tuple[GroupMaskInput, ...]
    commands: tuple[list[str], ...]
    qc_warnings: tuple[str, ...] = ()


def build_3dmask_tool_command(inputs: list[Path], fraction: float, output: Path) -> list[str]:
    return ["3dmask_tool", "-input", *[str(path) for path in inputs], "-frac", str(fraction), "-prefix", str(output)]


def build_group_mask(
    prep_dir: Path,
    cfg: dict[str, Any],
    force: bool = False,
    dry_run: bool = False,
) -> GroupMaskResult:
    paths = GroupMaskPaths(prep_dir)
    paths.ensure_dirs()
    records = collect_group_mask_inputs(prep_dir, cfg)
    _write_inputs_tsv(paths.inputs_tsv, records)
    valid = tuple(record for record in records if record.status == "included" and record.derived_mask_path is not None)

    min_valid = int(cfg.get("metrics", {}).get("group_mask", {}).get("min_valid_subjects", 1))
    if len(valid) < min_valid:
        raise RuntimeError(f"Found {len(valid)} valid group-mask inputs, fewer than required min_valid_subjects={min_valid}")

    thresholds = cfg.get("metrics", {}).get("group_mask", {}).get("thresholds", [0.7, 1.0])
    threshold_outputs = {0.7: paths.thr70_mask, 1.0: paths.thr100_mask}
    commands: list[list[str]] = []
    log_chunks: list[str] = []
    input_masks = [record.derived_mask_path for record in valid if record.derived_mask_path is not None]

    for threshold in thresholds:
        fraction = float(threshold)
        output = threshold_outputs.get(fraction, paths.group_mask_dir / f"grp_mask_thr{int(round(fraction * 100))}.nii.gz")
        command = build_3dmask_tool_command(input_masks, fraction, output)
        commands.append(command)

        if output.exists() and not force:
            log_chunks.append(f"SKIP existing {output}: {_quote(command)}\n")
            continue
        if dry_run:
            log_chunks.append(f"DRY RUN: {_quote(command)}\n")
            continue
        if output.exists() and force:
            output.unlink()
        if shutil.which("3dmask_tool") is None:
            raise RuntimeError("AFNI 3dmask_tool is not available on PATH.")
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        log_chunks.append(_format_completed(command, completed.returncode, completed.stdout, completed.stderr))
        if completed.returncode != 0:
            raise RuntimeError(f"3dmask_tool failed while writing {output}; see {paths.log_path}")

    paths.log_path.write_text("\n".join(log_chunks), encoding="utf-8")

    qc_warnings: tuple[str, ...] = ()
    if not dry_run and paths.thr100_mask.exists():
        from .group_mask_qc import render_group_mask_qc

        qc_warnings = tuple(render_group_mask_qc(paths, cfg))
    return GroupMaskResult(paths=paths, valid_inputs=valid, commands=tuple(commands), qc_warnings=qc_warnings)


def collect_group_mask_inputs(prep_dir: Path, cfg: dict[str, Any] | None = None) -> list[GroupMaskInput]:
    cfg = cfg or {}
    records: list[GroupMaskInput] = []
    for session_dir in sorted(prep_dir.glob("sub-*/ses-*")):
        if not session_dir.is_dir():
            continue
        subject = session_dir.parent.name
        session = session_dir.name
        anat_dir = session_dir / "anat"
        selected = _select_final_mni_t1(anat_dir)
        if selected is None:
            records.append(GroupMaskInput(subject, session, "", None, None, "excluded", "no_final_mni_t1"))
            continue

        selection_cfg = cfg.get("metrics", {}).get("group_mask", {}).get("selection", {})
        motion_decision = evaluate_session_motion(session_dir, selection_cfg)
        if not motion_decision["include"]:
            records.append(
                GroupMaskInput(
                    subject=subject,
                    session=session,
                    task=str(motion_decision.get("task", "")),
                    selected_t1_mni=selected,
                    derived_mask_path=None,
                    status="excluded",
                    exclusion_reason=str(motion_decision["reason"]),
                    motion_summary_path=motion_decision.get("motion_summary_path"),
                    mean_fd=motion_decision.get("mean_fd"),
                    fd_outlier_ratio=motion_decision.get("fd_outlier_ratio"),
                )
            )
            continue

        mask_path = anat_dir / f"{subject}_{session}_mni_coverage_mask.nii.gz"
        try:
            derive_coverage_mask(selected, mask_path)
        except Exception as exc:
            records.append(GroupMaskInput(subject, session, str(motion_decision.get("task", "")), selected, None, "excluded", str(exc)))
            continue
        records.append(
            GroupMaskInput(
                subject=subject,
                session=session,
                task=str(motion_decision.get("task", "")),
                selected_t1_mni=selected,
                derived_mask_path=mask_path,
                status="included",
                exclusion_reason="",
                motion_summary_path=motion_decision.get("motion_summary_path"),
                mean_fd=motion_decision.get("mean_fd"),
                fd_outlier_ratio=motion_decision.get("fd_outlier_ratio"),
            )
        )
    return records


def derive_coverage_mask(t1_mni_path: Path, output_path: Path) -> Path:
    img = nib.load(str(t1_mni_path))
    data = np.asanyarray(img.dataobj)
    mask = np.isfinite(data) & (data != 0)
    out = nib.Nifti1Image(mask.astype(np.uint8), img.affine, img.header)
    out.set_data_dtype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(output_path))
    return output_path


def _select_final_mni_t1(anat_dir: Path) -> Path | None:
    for filename in ("t1_s05_space_mni152_nonlinear.nii.gz", "t1_s06_space_mni152_nonlinear.nii.gz"):
        path = anat_dir / filename
        if path.exists():
            return path
    return None


def _write_inputs_tsv(path: Path, records: list[GroupMaskInput]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject",
        "session",
        "task",
        "selected_t1_mni",
        "derived_mask_path",
        "status",
        "exclusion_reason",
        "motion_summary_path",
        "mean_fd",
        "fd_outlier_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "subject": record.subject,
                    "session": record.session,
                    "task": record.task,
                    "selected_t1_mni": str(record.selected_t1_mni or ""),
                    "derived_mask_path": str(record.derived_mask_path or ""),
                    "status": record.status,
                    "exclusion_reason": record.exclusion_reason,
                    "motion_summary_path": str(record.motion_summary_path or ""),
                    "mean_fd": "" if record.mean_fd is None else record.mean_fd,
                    "fd_outlier_ratio": "" if record.fd_outlier_ratio is None else record.fd_outlier_ratio,
                }
            )


def _quote(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _format_completed(command: list[str], returncode: int, stdout: str, stderr: str) -> str:
    return "\n".join(
        [
            f"COMMAND: {_quote(command)}",
            f"RETURN_CODE: {returncode}",
            "STDOUT:",
            stdout,
            "STDERR:",
            stderr,
        ]
    )
