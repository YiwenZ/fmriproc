from __future__ import annotations

from pathlib import Path


def write_anat_readme(path: Path, backend: str, registration_backend: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Anatomical preprocessing",
                "",
                f"- backend: {backend}",
                f"- registration_backend: {registration_backend}",
                "- purpose: prepare T1 image, brain mask, tissue masks, and MNI registration transforms.",
                "- naming: files use step labels such as t1_s01, t1_s02, and t1_s05.",
                "- review: check QC snapshots in ../qc/task-*/snapshots/ when available.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_func_readme(path: Path, backend: str, registration_backend: str, task: str, motion_model: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Functional preprocessing",
                "",
                f"- task: {task}",
                f"- backend: {backend}",
                f"- registration_backend: {registration_backend}",
                f"- motion_model: {motion_model}",
                "- purpose: prepare BOLD time series, motion correction, nuisance regression, MNI transform, and smoothing.",
                "- naming: files use step labels such as bold_s03, bold_s08, and bold_s12.",
                "- review: check motion_summary.tsv and snapshots/ in the task QC directory.",
                "",
            ]
        ),
        encoding="utf-8",
    )
