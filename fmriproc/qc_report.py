from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .motion import plot_motion_parameters
from .paths import PrepPaths
from .qc_images import render_final_mni_overlay_png, render_overlay_png, render_slice_montage_png
from .readme_writer import write_anat_readme, write_func_readme


def _first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def _template_underlay(config: dict[str, Any]) -> Path | None:
    templates_cfg = config.get("templates", {})
    resolution = templates_cfg.get("mni152_default_resolution", "3mm")
    key = f"mni152_{resolution}_brain"
    configured = templates_cfg.get(key)
    if configured:
        configured_path = Path(configured)
        candidates = [
            configured_path,
            Path(__file__).parent.parent / configured,
        ]
        found = _first_existing(candidates)
        if found:
            return found

    fallback = Path(__file__).parent / "resources" / "templates" / "MNI152" / f"MNI152_T1_{resolution}_brain.nii.gz"
    if fallback.exists():
        return fallback
    return None


def _all_existing(paths: list[Path]) -> bool:
    return all(path.exists() for path in paths)


def _tools_available(names: list[str]) -> bool:
    return all(shutil.which(name) for name in names)


def _make_qc_epi_ref_mni(paths: PrepPaths, config: dict[str, Any], template: Path, tmp_dir: Path) -> Path | None:
    """Warp a 3D EPI reference to MNI for QC-only rendering.

    The final processed BOLD can be nearly flat after nuisance regression and
    grand-mean scaling, so using it as a grayscale underlay often produces
    washed-out MNI snapshots. This helper uses the same registration transforms
    on a preprocessed EPI reference instead.
    """
    runtime = config.get("runtime", {})
    backend = runtime.get("backend", "afni")
    reg_backend = runtime.get("registration_backend", "ants_if_available")

    try:
        if backend == "fsl" or reg_backend == "fsl":
            return _make_fsl_qc_epi_ref_mni(paths, template, tmp_dir)
        if reg_backend in {"ants", "ants_if_available"}:
            ants_out = _make_ants_qc_epi_ref_mni(paths, template, tmp_dir)
            if ants_out is not None:
                return ants_out
        if reg_backend == "afni" or backend == "afni":
            return _make_afni_qc_epi_ref_mni(paths, tmp_dir)
    except subprocess.CalledProcessError:
        return None
    return None


def _make_qc_epi_ref_t1(paths: PrepPaths, config: dict[str, Any], t1: Path, tmp_dir: Path) -> Path | None:
    """Warp a 3D EPI reference to T1 for QC-only rendering."""
    runtime = config.get("runtime", {})
    backend = runtime.get("backend", "afni")
    reg_backend = runtime.get("registration_backend", "ants_if_available")

    try:
        if backend == "fsl" or reg_backend == "fsl":
            return _make_fsl_qc_epi_ref_t1(paths, t1, tmp_dir)
        if reg_backend in {"ants", "ants_if_available"}:
            ants_out = _make_ants_qc_epi_ref_t1(paths, t1, tmp_dir)
            if ants_out is not None:
                return ants_out
        if reg_backend == "afni" or backend == "afni":
            return _make_afni_qc_epi_ref_t1(paths, t1, tmp_dir)
    except subprocess.CalledProcessError:
        return None
    return None


def _make_fsl_qc_epi_ref_t1(paths: PrepPaths, t1: Path, tmp_dir: Path) -> Path | None:
    epi_ref = paths.bold_output("bold_s03_ref.nii.gz")
    epi_mat = paths.reg_output("fsl_epi_to_anat_bbr.mat")
    if not _all_existing([epi_ref, epi_mat, t1]) or not _tools_available(["flirt"]):
        return None

    output = tmp_dir / "qc_epi_ref_to_t1.nii.gz"
    subprocess.run(
        [
            "flirt",
            "-in",
            str(epi_ref),
            "-ref",
            str(t1),
            "-applyxfm",
            "-init",
            str(epi_mat),
            "-interp",
            "trilinear",
            "-out",
            str(output),
        ],
        check=True,
    )
    return output


def _make_ants_qc_epi_ref_t1(paths: PrepPaths, t1: Path, tmp_dir: Path) -> Path | None:
    epi_ref = paths.bold_output("bold_s05_epi_ref_unifize.nii.gz")
    epi_to_t1_mat = paths.reg_output("ants_epi_to_t1_0GenericAffine.mat")
    if not _all_existing([epi_ref, epi_to_t1_mat, t1]) or not _tools_available(["antsApplyTransforms"]):
        return None

    output = tmp_dir / "qc_epi_ref_to_t1.nii.gz"
    subprocess.run(
        [
            "antsApplyTransforms",
            "-d",
            "3",
            "-i",
            str(epi_ref),
            "-r",
            str(t1),
            "-o",
            str(output),
            "-n",
            "Linear",
            "-t",
            str(epi_to_t1_mat),
        ],
        check=True,
    )
    return output


def _make_afni_qc_epi_ref_t1(paths: PrepPaths, t1: Path, tmp_dir: Path) -> Path | None:
    epi_ref = paths.bold_output("bold_s05_epi_ref_unifize.nii.gz")
    epi_affine = paths.reg_dir / "bold_s05_epi_ref_unifize_align_mat.aff12.1D"
    if not _all_existing([epi_ref, epi_affine, t1]) or not _tools_available(["3dAllineate"]):
        return None

    output = tmp_dir / "qc_epi_ref_to_t1.nii.gz"
    subprocess.run(
        [
            "3dAllineate",
            "-source",
            str(epi_ref),
            "-master",
            str(t1),
            "-1Dmatrix_apply",
            str(epi_affine),
            "-final",
            "linear",
            "-prefix",
            str(output),
            "-overwrite",
        ],
        check=True,
    )
    return output


def _make_fsl_qc_epi_ref_mni(paths: PrepPaths, template: Path, tmp_dir: Path) -> Path | None:
    epi_ref = paths.bold_output("bold_s03_ref.nii.gz")
    epi_mat = paths.reg_output("fsl_epi_to_anat_bbr.mat")
    anat_warp = paths.reg_output("fsl_anat_to_mni_warp.nii.gz")
    if not _all_existing([epi_ref, epi_mat, anat_warp, template]) or not _tools_available(["convertwarp", "applywarp"]):
        return None

    epi_to_mni_warp = tmp_dir / "qc_fsl_epi_ref_to_mni_warp.nii.gz"
    output = tmp_dir / "qc_epi_ref_to_mni.nii.gz"
    subprocess.run(
        [
            "convertwarp",
            "-r",
            str(template),
            "-o",
            str(epi_to_mni_warp),
            "-m",
            str(epi_mat),
            "-w",
            str(anat_warp),
        ],
        check=True,
    )
    subprocess.run(
        [
            "applywarp",
            "-i",
            str(epi_ref),
            "-r",
            str(template),
            "-w",
            str(epi_to_mni_warp),
            "--interp=trilinear",
            "-o",
            str(output),
        ],
        check=True,
    )
    return output


def _make_ants_qc_epi_ref_mni(paths: PrepPaths, template: Path, tmp_dir: Path) -> Path | None:
    epi_ref = paths.bold_output("bold_s05_epi_ref_unifize.nii.gz")
    epi_to_t1_mat = paths.reg_output("ants_epi_to_t1_0GenericAffine.mat")
    t1_to_mni_warp = paths.reg_output("ants_t1_to_mni_1Warp.nii.gz")
    t1_to_mni_mat = paths.reg_output("ants_t1_to_mni_0GenericAffine.mat")
    if not _all_existing([epi_ref, epi_to_t1_mat, t1_to_mni_warp, t1_to_mni_mat, template]) or not _tools_available(["antsApplyTransforms"]):
        return None

    output = tmp_dir / "qc_epi_ref_to_mni.nii.gz"
    subprocess.run(
        [
            "antsApplyTransforms",
            "-d",
            "3",
            "-i",
            str(epi_ref),
            "-r",
            str(template),
            "-o",
            str(output),
            "-n",
            "Linear",
            "-t",
            str(t1_to_mni_warp),
            "-t",
            str(t1_to_mni_mat),
            "-t",
            str(epi_to_t1_mat),
        ],
        check=True,
    )
    return output


def _make_afni_qc_epi_ref_mni(paths: PrepPaths, tmp_dir: Path) -> Path | None:
    epi_ref = paths.bold_output("bold_s05_epi_ref_unifize.nii.gz")
    warp = paths.reg_output("anat_s02_to_mni152_warp.nii.gz")
    anat_affine = paths.reg_output("anat_s01_to_mni152_affine.1D")
    epi_affine = paths.reg_dir / "bold_s05_epi_ref_unifize_align_mat.aff12.1D"
    if not _all_existing([epi_ref, warp, anat_affine, epi_affine]) or not _tools_available(["3dNwarpApply"]):
        return None

    output = tmp_dir / "qc_epi_ref_to_mni.nii.gz"
    subprocess.run(
        [
            "3dNwarpApply",
            "-source",
            str(epi_ref),
            "-nwarp",
            f"{warp} {anat_affine} {epi_affine}",
            "-master",
            "WARP",
            "-prefix",
            str(output),
            "-overwrite",
        ],
        check=True,
    )
    return output


def generate_qc_report(paths: PrepPaths, config: dict[str, Any]) -> list[Path]:
    generated: list[Path] = []
    qc_cfg = config.get("qc", {})
    snapshots_cfg = qc_cfg.get("snapshots", {})
    backend = config.get("runtime", {}).get("backend", "afni")
    reg_backend = config.get("runtime", {}).get("registration_backend", "ants_if_available")
    n_slices = int(snapshots_cfg.get("n_slices", 9))
    alpha = float(snapshots_cfg.get("overlay_alpha", 0.35))
    snap_dir = paths.qc_task_dir / "snapshots"

    if snapshots_cfg.get("enabled", True):
        t1 = _first_existing([
            paths.anat_output("t1_s03_unifize.nii.gz"),
            paths.anat_output("t1_s02_skullstrip_bet.nii.gz"),
            paths.anat_output("t1_s01_reorient2std.nii.gz"),
        ])
        epi = _first_existing([
            paths.bold_output("bold_s03_motion_correction.nii.gz"),
            paths.bold_output("bold_s04_motion_correction_mcflirt.nii.gz"),
        ])
        epi_ref = _first_existing([
            paths.bold_output("bold_s05_epi_ref_unifize.nii.gz"),
            paths.bold_output("bold_s03_ref.nii.gz"),
        ])
        epi_mni = _first_existing([
            paths.bold_output("bold_s10_space_mni152.nii.gz"),
            paths.bold_output("bold_s11_space_mni152.nii.gz"),
        ])
        t1_mni = _first_existing([
            paths.anat_output("t1_s05_space_mni152_nonlinear.nii.gz"),
            paths.anat_output("t1_s06_space_mni152_nonlinear.nii.gz"),
        ])

        if t1:
            out = snap_dir / "precheck_t1_slices.png"
            render_slice_montage_png(t1, out, "Precheck T1", n_slices=n_slices)
            generated.append(out)
        if epi:
            out = snap_dir / "precheck_epi_slices.png"
            render_slice_montage_png(epi, out, "Precheck EPI", n_slices=n_slices)
            generated.append(out)
        if t1 and epi_ref:
            out = snap_dir / "epi_to_t1_overlay.png"
            with tempfile.TemporaryDirectory(prefix="fmri_qc_epi_ref_t1_") as tmp_name:
                qc_epi_t1 = _make_qc_epi_ref_t1(paths, config, t1, Path(tmp_name))
                if qc_epi_t1:
                    render_final_mni_overlay_png(
                        qc_epi_t1,
                        t1,
                        out,
                        "EPI to T1",
                        n_slices=n_slices,
                        alpha=alpha,
                    )
                else:
                    render_overlay_png(
                        t1,
                        epi_ref,
                        out,
                        "EPI to T1",
                        n_slices=n_slices,
                        alpha=alpha,
                    )
            generated.append(out)
        if t1_mni and epi_mni:
            out = snap_dir / "epi_to_mni_overlay.png"
            mni_template = _template_underlay(config) or t1_mni
            with tempfile.TemporaryDirectory(prefix="fmri_qc_epi_ref_mni_") as tmp_name:
                qc_epi_mni = _make_qc_epi_ref_mni(paths, config, mni_template, Path(tmp_name))
                render_final_mni_overlay_png(
                    qc_epi_mni or epi_mni,
                    mni_template,
                    out,
                    "EPI to MNI",
                    n_slices=n_slices,
                    alpha=alpha,
                )
            generated.append(out)

    if qc_cfg.get("motion_plot", {}).get("enabled", True):
        motion_path = _first_existing([
            paths.qc_output("motion_parameters.1D"),
            paths.qc_output("motion_parameters.par"),
        ])
        if motion_path:
            out = snap_dir / "motion_parameters.png"
            plot_motion_parameters(motion_path, out, backend=backend)
            generated.append(out)

    if qc_cfg.get("readme", {}).get("enabled", True):
        motion_model = config.get("bold", {}).get("nuisance", {}).get("motion_model", "basic_6")
        write_anat_readme(paths.anat_dir / "README.md", backend, reg_backend)
        write_func_readme(paths.func_task_dir / "README.md", backend, reg_backend, paths.task, motion_model)
        generated.extend([paths.anat_dir / "README.md", paths.func_task_dir / "README.md"])

    return generated
