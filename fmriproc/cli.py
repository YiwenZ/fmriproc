from __future__ import annotations

import argparse
from pathlib import Path

from .bids_scan import choose_duplicate_bold, read_bids_json, resolve_t1_for_session, scan_bids
from .config import load_config
from .dataset_qc import write_dataset_qc
from .manifest import write_tsv
from .models import SelectedInputs
from .paths import MetricPaths, PrepPaths
from .runner import Runner
from .software import check_software, fail_on_required
from .steps import ANAT_STEPS, BOLD_STEPS


def _select_steps_for_backend(cfg: dict) -> tuple:
    """Return (anat_steps, bold_steps) based on runtime.backend config."""
    backend = cfg.get("runtime", {}).get("backend", "afni")
    if backend == "fsl":
        from .fsl_steps import FSL_ANAT_STEPS, FSL_BOLD_STEPS

        return FSL_ANAT_STEPS, FSL_BOLD_STEPS
    return ANAT_STEPS, BOLD_STEPS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fmriproc")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset_qc = subparsers.add_parser("dataset-qc")
    dataset_qc.add_argument("--bids-dir", required=True)
    dataset_qc.add_argument("--prep-dir", required=True)
    dataset_qc.add_argument("--config", default="configs/defaults.yaml")

    preprocess = subparsers.add_parser("preprocess")
    preprocess.add_argument("--bids-dir", required=True)
    preprocess.add_argument("--prep-dir", required=True)
    preprocess.add_argument("--subject", required=True)
    preprocess.add_argument("--session", required=True)
    preprocess.add_argument("--task", required=True)
    preprocess.add_argument("--config", default="configs/defaults.yaml")
    preprocess.add_argument("--from-step")
    preprocess.add_argument("--steps")
    preprocess.add_argument("--skip-steps")
    preprocess.add_argument("--force-step")
    preprocess.add_argument("--dry-run", action="store_true")

    qc_report = subparsers.add_parser("qc-report")
    qc_report.add_argument("--prep-dir", required=True)
    qc_report.add_argument("--subject", required=True)
    qc_report.add_argument("--session", required=True)
    qc_report.add_argument("--task", required=True)
    qc_report.add_argument("--config", default="configs/defaults.yaml")

    preprocess_summary = subparsers.add_parser("preprocess-summary")
    preprocess_summary.add_argument("--prep-dir", required=True)

    plot_matrix = subparsers.add_parser("plot-matrix")
    plot_matrix.add_argument("--matrix", required=True)
    plot_matrix.add_argument("--atlas", required=True)
    plot_matrix.add_argument("--output", required=True)
    plot_matrix.add_argument("--threshold", type=float, default=0)
    plot_matrix.add_argument("--mode", choices=["none", "positive", "negative", "abs", "both"], default="both")
    plot_matrix.add_argument("--title")
    plot_matrix.add_argument("--hide-diagonal", action="store_true")
    plot_matrix.add_argument("--transparent", action="store_true", default=True)
    plot_matrix.add_argument("--opaque", dest="transparent", action="store_false")
    plot_matrix.add_argument("--colorbar", action="store_true")
    plot_matrix.add_argument("--block-linewidth", type=float, default=0.85)
    plot_matrix.add_argument("--hemisphere-linewidth", type=float, default=1.1)
    plot_matrix.add_argument("--colorbar-fraction", type=float, default=0.036)
    plot_matrix.add_argument("--colorbar-pad", type=float, default=0.02)
    plot_matrix.add_argument("--font-family", default="Times New Roman")
    plot_matrix.add_argument("--dpi", type=int, default=300)
    plot_matrix.add_argument("--order", choices=["original", "network"], default="original")
    plot_matrix.add_argument("--networks", default="")

    features = subparsers.add_parser("features")
    features.add_argument("--prep-dir", required=True)
    features.add_argument("--subject", required=True)
    features.add_argument("--session", required=True)
    features.add_argument("--task", required=True)
    features.add_argument("--metrics", default="")
    features.add_argument("--atlas")
    features.add_argument("--config", default="configs/defaults.yaml")
    features.add_argument("--dry-run", action="store_true")
    features.add_argument("--force", action="store_true")

    group_mask = subparsers.add_parser("group-mask")
    group_mask.add_argument("--prep-dir", required=True)
    group_mask.add_argument("--config", default="configs/defaults.yaml")
    group_mask.add_argument("--dry-run", action="store_true")
    group_mask.add_argument("--force", action="store_true")

    features_dataset = subparsers.add_parser("features-dataset")
    features_dataset.add_argument("--prep-dir", required=True)
    features_dataset.add_argument("--tasks", default="")
    features_dataset.add_argument("--metrics", default="")
    features_dataset.add_argument("--atlas")
    features_dataset.add_argument("--subjects-file")
    features_dataset.add_argument("--sessions", default="")
    features_dataset.add_argument("--config", default="configs/defaults.yaml")
    features_dataset.add_argument("--run-missing", action="store_true")
    features_dataset.add_argument("--dry-run", action="store_true")
    features_dataset.add_argument("--force", action="store_true")

    submit_hpc = subparsers.add_parser("submit-hpc")
    submit_hpc.add_argument("--bids-dir", required=True)
    submit_hpc.add_argument("--prep-dir", required=True)
    submit_hpc.add_argument("--subjects-file", required=True)
    submit_hpc.add_argument("--sessions", required=True)
    submit_hpc.add_argument("--tasks", default="movieA,movieB,movieC")
    submit_hpc.add_argument("--config", default="configs/defaults.yaml")
    submit_hpc.add_argument("--scheduler", choices=["lsf", "slurm"])
    submit_hpc.add_argument("--dry-run", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "dataset-qc":
        cfg = load_config(args.config)
        index = scan_bids(Path(args.bids_dir))
        write_dataset_qc(index, Path(args.prep_dir), cfg["input_selection"]["expected_volumes"])
        return 0

    if args.command == "qc-report":
        cfg = load_config(args.config)
        from .qc_report import generate_qc_report

        paths = PrepPaths(Path(args.prep_dir), args.subject, args.session, args.task)
        generated = generate_qc_report(paths, cfg)
        for path in generated:
            print(f"[QC] {path}")
        return 0

    if args.command == "preprocess-summary":
        from .preprocess_summary import write_preprocess_summary

        outputs = write_preprocess_summary(Path(args.prep_dir))
        print(f"[PREPROCESS_SUMMARY] {outputs.preprocess_qc_tsv}")
        print(f"[PREPROCESS_SUMMARY] {outputs.assets_tsv}")
        print(f"[PREPROCESS_SUMMARY] {outputs.summary_json}")
        return 0

    if args.command == "plot-matrix":
        from .matrix_plot import plot_atlas_matrix

        result = plot_atlas_matrix(
            matrix_path=Path(args.matrix),
            atlas=args.atlas,
            output_path=Path(args.output),
            threshold=args.threshold,
            mode=args.mode,
            title=args.title,
            hide_diagonal=args.hide_diagonal,
            transparent=args.transparent,
            show_colorbar=args.colorbar,
            block_linewidth=args.block_linewidth,
            hemisphere_linewidth=args.hemisphere_linewidth,
            colorbar_fraction=args.colorbar_fraction,
            colorbar_pad=args.colorbar_pad,
            font_family=args.font_family,
            dpi=args.dpi,
            order=args.order,
            networks=[name.strip() for name in args.networks.split(",") if name.strip()],
        )
        print(f"[PLOT_MATRIX] {result.output_path}")
        return 0

    if args.command == "preprocess":
        return _run_pipeline(args)

    if args.command == "features":
        return _run_metrics(args)

    if args.command == "group-mask":
        return _run_group_mask(args)

    if args.command == "features-dataset":
        return _run_metrics_dataset(args)

    if args.command == "submit-hpc":
        from .hpc import submit_hpc_jobs

        return submit_hpc_jobs(args)

    raise ValueError(f"Unknown command: {args.command}")


def _run_group_mask(args) -> int:
    from .group_mask import build_group_mask

    cfg = load_config(args.config)
    result = build_group_mask(Path(args.prep_dir), cfg, force=args.force, dry_run=args.dry_run)
    for command in result.commands:
        print("[GROUP_MASK]", " ".join(command))
    print(f"[GROUP_MASK] {result.paths.thr70_mask}")
    print(f"[GROUP_MASK] {result.paths.thr100_mask}")
    print(f"[GROUP_MASK] {result.paths.inputs_tsv}")
    for warning in result.qc_warnings:
        print(f"[WARN] {warning}")
    return 0


def _run_metrics(args) -> int:
    from .atlas_registry import atlas_names_from_config, parse_atlas_names, resolve_atlases
    from .metric_selection import parse_metrics
    from .metrics_inputs import resolve_final_mni_bold
    from .metrics_qc import write_metrics_qc
    from .netcorr import run_3dnetcorr
    from .netcorr_outputs import standardize_netcorr_outputs

    cfg = load_config(args.config)
    paths = PrepPaths(Path(args.prep_dir), args.subject, args.session, args.task)
    metric_paths = MetricPaths(paths)
    metric_paths.ensure_dirs()

    selected_metrics = parse_metrics(args.metrics, cfg)
    if "gradient" in selected_metrics:
        raise SystemExit(
            "Gradient metrics are dataset-level. Use: fmriproc features-dataset --metrics gradient"
        )

    smoothed = bool(cfg.get("metrics", {}).get("input", {}).get("smoothed", False))
    bold_input = resolve_final_mni_bold(paths, cfg, smoothed=smoothed)
    requested_names = parse_atlas_names(args.atlas) or atlas_names_from_config(cfg)
    atlases = resolve_atlases(requested_names)

    generated: list[Path] = []

    # ── FC metrics ──
    if "fc" in selected_metrics:
        for atlas in atlases:
            try:
                result = run_3dnetcorr(
                    bold_path=bold_input.path,
                    atlas=atlas,
                    metric_paths=metric_paths,
                    cfg=cfg,
                    dry_run=args.dry_run,
                    force=args.force,
                )
                print("[METRICS]", " ".join(result.command))
                if args.dry_run:
                    generated.extend([result.netcc_path, result.netts_path, result.log_path])
                    continue

                standardized = standardize_netcorr_outputs(
                    netts_path=result.netts_path,
                    netcc_path=result.netcc_path,
                    timeseries_tsv=metric_paths.timeseries_output(atlas.name),
                    fc_correlation_tsv=metric_paths.fc_correlation_output(atlas.name),
                    fc_fisherz_tsv=metric_paths.fc_fisherz_output(atlas.name),
                    fc_long_tsv=metric_paths.fc_long_output(atlas.name),
                    subject=args.subject,
                    session=args.session,
                    task=args.task,
                    atlas=atlas.name,
                    label_path=atlas.label_path,
                )
                qc_tsv, metadata_json = write_metrics_qc(metric_paths, atlas.name, result, standardized, cfg)
                generated.extend(
                    [
                        standardized.timeseries_tsv,
                        standardized.fc_correlation_tsv,
                        standardized.fc_fisherz_tsv,
                        standardized.fc_long_tsv,
                        qc_tsv,
                        metadata_json,
                    ]
                )
            except Exception as exc:
                raise SystemExit(f"Metrics (FC) failed for atlas {atlas.name}: {exc}") from exc

    # ── Regional metrics (ALFF / fALFF / ReHo) ──
    regional_set = set(selected_metrics) & {"alff", "falff", "reho"}
    if regional_set:
        from .regional_metrics import run_regional_metrics

        result = run_regional_metrics(
            bold_path=bold_input.path,
            metric_paths=metric_paths,
            atlases=atlases,
            selected_metrics=selected_metrics,
            cfg=cfg,
            dry_run=args.dry_run,
            force=args.force,
        )
        generated.extend(result["output_paths"])

    # ── Graph metrics ──
    if "graph" in selected_metrics:
        from .graph_metrics import run_graph_metrics

        result = run_graph_metrics(
            metric_paths=metric_paths,
            atlases=atlases,
            cfg=cfg,
            dry_run=args.dry_run,
            force=args.force,
        )
        generated.extend(result["output_paths"])

    for path in generated:
        print(f"[METRICS] {path}")
    return 0


def _run_metrics_dataset(args) -> int:
    from .atlas_registry import atlas_names_from_config, parse_atlas_names, resolve_atlases
    from .metric_selection import parse_metrics
    from .metrics_dataset import run_dataset_metrics

    cfg = load_config(args.config)
    prep_dir = Path(args.prep_dir)

    selected_metrics = parse_metrics(args.metrics, cfg)
    requested_names = parse_atlas_names(args.atlas) or atlas_names_from_config(cfg)
    atlases = resolve_atlases(requested_names)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] if args.tasks else None
    sessions = [s.strip() for s in args.sessions.split(",") if s.strip()] if args.sessions else None

    subjects: list[str] | None = None
    if args.subjects_file:
        subjects = Path(args.subjects_file).read_text(encoding="utf-8").strip().splitlines()
        subjects = [s.strip() for s in subjects if s.strip()]

    if args.dry_run:
        print(f"[METRICS_DATASET] DRY RUN — prep_dir={prep_dir}")
        print(f"[METRICS_DATASET] metrics={selected_metrics}")
        print(f"[METRICS_DATASET] atlases={[a.name for a in atlases]}")
        if tasks:
            print(f"[METRICS_DATASET] tasks={tasks}")
        if subjects:
            print(f"[METRICS_DATASET] subjects={subjects}")
        if sessions:
            print(f"[METRICS_DATASET] sessions={sessions}")

    result = run_dataset_metrics(
        prep_dir=prep_dir,
        cfg=cfg,
        atlases=atlases,
        tasks=tasks,
        subjects=subjects,
        sessions=sessions,
        selected_metrics=selected_metrics,
        run_missing=args.run_missing,
        dry_run=args.dry_run,
        force=args.force,
    )

    included_count = len(result["included_runs"])
    excluded_count = len([r for r in result["runs"] if r.status == "excluded"])
    pending_count = len([r for r in result["runs"] if r.status == "pending"])
    print(f"[METRICS_DATASET] runs: {len(result['runs'])} discovered, "
          f"{included_count} included, {excluded_count} excluded, {pending_count} pending")

    for run in result["runs"]:
        if run.status == "excluded":
            print(f"[METRICS_DATASET] excluded: {run.subject}/{run.session}/task-{run.task} — {run.reason}")

    for path in result["generated_paths"]:
        print(f"[METRICS_DATASET] {path}")

    return 0


def _resolve_template_path(cfg: dict, package_dir: Path) -> str:
    """Resolve the default MNI template path from config, relative to the package root."""
    templates_cfg = cfg.get("templates", {})
    resolution = templates_cfg.get("mni152_default_resolution", "3mm")
    key = f"mni152_{resolution}_brain"
    rel_path = templates_cfg.get(key, "")

    template_path = None
    if rel_path:
        template_path = str(package_dir.parent / rel_path)
        if Path(template_path).exists():
            return template_path

    # If config path doesn't exist, try default relative to package.
    fallback = str(package_dir / "resources" / "templates" / "MNI152" / f"MNI152_T1_{resolution}_brain.nii.gz")
    if Path(fallback).exists():
        return fallback

    err_msg = f"MNI template not found. Config key 'templates.{key}' points to '{rel_path}'."
    if template_path:
        err_msg += f" Looked at '{template_path}'."
    err_msg += f" Fallback '{fallback}' also failed. Place the template or update the config."

    raise SystemExit(err_msg)


def _write_nuisance_metadata(paths: PrepPaths, model: str, matrix: RegressorMatrix, bold_path: str, masks: dict[str, str]) -> None:
    import json

    metadata = {
        "model": model,
        "bold_path": bold_path,
        "masks": masks,
        "n_columns": len(matrix.names),
        "columns": matrix.names,
    }
    paths.qc_output("nuisance_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_nuisance_regressors(paths: PrepPaths, cfg: dict) -> None:
    from .motion import read_motion_1d
    from .nuisance import (
        RegressorMatrix,
        build_motion_regressors,
        compute_mask_mean_regressor,
        merge_regressor_matrices,
        resolve_nuisance_model,
        write_regressor_text,
        write_regressor_tsv,
    )

    afni_motion = paths.qc_output("motion_parameters.1D")
    fsl_motion = paths.qc_output("motion_parameters.par")

    if afni_motion.exists():
        motion_rows = read_motion_1d(str(afni_motion))
    elif fsl_motion.exists():
        motion_rows = read_motion_1d(str(fsl_motion))
    else:
        print("[WARN] No motion parameter file found; skipping nuisance regressor generation.")
        return

    nuisance_cfg = cfg.get("bold", {}).get("nuisance", {})
    model = resolve_nuisance_model(nuisance_cfg)
    motion_model = "basic_6" if model == "acompcor" else model
    motion_matrix = build_motion_regressors(motion_rows, model=motion_model)

    backend = cfg.get("runtime", {}).get("backend", "afni")
    if backend == "fsl":
        from .fsl_step_args import _post_motion_fsl
        bold_path = str(_post_motion_fsl(paths, cfg))
    else:
        from .step_args import _post_motion_bold_input
        bold_path = str(_post_motion_bold_input(paths, cfg))

    extra_matrices: list[RegressorMatrix] = []
    mask_file_map = {
        "wm": str(paths.masks_dir / "t1_wm_mask_to_epi.nii.gz"),
        "csf": str(paths.masks_dir / "t1_csf_mask_to_epi.nii.gz"),
    }

    if bool(nuisance_cfg.get("wm", False)):
        extra_matrices.append(compute_mask_mean_regressor(bold_path, mask_file_map["wm"], "wm_mean"))
    if bool(nuisance_cfg.get("csf", False)):
        extra_matrices.append(compute_mask_mean_regressor(bold_path, mask_file_map["csf"], "csf_mean"))

    acompcor_results = []
    if model == "acompcor":
        from .acompcor import compute_acompcor_regressors, write_acompcor_sidecar

        acompcor_cfg = nuisance_cfg.get("acompcor", {})
        n_components = int(acompcor_cfg.get("n_components", 5))
        masks_to_use = acompcor_cfg.get("masks", ["wm", "csf"])
        for mask_label in masks_to_use:
            mask_path = mask_file_map.get(mask_label)
            if mask_path is None:
                raise ValueError(f"aCompCor: unknown mask '{mask_label}'. Must be 'wm' or 'csf'.")
            result = compute_acompcor_regressors(bold_path, mask_path, n_components, label=mask_label)
            extra_matrices.append(result.matrix)
            acompcor_results.append(result)
            write_acompcor_sidecar(result, mask_label, n_components, paths.qc_output(f"acompcor_{mask_label}_metadata.json"))

    matrix = merge_regressor_matrices([motion_matrix] + extra_matrices)

    write_regressor_tsv(matrix, paths.qc_output("nuisance_regressors.tsv"))
    write_regressor_text(matrix, paths.qc_output("nuisance_regressors.1D"))
    write_regressor_text(matrix, paths.qc_output("nuisance_regressors.txt"))
    _write_nuisance_metadata(paths, model, matrix, bold_path, mask_file_map)


def _run_pipeline(args) -> int:
    cfg = load_config(args.config)
    bids_dir = Path(args.bids_dir)
    prep_dir = Path(args.prep_dir)
    subject = args.subject
    session = args.session
    task = args.task

    # Scan BIDS and select inputs.
    index = scan_bids(bids_dir)

    # Find BOLD candidates for this subject/session/task.
    bold_candidates = [
        item for item in index.bold
        if item.subject == subject and item.session == session and item.task == task
    ]
    if not bold_candidates:
        raise SystemExit(f"No BOLD found for {subject}/{session}/task-{task}")

    selected_bold, unselected_bold, bold_reason = choose_duplicate_bold(bold_candidates)

    # Resolve T1.
    selected_t1, t1_reason = resolve_t1_for_session(subject, session, index.t1w)

    # Find SBRef and fieldmaps.
    sbref_candidates = [
        item for item in index.sbref
        if item.subject == subject and item.session == session and item.task == task
    ]
    sbref = sbref_candidates[0] if sbref_candidates else None

    fmap_ap = next(
        (item for item in index.fmap_ap if item.subject == subject and item.session == session),
        None,
    )
    fmap_pa = next(
        (item for item in index.fmap_pa if item.subject == subject and item.session == session),
        None,
    )

    inputs = SelectedInputs(
        subject=subject,
        session=session,
        task=task,
        bold=selected_bold,
        t1=selected_t1,
        sbref=sbref,
        fmap_ap=fmap_ap,
        fmap_pa=fmap_pa,
        unselected_bold=tuple(unselected_bold),
        t1_fallback_reason=t1_reason,
        bold_selection_reason=bold_reason,
    )

    # Resolve template path (P0 template check).
    package_dir = Path(__file__).parent
    template_path = _resolve_template_path(cfg, package_dir)

    # Resolve TR from BIDS JSON sidecar if configured as "auto" (P0 TR normalization).
    if cfg.get("bold", {}).get("tr") == "auto":
        sidecar = read_bids_json(selected_bold.path)
        tr_value = sidecar.get("RepetitionTime")
        if tr_value is not None and isinstance(tr_value, (int, float)):
            cfg["bold"]["tr"] = float(tr_value)
        else:
            raise SystemExit(
                f"bold.tr is 'auto' but no RepetitionTime found in BIDS sidecar for {selected_bold.path}. "
                f"Set bold.tr explicitly in config, or add RepetitionTime to the BIDS JSON."
            )

    # Normalize lowpass null → sentinel for AFNI/ANTs backend only.
    # FSL backend preserves null (maps to lp_sigma=-1 in fsl_step_args.py).
    backend = cfg.get("runtime", {}).get("backend", "afni")
    if backend != "fsl" and cfg.get("bold", {}).get("lowpass") is None:
        cfg["bold"]["lowpass"] = 99999

    # Check software availability before non-dry-run (P1 software check).
    if not args.dry_run:
        sw_required, sw_warnings = check_software(cfg)
        for warning in sw_warnings:
            print(f"[WARN] {warning}")
        fail_on_required(sw_required)

    paths = PrepPaths(prep_dir, subject, session, task)
    paths.ensure_dirs()

    # Write run-level input QC records.
    _write_input_qc(paths, inputs)

    # Build runner and determine steps to execute.
    runner = Runner(
        paths=paths,
        inputs=inputs,
        config=cfg,
        dry_run=args.dry_run,
        template_path=template_path,
    )

    anat_steps, bold_steps = _select_steps_for_backend(cfg)
    all_steps = list(anat_steps) + list(bold_steps)
    step_map = {step.step_id: step for step in all_steps}

    # Determine which steps to run.
    if args.steps:
        step_ids = [value.strip() for value in args.steps.split(",")]
    else:
        step_ids = [step.step_id for step in all_steps]

    # Apply --from-step: skip steps before the named step.
    if args.from_step:
        try:
            from_idx = step_ids.index(args.from_step)
            step_ids = step_ids[from_idx:]
        except ValueError:
            raise SystemExit(f"--from-step {args.from_step} not found in step list")

    skip_ids = set()
    if args.skip_steps:
        skip_ids = {value.strip() for value in args.skip_steps.split(",")}

    force_ids = set()
    if args.force_step:
        force_ids = {value.strip() for value in args.force_step.split(",")}

    # Execute steps.
    for step_id in step_ids:
        if step_id in skip_ids:
            from .manifest import append_status

            append_status(
                paths.log_task_dir / "status.jsonl",
                {
                    "subject": subject,
                    "session": session,
                    "task": task,
                    "step_id": step_id,
                    "status": "skipped",
                    "reason": "skipped_by_user",
                    "command": [],
                },
            )
            continue

        step = step_map[step_id]
        force = step_id in force_ids

        # Generate nuisance regressor matrix before bold_s08.
        if not args.dry_run and step_id == "bold_s08_nuisance_regression":
            _write_nuisance_regressors(paths, cfg)

        try:
            result = runner.run_step(step, force=force)
        except RuntimeError as exc:
            print(f"Step {step_id} failed: {exc}")
            return 1
        if result.status == "failed":
            print(f"Step {step_id} failed: {result.reason}")
            return 1

    # Write motion QC.
    _write_motion_qc(paths, inputs, cfg)

    # Auto-generate QC report after successful run.
    if not args.dry_run and cfg.get("qc", {}).get("report_after_run", True):
        from .qc_report import generate_qc_report

        generated = generate_qc_report(paths, cfg)
        for path in generated:
            print(f"[QC] {path}")

    return 0


def _write_input_qc(paths: PrepPaths, inputs: SelectedInputs) -> None:
    """Write run-level input QC records."""
    qc_dir = paths.qc_task_dir
    qc_dir.mkdir(parents=True, exist_ok=True)

    # selected_inputs.tsv
    write_tsv(
        qc_dir / "selected_inputs.tsv",
        ["subject", "session", "task", "bold_path", "t1_path", "t1_fallback_reason",
         "bold_selection_reason", "sbref_path", "fmap_ap_path", "fmap_pa_path"],
        [{
            "subject": inputs.subject,
            "session": inputs.session,
            "task": inputs.task,
            "bold_path": str(inputs.bold.path),
            "t1_path": str(inputs.t1.path),
            "t1_fallback_reason": inputs.t1_fallback_reason,
            "bold_selection_reason": inputs.bold_selection_reason,
            "sbref_path": str(inputs.sbref.path) if inputs.sbref else "",
            "fmap_ap_path": str(inputs.fmap_ap.path) if inputs.fmap_ap else "",
            "fmap_pa_path": str(inputs.fmap_pa.path) if inputs.fmap_pa else "",
        }],
    )

    # anat_source.tsv
    write_tsv(
        qc_dir / "anat_source.tsv",
        ["subject", "session", "selected_t1_session", "selected_t1_path", "fallback_reason"],
        [{
            "subject": inputs.subject,
            "session": inputs.session,
            "selected_t1_session": inputs.t1.session or "",
            "selected_t1_path": str(inputs.t1.path),
            "fallback_reason": inputs.t1_fallback_reason,
        }],
    )

    # input_qc.tsv
    write_tsv(
        qc_dir / "input_qc.tsv",
        ["subject", "session", "task", "has_t1", "has_bold", "has_sbref", "has_fmap_ap", "has_fmap_pa",
         "n_unselected_bold", "unselected_bold_paths"],
        [{
            "subject": inputs.subject,
            "session": inputs.session,
            "task": inputs.task,
            "has_t1": "True",
            "has_bold": "True",
            "has_sbref": str(inputs.sbref is not None),
            "has_fmap_ap": str(inputs.fmap_ap is not None),
            "has_fmap_pa": str(inputs.fmap_pa is not None),
            "n_unselected_bold": len(inputs.unselected_bold),
            "unselected_bold_paths": ";".join(str(item.path) for item in inputs.unselected_bold),
        }],
    )


def _write_motion_qc(paths: PrepPaths, inputs: SelectedInputs, cfg: dict) -> None:
    """Write motion QC summaries if motion files exist."""
    backend = cfg.get("runtime", {}).get("backend", "afni")

    # FSL backend uses MCFLIRT .par format (rotations already in radians).
    motion_par = paths.qc_output("motion_parameters.par")
    if backend == "fsl" and motion_par.exists():
        from .motion import summarize_fsl_motion

        try:
            summary = summarize_fsl_motion(str(motion_par))
        except Exception:
            return
        summary["subject"] = inputs.subject
        summary["session"] = inputs.session
        summary["task"] = inputs.task
        write_tsv(
            paths.qc_output("motion_summary.tsv"),
            list(summary.keys()),
            [summary],
        )
        return

    motion_1d = paths.qc_output("motion_parameters.1D")
    if not motion_1d.exists():
        return

    from .motion import read_motion_1d, summarize_motion

    try:
        rows = read_motion_1d(str(motion_1d))
    except Exception:
        return

    bold_vols = inputs.bold.volume_count or 0
    n_dummy = cfg.get("bold", {}).get("dummy_scans", 10)
    summary = summarize_motion(
        subject=inputs.subject,
        session=inputs.session,
        task=inputs.task,
        n_volumes_raw=bold_vols,
        n_dummy_removed=n_dummy,
        motion_rows=rows,
    )

    write_tsv(
        paths.qc_output("motion_summary.tsv"),
        list(summary.keys()),
        [summary],
    )

    from .motion import compute_power_fd

    fd = compute_power_fd(rows)
    fd_rows = [{"volume": i, "fd_power": value} for i, value in enumerate(fd)]
    write_tsv(
        paths.qc_output("motion_fd_power.tsv"),
        ["volume", "fd_power"],
        fd_rows,
    )


if __name__ == "__main__":
    raise SystemExit(main())
