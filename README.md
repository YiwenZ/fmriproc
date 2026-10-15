# fmriproc

### Why another fMRI preprocessing pipeline?
There is no shortage of fMRI preprocessing pipelines these days. If you are looking for a well-established and widely accepted workflow, **fMRIPrep** is probably one of the best choices out there. **SPM** is another great option, especially if you prefer a more traditional approach.

But neither is perfect, and each comes with its own trade-offs.

For example, registration in SPM sometimes requires extra attention. **fMRIPrep**, while widely regarded as a standard preprocessing solution, can be surprisingly demanding in terms of computing resources and storage. A dataset that originally takes up just a few hundred megabytes can end up generating tens of gigabytes of derivatives and intermediate files, depending on the configuration. Since fMRIPrep integrates multiple neuroimaging software packages, running it can also require a fairly complex software environment, often on a computing cluster. Even if you have the computing power, storing all those outputs can become a headache.

Another consideration is anatomical processing. Skull stripping and tissue segmentation are essential steps, and FreeSurfer provides excellent tools for these tasks. However, a full anatomical processing workflow can be time-consuming, especially when a simpler solution would be sufficient for your analysis. For example, if all I need is tissue segmentation, why not just use FreeSurfer's `run_samseg` directly instead of the `recon-all`?

Then there is the question of software dependencies. Many preprocessing workflows combine commands from AFNI, FSL, ANTs, and other packages. While this makes it possible to take advantage of the strengths of different tools, switching between software packages can also introduce additional complexity in image orientations, coordinate conventions, and transformations.

Personally, I think **AFNI and FSL are both powerful enough to handle a complete fMRI preprocessing workflow on their own** for many common applications. ANTs offers excellent nonlinear registration, but AFNI and FSL also provide reliable registration tools that are perfectly suitable for many preprocessing tasks. This becomes particularly relevant when working on an HPC cluster, where you may not have permission to install software yourself. If you need Docker or other containers to manage your environment, getting multiple neuroimaging packages to work together can be another challenge.

And finally, there is something that I think deserves more attention: **the outputs themselves**. If you are new to neuroimaging, figuring out what all those intermediate files mean, why they have different names, and in what order the processing steps were performed can be frustrating. Sometimes, understanding the outputs feels almost as complicated as understanding the preprocessing itself.

### So, what if we made things simpler?

That was the motivation behind **fMRIProc**. I wanted a pipeline where I could simply choose **AFNI or FSL as the main preprocessing backend**, run the entire workflow, and get outputs organized in the order they were generated:
bold_s01_*
bold_s02_*
...
or
t1_s01_*
t1_s02_*
...
No need to guess what happened along the way. Just follow the steps, and you can easily see how your data were processed.

I also wanted the flexibility to use additional software **only when I actually need it**. Have FreeSurfer installed but don't want to run a full FreeSurfer workflow? Just use `run_samseg` for tissue segmentation (or just use AFNI/FSL to do the job). Have ANTs available? Great, you can use it for registration. Otherwise, AFNI or FSL can handle the job without requiring extra dependencies.

In short, fMRIProc brings together the preprocessing steps I regularly use and have tested in my own neuroimaging research. You can choose an **AFNI-based or FSL-based workflow**, keep the processing steps organized, and obtain clearly labeled outputs that make it easy to understand what happened to your data.

### Beyond preprocessing

Of course, preprocessing is only part of the story.

In my own research, I often need to examine head motion, check registration quality, identify participants who pass quality-control criteria, and prepare datasets for group-level analyses. So I also included several **quality-control (QC) and analysis utilities** that I find particularly useful. These tools can help you review motion statistics and registration quality, select participants based on QC results, and generate group-level masks for subsequent analyses. The pipeline also includes functions for calculating commonly used resting-state fMRI metrics, including **functional connectivity (FC), ALFF, fALFF, and ReHo**. Since the workflow is orchestrated in Python, I have also integrated tools for **graph-theoretical analysis (`python-igraph`) and functional connectivity gradients (`BrainSpace`)**, making it easier to move from preprocessing to network-level analyses. One small feature I particularly like is the FC matrix visualization function. It supports common functional brain parcellations, including Schaefer-400 with Yeo's 7-network organization and the Gordon atlas, and can display network-specific color bars alongside the connectivity matrix. It is a simple addition, but it makes connectivity figures much easier to read and present.

### A final note
fMRIProc is not intended to replace fMRIPrep, SPM, or any other established preprocessing software. Those are excellent tools, and they remain great choices for many studies. This project is simply my attempt to make fMRI preprocessing more flexible, easier to follow, and less demanding in terms of software dependencies and storage, based on the workflows I have found useful in my own research.

If you are looking for a relatively lightweight AFNI- or FSL-based preprocessing workflow, or just want to better understand what happens to your images at each step, I hope you will find something useful here.


## Deployable Folder

This folder is the deployable pipeline package:

```text
fmri_pipeline/
  fmriproc/
    bash/               # Backend Bash scripts (afni_steps.sh, fsl_steps.sh, ants_steps.sh, common.sh)
    resources/          # MNI templates
    cli.py              # CLI entry point (fmriproc)
    config.py           # Config loading with defaults and deep-merge
    runner.py           # Step execution engine
    paths.py            # Output directory layout (PrepPaths)
    models.py           # Data models (BidsFile, StepDefinition, SelectedInputs, …)
    steps.py            # AFNI/ANTs step registry
    fsl_steps.py        # FSL step registry
    step_args.py        # AFNI/ANTs argument resolver
    fsl_step_args.py    # FSL argument resolver
    bids_scan.py        # BIDS directory scanner
    software.py         # Software availability checker
    motion.py           # Motion parameter reader, FD computation, plotting
    acompcor.py         # aCompCor PCA-based noise component extraction
    nuisance.py         # Nuisance regressor builder (none/basic_6/friston_24/acompcor) and mask mean extraction
    qc_images.py        # NIfTI slice montage and overlay renderer
    qc_report.py        # QC report orchestrator
    readme_writer.py    # Per-directory README generator
    hpc.py              # HPC job submission (LSF + SLURM)
    manifest.py         # Status and command logging
    dataset_qc.py       # Dataset-level QC
    atlas_registry.py   # Atlas registry (Schaefer, Gordon)
    metrics_inputs.py   # Final MNI BOLD resolution
    netcorr.py          # AFNI 3dNetCorr runner
    netcorr_outputs.py  # Parse .netcc/.netts → standardized TSV
    metrics_qc.py       # Per-run metrics QC and metadata
    metrics_dataset.py  # Dataset-level FC aggregation
    motion_qc_utils.py  # Shared motion QC helpers
    group_mask.py       # Dataset-level group mask builder
    group_mask_qc.py    # Group mask visual QC
    metric_selection.py # Metric family parsing
    regional_afni.py    # AFNI 3dRSFC / 3dReHo runners
    regional_maps.py    # Within-mask z-scoring
    regional_roi.py     # Atlas ROI summary
    regional_qc.py      # Regional metrics QC + metadata
    regional_metrics.py # Single-run regional metrics orchestrator
    graph_backend_igraph.py # Python-igraph graph theory backend
    graph_outputs.py    # Graph output helpers + dataset aggregation
    graph_metrics.py    # Single-run graph metrics orchestrator
    gradient_backend.py # BrainSpace gradient backend
    gradient_outputs.py # Gradient output helpers + dataset aggregation
    gradient_dataset.py # Dataset-level gradient reference/alignment workflow
  configs/
    defaults.yaml       # Default pipeline settings
    acompcor_enabled.yaml # Example settings with aCompCor nuisance regression
  pyproject.toml
  README.md
```

Copy `fmri_pipeline/` to a server when you want to run the pipeline elsewhere.
Research data and run outputs are not included in this package.

## Data Layout

```text
/path/to/dataset/
  RAW_DICOM/
  DICOM/
  NIFTI/              # BIDS input (reads from here)
  Prep/               # Pipeline output (writes only here)
```

The pipeline reads from `NIFTI` by default and writes only to `Prep`.

## Environment

```bash
cd /path/to/fmri_pipeline
conda create -n fmriproc python=3.11
conda activate fmriproc
pip install -e .
```

Dependencies: Python ≥ 3.10, `nibabel`, `numpy`, `matplotlib`, `PyYAML`.
Backend tools (AFNI, ANTs, or FSL) must be installed separately and on `PATH`.

## Architecture

The pipeline separates **step definitions** (what to run), **argument resolution** (what parameters to pass), and **Bash execution** (how to run it).

```
                    ┌─────────────────────────────────────┐
                    │           CLI (cli.py)               │
                    │  fmriproc preprocess|features|features-dataset|…  │
                    └──────────────┬──────────────────────┘
                                   │
              ┌────────────────────┼────────────────────────┐
              │                    │                        │
     ┌────────▼──────┐  ┌──────────▼──────────┐  ┌──────────▼──────────┐
     │  Preprocessing │  │     Metrics          │  │       QC             │
     │  runner.py     │  │  netcorr.py          │  │  qc_report.py        │
     │  steps.py      │  │  netcorr_outputs.py  │  │  qc_images.py        │
     │  fsl_steps.py  │  │  regional_afni.py    │  │  motion.py           │
     │  afni_steps.sh │  │  regional_metrics.py │  │  readme_writer.py    │
     │  fsl_steps.sh  │  │  metrics_dataset.py  │  │  group_mask_qc.py    │
     │  ants_steps.sh │  │  regional_roi.py     │  │                      │
     └────────────────┘  │  regional_maps.py    │  └──────────────────────┘
                         │  metric_selection.py │
                         │  metrics_inputs.py   │
                         │  metrics_qc.py       │
                         │  regional_qc.py      │
                         │  group_mask.py       │
                         └──────────────────────┘
```

### Backend Selection

| Config key | Values | What it controls |
|---|---|---|
| `runtime.backend` | `afni` (default) / `fsl` | Preprocessing toolchain and BOLD step list |
| `runtime.registration_backend` | `ants_if_available` (default) / `ants` / `afni` / `fsl` | Registration method for anat steps |

When `backend: fsl`, both preprocessing and registration use FSL tools exclusively.
When `backend: afni`, preprocessing uses AFNI and registration uses ANTs or AFNI depending on `registration_backend`.

### Anatomy Segmentation

T1 tissue masks default to FreeSurfer SAMSEG when `run_samseg` is available.
This does **not** run full FreeSurfer `recon-all`; it only calls `run_samseg`
on the selected raw T1 image and converts SAMSEG labels into the canonical
pipeline masks:

```text
masks/t1_brain_mask.nii.gz
masks/t1_wm_mask.nii.gz
masks/t1_csf_mask.nii.gz
masks/t1_gm_mask.nii.gz
```

If `run_samseg` is unavailable and `anat.segmentation_backend` is
`samseg_if_available`, the pipeline falls back to the native segmentation for
the selected preprocessing backend: AFNI `3dSeg` or FSL `FAST`.

## Pipeline Steps

### AFNI/ANTs Backend (18 steps)

| Anat step | Function | Output |
|---|---|---|
| t1_s01_deoblique | `3drefit -deoblique` | `t1_s01_deoblique.nii.gz` |
| t1_s02_skullstrip | `3dSkullStrip` | `t1_s02_skullstrip.nii.gz` |
| t1_s03_unifize | `3dUnifize` | `t1_s03_unifize.nii.gz` |
| t1_s04_space_mni152_affine | ANTs or AFNI affine | `t1_s04_space_mni152_affine.nii.gz` |
| t1_s05_space_mni152_nonlinear | ANTs or AFNI nonlinear | `t1_s05_space_mni152_nonlinear.nii.gz` |
| t1_s06_segmentation | `3dSeg` | `t1_s06_segmentation.done` |
| t1_s07_masks | Threshold tissue maps | `t1_s07_masks.done` |

| BOLD step | Function | Output |
|---|---|---|
| bold_s01_drop_dummy | `3dTcat` | `bold_s01_drop_dummy.nii.gz` |
| bold_s02_deoblique | `3drefit -deoblique` | `bold_s02_deoblique.nii.gz` |
| bold_s03_motion_correction | `3dvolreg` | `bold_s03_motion_correction.nii.gz` |
| bold_s04_slice_timing* | `3dTshift` | `bold_s04_slice_timing.nii.gz` |
| bold_s05_epi_ref_unifize | `3dUnifize` on ref vol | `bold_s05_epi_ref_unifize.nii.gz` |
| bold_s06_epi_to_anat | ANTs `antsRegistration` | `bold_s06_epi_to_anat.done` |
| bold_s07_masks_to_epi | `3dAllineate` / ANTs | `bold_s07_masks_to_epi.done` |
| bold_s08_nuisance_regression | `3dTproject` | `bold_s08_nuisance_regression.nii.gz` |
| bold_s09_grand_mean_scaling_10000 | `3dcalc` × 10000 | `bold_s09_grand_mean_scaling_10000.nii.gz` |
| bold_s10_space_mni152 | `3dNwarpApply` | `bold_s10_space_mni152.nii.gz` |
| bold_s11_smooth_sm6 | `3dmerge -1blur_fwhm 6` | `bold_s11_smooth_sm6.nii.gz` |

### FSL Backend (18 steps)

| Anat step | Function | Output |
|---|---|---|
| t1_s01_reorient2std | `fslreorient2std` | `t1_s01_reorient2std.nii.gz` |
| t1_s02_skullstrip_bet | `bet -f 0.1 -B -R -m` | `t1_s02_skullstrip_bet.nii.gz` |
| t1_s03_fast_segmentation | `fast` | `t1_s03_fast_segmentation.done` |
| t1_s04_masks | `fslmaths -thr` on PVE maps | `t1_s04_masks.done` |
| t1_s05_space_mni152_affine | `flirt -dof 12` | `t1_s05_space_mni152_affine.nii.gz` |
| t1_s06_space_mni152_nonlinear | `fnirt` | `t1_s06_space_mni152_nonlinear.nii.gz` |

| BOLD step | Function | Output |
|---|---|---|
| bold_s01_reorient2std | `fslreorient2std` | `bold_s01_reorient2std.nii.gz` |
| bold_s02_drop_dummy | `fslroi` | `bold_s02_drop_dummy.nii.gz` |
| bold_s03_ref | `fslroi 0 1` | `bold_s03_ref.nii.gz` |
| bold_s04_motion_correction_mcflirt | `mcflirt` | `bold_s04_motion_correction_mcflirt.nii.gz` |
| bold_s05_slice_timing* | `slicetimer` | `bold_s05_slice_timing.nii.gz` |
| bold_s06_epi_to_anat_bbr | `epi_reg` (BBR) | `bold_s06_epi_to_anat_bbr.done` |
| bold_s07_masks_to_epi | `flirt -applyxfm` | `bold_s07_masks_to_epi.done` |
| bold_s08_nuisance_regression | `fsl_glm` | `bold_s08_nuisance_regression.nii.gz` |
| bold_s09_temporal_filtering | `fslmaths -bptf` | `bold_s09_temporal_filtering.nii.gz` |
| bold_s10_grand_mean_scaling_10000 | `fslmaths -div -mul` | `bold_s10_grand_mean_scaling_10000.nii.gz` |
| bold_s11_space_mni152 | `convertwarp` + `applywarp` | `bold_s11_space_mni152.nii.gz` |
| bold_s12_smooth_sm6 | `fslmaths -kernel gauss -fmean` | `bold_s12_smooth_sm6.nii.gz` |

*\* Slice timing steps are optional — enabled via `bold.slice_timing.enabled: true`.*

## Configuration Files

Configs are YAML files that deep-merge with `config.py` defaults.

| File | Purpose |
|---|---|
| `configs/defaults.yaml` | Full default settings — reference for all available keys |
| `configs/acompcor_enabled.yaml` | Example configuration with aCompCor nuisance regression |

All subject identifiers, session labels, task names, and `/path/to/` paths below are illustrative. Replace them with your own dataset values. Before running, update `dataset.bids_dir` and `dataset.prep_dir` in the selected configuration, along with any task-specific settings. For gradients, replace `metrics.gradient.reference.condition_groups` with the intended grouping (for example, `condition1: [taskA, taskB, taskC]`); the task names below are not automatically added to the supplied configurations. For HPC submission, set `hpc.queue` and `hpc.group` to your cluster's partition/queue and account.

Key config sections:

```yaml
runtime:
  backend: afni                    # afni | fsl
  registration_backend: ants_if_available  # ants_if_available | ants | afni | fsl
  mode: tested                    # tested | test (dry-run marker)

anat:
  segmentation_backend: samseg_if_available  # samseg_if_available | samseg | native | afni | fsl
  samseg:
    threads: 4

bold:
  dummy_scans: 10
  tr: auto                        # auto reads from BIDS sidecar
  highpass: 0.01                  # Hz
  lowpass: null                   # null = disabled
  nuisance:
    model: acompcor               # acompcor | basic_6 | friston_24 | none
    motion_model: basic_6         # legacy fallback; explicit model takes precedence
    wm: false                     # extra WM mean; use true for a motion + mean recipe
    csf: false                    # extra CSF mean; use true for a motion + mean recipe
    acompcor:
      enabled: true
      n_components: 5
      masks: [wm, csf]
  grand_mean: 10000
  smooth_fwhm: 6

qc:
  report_after_run: true          # auto-generate QC after run
  snapshots:
    enabled: true
    n_slices: 9
    overlay_alpha: 0.35
  readme:
    enabled: true
  motion_plot:
    enabled: true

metrics:
  default: [fc]                   # default --metrics value
  atlases:
    default: [Schaefer400_7Networks, Gordon]
    resolution: auto              # auto | 1mm | 2mm | 3mm
  group_mask:
    apply: true
    default_constraint: grp_mask_thr70.nii.gz
    selection:
      require_motion_qc: true
      motion:
        mean_fd_max: 0.5
        fd_outlier_ratio_max: 0.2
  regional:
    frequency:
      low_hz: 0.01
      high_hz: 0.1
    reho:
      neighborhood: 27            # 7 | 19 | 27
    group_mask:
      apply: true
    standardize:
      zscore_within_mask: true
    roi_summary:
      min_roi_coverage: 0.5
  graph:
    backend: python_igraph
    densities: [0.10, 0.15, 0.20, 0.25]
    random_networks: 100         # null graphs for small-world normalisation
    seed: 112                    # base seed (per-run via run_hash strategy)
    seed_strategy: run_hash      # run_hash | fixed
    require_leiden: false
    keep_adjacency: true
  gradient:
    backend: brainspace
    n_components: 3
    approach: dm
    kernel: normalized_angle
    alignment: procrustes
    sparsity: 0.9
    random_state: 42
    reference:
      scope: condition_group
      min_runs_per_group: 2
      unmapped_task_policy: fail
      condition_groups:
        condition1: [taskA, taskB, taskC]
  dataset:
    run_missing: false
    include_existing_metrics_only: true
    inclusion:
      require_group_mask: true
      require_motion_qc: true
```

## CLI Commands

### `fmriproc preprocess` — Run preprocessing

```bash
fmriproc preprocess \
  --bids-dir /path/to/dataset/NIFTI \
  --prep-dir /path/to/dataset/Prep \
  --subject sub-001 \
  --session ses-01 \
  --task taskB \
  --config configs/defaults.yaml
```

Options:

| Flag | Description |
|---|---|
| `--from-step STEP_ID` | Start from a specific step (skips prior steps) |
| `--steps STEP_IDS` | Run only comma-separated steps |
| `--skip-steps STEP_IDS` | Skip comma-separated steps |
| `--force-step STEP_IDS` | Force re-run even if outputs exist |
| `--dry-run` | Log commands without executing |

Choose the backend by editing the `runtime` section of your configuration:

| Processing path | `runtime.backend` | `runtime.registration_backend` |
|---|---|---|
| AFNI with ANTs registration | `afni` | `ants` |
| AFNI with automatic ANTs fallback | `afni` | `ants_if_available` |
| AFNI with AFNI registration | `afni` | `afni` |
| FSL preprocessing and registration | `fsl` | `fsl` |

With `ants_if_available`, registration uses ANTs when its required tools are available and otherwise uses AFNI. Both supplied configurations use this setting. To use the aCompCor example, pass `--config configs/acompcor_enabled.yaml` with the same input and output arguments shown above.

### `fmriproc qc-report` — Generate visual QC

```bash
fmriproc qc-report \
  --prep-dir /path/to/dataset/Prep \
  --subject sub-001 \
  --session ses-01 \
  --task taskB \
  --config configs/defaults.yaml
```

Generates (see Output Layout below):
- T1 and EPI slice montage PNGs
- EPI-to-T1 and EPI-to-MNI overlay PNGs
- Motion parameter plot PNG
- `anat/README.md` and `func/task-<task>/README.md`

QC report runs automatically after `fmriproc preprocess` unless `qc.report_after_run: false`.

### `fmriproc preprocess-summary` — Summarize preprocessing QC

```bash
fmriproc preprocess-summary --prep-dir /path/to/dataset/Prep
```

Scans completed run-level QC/log outputs and writes dataset-level preprocessing QC tables under `Prep/derivatives/qc/`:

- `preprocess_qc.tsv`
- `preprocess_qc_assets.tsv`
- `preprocess_qc_summary.json`

### `fmriproc features` — Extract FC and regional metrics

```bash
fmriproc features \
  --prep-dir /path/to/dataset/Prep \
  --subject sub-001 \
  --session ses-01 \
  --task taskB \
  --metrics fc,alff,falff,reho \
  --atlas Schaefer400_7Networks,Gordon \
  --config configs/defaults.yaml
```

The metrics stage runs after preprocessing. It auto-resolves the final MNI-space BOLD file for AFNI or FSL outputs, then dispatches selected metric families:

- **`fc`**: AFNI `3dNetCorr` → ROI time series, FC correlation/Fisher-z matrices, long-format edge table.
- **`alff` / `falff`**: AFNI `3dRSFC` → ALFF/fALFF voxel-wise maps (raw + z-scored), atlas ROI summaries.
- **`reho`**: AFNI `3dReHo` → ReHo voxel-wise map (raw + z-scored), atlas ROI summaries.
- **`graph`**: Python-igraph -> global graph metrics from Fisher-z FC matrices. Requires existing FC outputs. Density-based binary graph construction, small-world normalisation (Erdos-Renyi random null graphs), Leiden/Louvain community detection.

Default is `--metrics fc`. ALFF/fALFF use a fixed `0.01–0.1 Hz` passband; ReHo defaults to `nneigh=27`; Graph uses 10–25% density thresholds with 100 random networks. The dataset group mask (`grp_mask_thr70.nii.gz`) is applied for FC and regional metrics.

Raw AFNI outputs are kept under `metrics/task-<task>/raw/`. Standardized outputs live under `timeseries/`, `connectivity/`, `maps/`, `regional/`, and `qc/`.

Gradient metrics are dataset-level only. Run them with `fmriproc features-dataset --metrics gradient` after FC has been generated.

### `fmriproc group-mask` — Build dataset-level MNI coverage masks

```bash
fmriproc group-mask \
  --prep-dir /path/to/dataset/Prep \
  --config configs/defaults.yaml
```

This command scans completed subject/session anatomical outputs, derives binary MNI coverage masks, and writes dataset-level masks under `Prep/group_mask/`. Each subject-session is treated as a separate input: a subject with three usable sessions contributes three masks, while a subject with two usable sessions contributes two. By default, a session enters the group mask only if at least one task-level `motion_summary.tsv` passes the loose, PhiPipe-aligned motion rule `mean FD < 0.5` and `FD > 0.5` outlier ratio `< 0.2`; exclusions are recorded in `group_mask_inputs.tsv`. Set `metrics.group_mask.selection.motion.fd_outlier_threshold: 0.2` for a stricter outlier definition.

The default metrics constraint is `grp_mask_thr70.nii.gz`; `grp_mask_thr100.nii.gz` is stricter and mainly used as a visual QC guardrail. When FSL `slicer` is available, the mask itself is rendered directly to `group_mask/qc/grp_mask_thr100_slicer.png` without a template underlay. With a one-subject dataset, both masks are equivalent to that subject's coverage mask, but the same group-mask workflow is still followed.

Run `group-mask` before `features` when `metrics.group_mask.apply: true` (the default). Metrics resamples the group mask to the BOLD grid when needed, applies it to the ROI atlas, and records the original mask, resampled mask, and constrained ROI path in the JSON sidecar.

### `fmriproc features-dataset` — Aggregate dataset-level metrics tables

```bash
fmriproc features-dataset \
  --prep-dir /path/to/dataset/Prep \
  --tasks taskA,taskB,taskC \
  --metrics fc,alff,falff,reho,graph,gradient \
  --atlas Schaefer400_7Networks,Gordon \
  --config configs/defaults.yaml
```

This command discovers completed subject/session/task runs under `Prep/`, applies inclusion rules (final BOLD existence, motion QC, group-mask availability), optionally computes missing single-run metrics with `--run-missing`, and writes analysis-ready group tables under `Prep/derivatives/metrics/`. Gradient uses the same included runs but requires completed FC Fisher-z matrices, FC QC TSVs, and FC metadata JSON files.

**Two modes:**
- **Aggregate-only (default):** discovers runs, writes an inclusion table, and concatenates already-existing run-level standardized outputs. Runs with missing metrics are marked `excluded` with reason `missing_metrics_outputs`.
- **Run-missing (`--run-missing`):** for included runs with missing outputs, automatically executes the required single-run metric families (FC via `3dNetCorr`, regional via `3dRSFC`/`3dReHo`) before aggregation. The `--run-missing` flag passes through to inclusion evaluation so missing outputs do not prevent a run from being included.

Gradient is generated during the dataset stage. It builds a group-average FC reference per atlas and configured condition group, fits BrainSpace diffusion-map gradients, aligns each run to its condition-group reference with Procrustes, and writes ROI, run-summary, network-summary, and reference tables. Configure flexible task grouping through `metrics.gradient.reference.condition_groups`, for example `condition1: [taskA, taskB, taskC]`. Unmapped tasks fail by default so condition definitions stay explicit.

**Inclusion rules** use task-specific motion QC (each task's own `motion_summary.tsv`), the same loose defaults as `group-mask` (mean FD < 0.5, FD outlier ratio < 0.2). These can be configured under `metrics.dataset.inclusion.motion`.

**Optional filters:**
- `--tasks taskA,taskB` — only process specified tasks.
- `--sessions ses-01,ses-02` — only process specified sessions.
- `--subjects-file subjects.txt` — only process subjects listed in file (one per line).

**Generated outputs under `Prep/derivatives/metrics/`:**

*FC:*
- `all_fc_long_atlas-<atlas>.tsv` — edge-level FC long table.
- `all_connectivity_summary.tsv` — per-run/per-atlas (n_edges, mean_z, sd_z).
- `all_roi_timeseries_qc.tsv` — per-run/per-atlas ROI/time-series QC.
- `participants_metrics_qc.tsv` — concatenated per-run FC metrics QC.

*Regional:*
- `all_regional_metrics_long.tsv` — per-ROI ALFF/fALFF/ReHo values.
- `all_regional_metrics_qc.tsv` — concatenated per-run regional QC.

*Gradient:*
- `all_gradient_roi.tsv` — aligned ROI gradient values by run/atlas/component.
- `all_gradient_summary.tsv` — per-run eigenvalue, explained variance ratio, gradient spread, and similarity/distance to reference.
- `all_gradient_network.tsv` — network centroids, within-network dispersion, and between-network separation when network labels can be inferred.
- `all_gradient_references.tsv` — condition-group reference eigenvalues, explained variance ratios, and reference run counts.

*Shared:*
- `participants_inclusion.tsv` — one row per discovered run with status/reason.
- `metrics_dataset_manifest.json` — command, config, atlas list, run counts, generated files.

The analysis unit is subject-session-task. Multiple sessions from the same subject remain separate rows.

### FC ComBat harmonization

`features-dataset` can optionally harmonize Fisher-z FC matrices with ComBat before FC-derived metrics are computed. This is disabled by default. Enable it only when a covariate CSV is available with required columns `ID`, `Age`, `Sex`, `MeanFD`, and `Site`.

```yaml
metrics:
  input:
    fc_variant: combat
  combat:
    enabled: true
    covariates_csv: /path/to/dataset/combat_covariates.csv
    match_keys: [ID, session, task]
    group_by: [session, modal, atlas]
```

Run:

```bash
fmriproc features-dataset --prep-dir /path/to/dataset/Prep --metrics combat,graph,gradient
```

ComBat reads completed raw FC outputs and writes `*_fc-fisherz-combat.tsv` plus `*_fc-combat.json` for each harmonized run. Dataset-level summaries are written to `Prep/derivatives/metrics/fc_combat_manifest.json` and `Prep/derivatives/metrics/fc_combat_groups.tsv`.

If `group_by` omits `task`, runs such as `taskA`, `taskB`, and `taskC` can be harmonized together when their CSV rows share `modal=condition1`. Add `task` to `group_by` when different task conditions should be harmonized separately.

### `fmriproc plot-matrix` — Render atlas matrix PNGs

```bash
fmriproc plot-matrix \
  --matrix run_fc-fisherz.tsv \
  --atlas Gordon \
  --output run_fc-fisherz.png \
  --threshold 0.2 \
  --mode both \
  --order network
```

Supports `Gordon` and `Schaefer<N>_7Networks` / `Schaefer<N>_17Networks` atlas labels. Thresholding can keep positive, negative, absolute, or both tails.

### `fmriproc dataset-qc` — Validate BIDS dataset

```bash
fmriproc dataset-qc \
  --bids-dir /path/to/dataset/NIFTI \
  --prep-dir /path/to/dataset/Prep \
  --config configs/defaults.yaml
```

### `fmriproc submit-hpc` — Batch HPC submission

Submit preprocessing jobs to an HPC scheduler (LSF or SLURM).

```bash
fmriproc submit-hpc \
  --bids-dir /path/to/dataset/NIFTI \
  --prep-dir /path/to/dataset/Prep \
  --subjects-file subjects.txt \
  --sessions ses-01,ses-02 \
  --tasks taskA,taskB,taskC \
  --config configs/defaults.yaml \
  --scheduler slurm          # or 'lsf' (default from config)
```

## Output Layout

```text
Prep/sub-<subject>/<session>/
  anat/
    README.md                        # Anatomical preprocessing summary
    t1_s01_deoblique.nii.gz          # (AFNI) or reorient2std (FSL)
    t1_s02_skullstrip.nii.gz
    t1_s03_unifize.nii.gz            # (AFNI only)
    t1_s04_space_mni152_affine.nii.gz
    t1_s05_space_mni152_nonlinear.nii.gz
    Segsy/                           # AFNI segmentation output
    seg/                             # FSL FAST output
  func/task-<task>/
    README.md                        # Functional preprocessing summary
    bold_s01_drop_dummy.nii.gz
    bold_s02_deoblique.nii.gz        # (AFNI only)
    bold_s03_motion_correction.nii.gz
    bold_s04_slice_timing.nii.gz     # Optional
    bold_s05_epi_ref_unifize.nii.gz  # (AFNI only)
    bold_s08_nuisance_regression.nii.gz
    bold_s09_grand_mean_scaling_10000.nii.gz
    bold_s10_space_mni152.nii.gz     # (AFNI) or bold_s11 (FSL)
    bold_s11_smooth_sm6.nii.gz       # (AFNI) or bold_s12 (FSL)
  masks/
    t1_brain_mask.nii.gz
    t1_wm_mask.nii.gz
    t1_csf_mask.nii.gz
    t1_*_to_epi.nii.gz
  reg/
    ants_*                           # ANTs registration outputs
    fsl_*                            # FSL registration outputs
  qc/task-<task>/
    motion_parameters.1D             # AFNI motion (deg + mm)
    motion_parameters.par            # FSL motion (rad + mm)
    motion_summary.tsv               # FD stats
    motion_fd_power.tsv              # Per-volume FD
    nuisance_regressors.tsv          # Human-readable nuisance matrix
    nuisance_regressors.1D           # AFNI no-header matrix
    nuisance_regressors.txt          # FSL no-header matrix
    nuisance_metadata.json           # Nuisance regression model metadata
    acompcor_<mask_label>_metadata.json # aCompCor components metadata (e.g. wm, csf)
    selected_inputs.tsv
    anat_source.tsv
    input_qc.tsv
    snapshots/
      precheck_t1_slices.png         # T1 axial montage
      precheck_epi_slices.png        # EPI axial montage
      epi_to_t1_overlay.png          # EPI→T1 registration overlay
      epi_to_mni_overlay.png         # EPI→MNI registration overlay
      motion_parameters.png          # 6-parameter motion plot
  logs/task-<task>/
    status.jsonl                     # Step execution log
    commands.log                     # Dry-run or executed commands
  metrics/task-<task>/
    raw/
      afni_3dNetCorr/              # Raw .netcc/.netts/.log from FC
      afni_3dRSFC/                 # Raw ALFF/fALFF HEAD/BRIK + log
      afni_3dReHo/                 # Raw ReHo HEAD/BRIK + log
    timeseries/                    # ROI time series TSV (FC)
    connectivity/                  # FC correlation, Fisher-z, long-edge TSVs
    maps/                          # Voxel-wise maps: alff/falff/reho + z-scored
    regional/                      # Per-atlas/per-metric ROI summary TSVs
    graph/                         # Threshold-wise + summary graph metrics TSVs
    gradient/                      # Run-level gradient ROI/summary/network TSVs
    raw/python_igraph_graph_metrics/ # Adjacency matrices + atlas-specific logs
    qc/                            # FC, regional, graph, and gradient QC + JSON metadata
  group_mask/
    grp_mask_thr70.nii.gz          # Default analysis constraint (70% overlap)
    grp_mask_thr100.nii.gz         # Strict visual QC mask (100% overlap)
    group_mask_inputs.tsv
    raw/afni_3dmask_tool/
      grp_mask_3dmask_tool.log
    qc/
      grp_mask_thr100_slicer.png
derivatives/metrics/               # Dataset-level aggregation
  participants_inclusion.tsv       # One row per discovered run
  participants_metrics_qc.tsv      # Concatenated FC metrics QC
  all_fc_long_atlas-<atlas>.tsv    # Concatenated FC long table
  all_regional_metrics_long.tsv    # Concatenated regional ROI summaries
  all_regional_metrics_qc.tsv      # Concatenated regional QC
  all_connectivity_summary.tsv     # Per-run/per-atlas FC summary
  all_roi_timeseries_qc.tsv        # Per-run/per-atlas ROI/time-series QC
  all_gradient_roi.tsv             # Dataset-level ROI gradient values
  all_gradient_summary.tsv         # Dataset-level run gradient summaries
  all_gradient_network.tsv         # Dataset-level network gradient summaries
  all_gradient_references.tsv      # Dataset-level condition reference metadata
  metrics_dataset_manifest.json    # Command, config, run counts
```

### Nuisance Regression

The pipeline generates one final nuisance design matrix before `bold_s08_nuisance_regression`:

- `qc/task-<task>/nuisance_regressors.tsv` — human-readable final matrix with column names.
- `qc/task-<task>/nuisance_regressors.1D` — AFNI matrix consumed by `3dTproject`.
- `qc/task-<task>/nuisance_regressors.txt` — FSL text matrix converted by `Text2Vest`.
- `qc/task-<task>/nuisance_regressors.mat` — FSL design matrix consumed by `fsl_glm`.
- `qc/task-<task>/nuisance_metadata.json` — selected nuisance model, input paths, and final column names.
- `qc/task-<task>/acompcor_<mask>_metadata.json` — per-mask aCompCor explained variance.

aCompCor recipe (the supplied default):

```yaml
bold:
  nuisance:
    model: acompcor        # Explicit model selects aCompCor plus 6 motion parameters.
    wm: false              # No extra WM mean; WM principal components are still used.
    csf: false             # No extra CSF mean; CSF principal components are still used.
    acompcor:
      enabled: true        # Legacy selector; explicit model takes precedence.
      n_components: 5      # Per mask, giving 10 tissue PCs in total.
      masks: [wm, csf]
```

`basic_6` uses six rigid-body motion parameters. `friston_24` uses six motion parameters, their derivatives, squares, and squared derivatives. `acompcor` uses six motion parameters plus principal components from WM/CSF masks.

The `wm` and `csf` switches control **additional tissue-mean regressors**, not whether aCompCor uses these tissues. With the default aCompCor recipe, WM/CSF-derived components are already included, so both mean switches remain `false`. Adding the means is an explicit alternative, not an automatic requirement or an exactly equivalent operation.

For a conventional motion-plus-tissue-means recipe, change the settings together:

```yaml
bold:
  nuisance:
    model: friston_24      # Use basic_6 for six motion columns instead.
    wm: true              # Include the WM mean time series.
    csf: true             # Include the CSF mean time series.
    acompcor:
      enabled: false      # Keep consistent with the selected non-aCompCor model.
```

| Recipe | Motion columns | Tissue regressors | Total nuisance columns |
|---|---:|---|---:|
| Default aCompCor, means off | 6 | 5 WM PCs + 5 CSF PCs | 16 |
| `basic_6`, both means on | 6 | WM mean + CSF mean | 8 |
| `friston_24`, both means on | 24 | WM mean + CSF mean | 26 |

These counts exclude any intercept, trends, or filtering terms handled by the backend; they are column counts, not guaranteed matrix ranks. Recipes are starting points to validate for the analysis, not a universal ranking of denoising quality.

**The switches do not change automatically when `model` changes.** An explicit `model` takes precedence over `motion_model` and `acompcor.enabled`. In particular, `model: acompcor` always uses six motion parameters in the current implementation. The legacy `motion` flag does not disable these columns, and `global_signal` is currently a reserved configuration field: the builder does not implement global-signal regression. `model: none` disables motion and aCompCor columns but leaves any explicitly enabled WM/CSF means active; it should not be interpreted as a complete no-regression mode.

Check the actual design matrix in `nuisance_regressors.tsv` and its column names in `nuisance_metadata.json` to confirm the regressors used.


## Design

- **Step registry**: Each backend declares its step list as a tuple of `StepDefinition` objects with dependencies. The runner resolves effective dependencies at runtime, skipping completed or disabled steps.
- **Argument resolution**: Separate `step_args.py` and `fsl_step_args.py` map each step ID to a positional argument list for the corresponding Bash function.
- **Bash execution**: Each backend has its own `*_steps.sh` script that defines one shell function per step. Python calls the script with the function name and its arguments.
- **Output validation**: Every step validates its primary output exists and is non-empty. Marker files (`.done`) are used for steps that produce directory trees or sidecar-only outputs.
- **Dry-run safety**: `--dry-run` logs commands without executing and skips software checks and QC generation.
- **Backend-aware runner**: `_step_by_id()`, `_effective_dependencies()`, and `_enabled_by_config()` are all backend-aware, selecting from the correct step registry based on `runtime.backend`.

This folder contains the deployable package. Development tests are not included in this distribution.

## Recommended Workflow

```bash
# 1. Preprocess each subject/session/task
fmriproc preprocess \
  --bids-dir /path/to/dataset/NIFTI \
  --prep-dir /path/to/dataset/Prep \
  --subject sub-001 --session ses-01 --task taskB \
  --config configs/defaults.yaml

# 2. Build dataset-level group mask (run once per dataset)
fmriproc group-mask \
  --prep-dir /path/to/dataset/Prep \
  --config configs/defaults.yaml

# 3. Extract per-run metrics (FC + regional)
fmriproc features \
  --prep-dir /path/to/dataset/Prep \
  --subject sub-001 --session ses-01 --task taskB \
  --metrics fc,alff,falff,reho \
  --config configs/defaults.yaml

# 4. Aggregate across the dataset
fmriproc features-dataset \
  --prep-dir /path/to/dataset/Prep \
  --tasks taskA,taskB,taskC \
  --metrics fc,alff,falff,reho \
  --config configs/defaults.yaml

# Or: aggregate + auto-compute missing runs
fmriproc features-dataset \
  --prep-dir /path/to/dataset/Prep \
  --tasks taskA,taskB,taskC \
  --metrics fc,alff,falff,reho \
  --run-missing
```

## Supported Atlases

| Name | Parcels | Networks | Space | Label source |
|------|---------|----------|-------|-------------|
| `Schaefer400_7Networks` | 400 | 7 | MNI152 1/2/3mm | FreeSurfer LUT |
| `Schaefer400_17Networks` | 400 | 17 | MNI152 1/2/3mm | FreeSurfer LUT |
| `Gordon` | 333 | — | MNI152NLin6Asym | BIDS TSV |

Atlas resources live under `fmriproc/resources/atlas/`. Resolution is auto-selected to match the BOLD grid; add `_1mm`, `_2mm`, or `_3mm` candidates for explicit control.

## Metrics Reference

| Family | Tool | Output | Key Config |
|--------|------|--------|------------|
| FC | `3dNetCorr` | timeseries, correlation, Fisher-z, long-edge TSVs | `metrics.atlases` |
| ALFF | `3dRSFC` | `_alff.nii.gz` + `_alff_z.nii.gz` | `metrics.regional.frequency` |
| fALFF | `3dRSFC` | `_falff.nii.gz` + `_falff_z.nii.gz` | (same 3dRSFC run as ALFF) |
| ReHo | `3dReHo` | `_reho.nii.gz` + `_reho_z.nii.gz` | `metrics.regional.reho.neighborhood` |
| Graph | python-igraph | threshold + summary TSVs, adjacency matrices | `metrics.graph.densities`, `metrics.graph.random_networks` |
| Gradient | BrainSpace | ROI values, run summary, network dispersion/separation, reference TSVs | `metrics.gradient.reference.condition_groups`, `metrics.gradient.n_components` |

ALFF and fALFF are always computed together by `3dRSFC`. Selecting either `--metrics alff` or `--metrics falff` produces both maps and ROI summaries. The default passband is 0.01–0.1 Hz. TR must be set via `bold.tr` in config or available from a BIDS JSON sidecar in the func task directory.

Graph metrics require completed FC outputs. They derive undirected unweighted binary graphs from Fisher-z matrices at configured density thresholds (default 10–25%), compute global metrics (clustering coefficient, characteristic path length, global efficiency, modularity, small-worldness), and normalise via Erdős-Rényi random null graphs. Community detection prefers Leiden (`leidenalg`) with Louvain and fast-greedy fallbacks.

Gradient metrics require completed FC outputs and BrainSpace (`brainspace>=0.2.0`). They are computed through `features-dataset`, not the single-run `features` command, because reference gradients and Procrustes alignment are group concepts. For the example grouping `condition1: [taskA, taskB, taskC]`, gradients share a reference within `condition1`. Set this mapping explicitly in your configuration for your dataset. Each atlas/condition group requires at least two included runs by default.
