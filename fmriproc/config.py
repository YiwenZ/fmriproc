from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "dataset": {
        "bids_dir": "../Data/Dataset/NIFTI",
        "prep_dir": "../Data/Dataset/Prep",
    },
    "runtime": {
        "conda_env": "image",
        "backend": "afni",
        "registration_backend": "ants_if_available",
        "mode": "tested",
    },
    "anat": {
        "strategy": "session",
        "fallback": "later_then_earlier",
        "segmentation_backend": "samseg_if_available",
        "samseg": {
            "threads": 4,
        },
    },
    "input_selection": {
        "duplicate_run_policy": "longest_then_latest",
        "expected_volumes": {},
        "expected_volumes_missing": "warn",
    },
    "bold": {
        "dummy_scans": 10,
        "slice_timing": {
            "enabled": False,
            "source": "afni_pattern",
            "afni_pattern": "alt+z",
        },
        "tr": "auto",
        "highpass": 0.01,
        "lowpass": None,
        "nuisance": {
            "wm": False,
            "csf": False,
            "motion": True,
            "model": "acompcor",
            "motion_model": "basic_6",
            "global_signal": False,
            "acompcor": {
                "enabled": True,
                "n_components": 5,
                "masks": ["wm", "csf"],
            },
        },
        "grand_mean": 10000,
        "smooth_fwhm": 6,
    },
    "qc": {
        "report_after_run": True,
        "snapshots": {
            "enabled": True,
            "n_slices": 9,
            "orientation": "axial",
            "overlay_alpha": 0.35,
        },
        "readme": {
            "enabled": True,
        },
        "motion_plot": {
            "enabled": True,
        },
    },
    "metrics": {
        "backend": "afni",
        "default": ["fc"],
        "input": {
            "space": "mni152",
            "smoothed": False,
        },
        "atlases": {
            "default": ["Schaefer400_7Networks", "Gordon"],
            "resolution": "auto",
        },
        "netcorr": {
            "fisher_z": True,
            "save_individual_roi_timeseries": True,
            "push_thru_many_zeros": True,
        },
        "group_mask": {
            "enabled": True,
            "apply": True,
            "default_constraint": "grp_mask_thr70.nii.gz",
            "qc_snapshot_mask": "grp_mask_thr100.nii.gz",
            "min_valid_subjects": 1,
            "selection": {
                "require_motion_qc": True,
                "motion": {
                    "mean_fd_max": 0.5,
                    "fd_outlier_ratio_max": 0.2,
                    "fd_outlier_threshold": 0.5,
                },
            },
            "thresholds": [0.7, 1.0],
        },
        "qc": {
            "min_roi_coverage": 0.5,
            "report_after_run": True,
        },
        "graph": {
            "enabled": True,
            "backend": "python_igraph",
            "densities": [0.10, 0.15, 0.20, 0.25],
            "random_networks": 100,
            "seed": 112,
            "seed_strategy": "run_hash",
            "require_leiden": False,
            "keep_adjacency": True,
        },
        "gradient": {
            "enabled": True,
            "backend": "brainspace",
            "n_components": 3,
            "approach": "dm",
            "kernel": "normalized_angle",
            "alignment": "procrustes",
            "sparsity": 0.9,
            "random_state": 42,
            "reference": {
                "scope": "condition_group",
                "min_runs_per_group": 2,
                "unmapped_task_policy": "fail",
                "condition_groups": {
                    "movie": ["movieA", "movieB", "movieC"]
                }
            },
            "network_summary": {
                "enabled": True,
                "min_rois_per_network": 2,
            }
        },
        "dataset": {
            "prep_dir": None,
            "output_dir": "derivatives/metrics",
            "run_missing": False,
            "include_existing_metrics_only": True,
            "inclusion": {
                "require_group_mask": True,
                "require_motion_qc": True,
                "motion": {
                    "mean_fd_max": 0.5,
                    "fd_outlier_ratio_max": 0.2,
                    "fd_outlier_threshold": 0.5,
                }
            }
        },
        "regional": {
            "enabled": True,
            "backend": "afni",
            "metrics": ["alff", "falff", "reho"],
            "input": {
                "space": "mni152",
                "smoothed": False,
            },
            "frequency": {
                "low_hz": 0.01,
                "high_hz": 0.1,
            },
            "reho": {
                "neighborhood": 27,
            },
            "group_mask": {
                "apply": True,
                "constraint": "grp_mask_thr70.nii.gz",
            },
            "standardize": {
                "zscore_within_mask": True,
            },
            "roi_summary": {
                "atlases": ["Schaefer400_7Networks", "Gordon"],
                "min_roi_coverage": 0.5,
            }
        }
    },
    "cleanup": {
        "intermediate": True,
        "keep_failed_workdir": True,
        "keep_step_outputs": True,
    },
    "templates": {
        "mni152_default_resolution": "3mm",
        "mni152_3mm_brain": "fmriproc/resources/templates/MNI152/MNI152_T1_3mm_brain.nii.gz",
        "mni152_2mm_brain": "fmriproc/resources/templates/MNI152/MNI152_T1_2mm_brain.nii.gz",
        "mni152_1mm_brain": "fmriproc/resources/templates/MNI152/MNI152_T1_1mm_brain.nii.gz",
    },
    "hpc": {
        "scheduler": "lsf",
        "queue": "general",
        "group": "compute-perlmansusan",
        "docker_image": None,
        "n_cores": 4,
        "memory_mb": 32000,
        "sleep_seconds_between_jobs": 0.5,
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    # Try loading configs/defaults.yaml from the project root first
    project_root = Path(__file__).resolve().parent.parent
    defaults_file = project_root / "configs" / "defaults.yaml"
    
    base_config = deepcopy(DEFAULT_CONFIG)
    if defaults_file.exists():
        with defaults_file.open("r", encoding="utf-8") as handle:
            disk_defaults = yaml.safe_load(handle) or {}
            if isinstance(disk_defaults, dict):
                base_config = deep_merge(base_config, disk_defaults)

    if path is None:
        return base_config

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    return deep_merge(base_config, loaded)
