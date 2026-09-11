from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PrepPaths:
    prep_dir: Path
    subject: str
    session: str
    task: str

    @property
    def session_dir(self) -> Path:
        return self.prep_dir / self.subject / self.session

    @property
    def anat_dir(self) -> Path:
        return self.session_dir / "anat"

    @property
    def func_task_dir(self) -> Path:
        return self.session_dir / "func" / f"task-{self.task}"

    @property
    def reg_dir(self) -> Path:
        return self.session_dir / "reg"

    @property
    def masks_dir(self) -> Path:
        return self.session_dir / "masks"

    @property
    def qc_task_dir(self) -> Path:
        return self.session_dir / "qc" / f"task-{self.task}"

    @property
    def log_task_dir(self) -> Path:
        return self.session_dir / "logs" / f"task-{self.task}"

    @property
    def anat_log_dir(self) -> Path:
        """Session-level directory for anat step done records, shared across tasks."""
        return self.session_dir / "logs" / "anat"

    @property
    def work_dir(self) -> Path:
        return self.log_task_dir / "work"

    def anat_output(self, filename: str) -> Path:
        self._validate_generated_name(filename)
        return self.anat_dir / filename

    def bold_output(self, filename: str) -> Path:
        self._validate_generated_name(filename)
        return self.func_task_dir / filename

    def reg_output(self, filename: str) -> Path:
        self._validate_generated_name(filename)
        return self.reg_dir / filename

    def mask_output(self, filename: str) -> Path:
        self._validate_generated_name(filename)
        return self.masks_dir / filename

    def qc_output(self, filename: str) -> Path:
        self._validate_generated_name(filename)
        return self.qc_task_dir / filename

    def ensure_dirs(self) -> None:
        for directory in [
            self.anat_dir,
            self.func_task_dir,
            self.reg_dir,
            self.masks_dir,
            self.qc_task_dir,
            self.log_task_dir,
            self.anat_log_dir,
            self.work_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_generated_name(filename: str) -> None:
        if "-" in filename:
            raise ValueError(f"Generated pipeline filenames must not contain hyphens: {filename}")


@dataclass(frozen=True)
class MetricPaths:
    prep_paths: PrepPaths

    @property
    def metrics_task_dir(self) -> Path:
        return self.prep_paths.session_dir / "metrics" / f"task-{self.prep_paths.task}"

    def raw_tool_dir(self, tool_name: str) -> Path:
        self._validate_tool_name(tool_name)
        return self.metrics_task_dir / "raw" / tool_name

    @property
    def timeseries_dir(self) -> Path:
        return self.metrics_task_dir / "timeseries"

    @property
    def connectivity_dir(self) -> Path:
        return self.metrics_task_dir / "connectivity"

    @property
    def metrics_qc_dir(self) -> Path:
        return self.metrics_task_dir / "qc"

    def stem(self, atlas: str) -> str:
        self._validate_atlas_name(atlas)
        return (
            f"{self.prep_paths.subject}_{self.prep_paths.session}_"
            f"task-{self.prep_paths.task}_atlas-{atlas}"
        )

    def timeseries_output(self, atlas: str) -> Path:
        return self.timeseries_dir / f"{self.stem(atlas)}_timeseries.tsv"

    def fc_correlation_output(self, atlas: str) -> Path:
        return self.connectivity_dir / f"{self.stem(atlas)}_fc-correlation.tsv"

    def fc_fisherz_output(self, atlas: str) -> Path:
        return self.connectivity_dir / f"{self.stem(atlas)}_fc-fisherz.tsv"

    def fc_long_output(self, atlas: str) -> Path:
        return self.connectivity_dir / f"{self.stem(atlas)}_fc-long.tsv"

    def qc_tsv_output(self, atlas: str) -> Path:
        return self.metrics_qc_dir / f"{self.stem(atlas)}_metrics-qc.tsv"

    def metadata_json_output(self, atlas: str) -> Path:
        return self.metrics_qc_dir / f"{self.stem(atlas)}_metrics.json"

    def fc_fisherz_combat_output(self, atlas: str) -> Path:
        return self.connectivity_dir / f"{self.stem(atlas)}_fc-fisherz-combat.tsv"

    def fc_combat_metadata_json_output(self, atlas: str) -> Path:
        return self.metrics_qc_dir / f"{self.stem(atlas)}_fc-combat.json"

    # ── Regional metrics (maps, regional summaries) ──────────────────

    @property
    def maps_dir(self) -> Path:
        return self.metrics_task_dir / "maps"

    @property
    def regional_dir(self) -> Path:
        return self.metrics_task_dir / "regional"

    def _metric_stem(self, metric: str) -> str:
        """Stem for metric-specific filenames (no atlas)."""
        self._validate_metric_name(metric)
        return (
            f"{self.prep_paths.subject}_{self.prep_paths.session}_"
            f"task-{self.prep_paths.task}"
        )

    def map_output(self, metric: str) -> Path:
        return self.maps_dir / f"{self._metric_stem(metric)}_{metric}.nii.gz"

    def z_map_output(self, metric: str) -> Path:
        return self.maps_dir / f"{self._metric_stem(metric)}_{metric}_z.nii.gz"

    def roi_summary_output(self, atlas: str, metric: str) -> Path:
        self._validate_atlas_name(atlas)
        self._validate_metric_name(metric)
        return self.regional_dir / f"{self.stem(atlas)}_{metric}_roi.tsv"

    def regional_qc_tsv_output(self) -> Path:
        return self.metrics_qc_dir / f"{self._metric_stem('regional')}_regional_metrics_qc.tsv"

    def regional_metadata_json_output(self) -> Path:
        return self.metrics_qc_dir / f"{self._metric_stem('regional')}_regional_metrics.json"

    # ── Graph metrics ─────────────────────────────────────────────

    @property
    def graph_dir(self) -> Path:
        return self.metrics_task_dir / "graph"

    def graph_thresholds_output(self, atlas: str) -> Path:
        self._validate_atlas_name(atlas)
        return self.graph_dir / f"{self.stem(atlas)}_graph_thresholds.tsv"

    def graph_summary_output(self, atlas: str) -> Path:
        self._validate_atlas_name(atlas)
        return self.graph_dir / f"{self.stem(atlas)}_graph_summary.tsv"

    def graph_metadata_json_output(self, atlas: str) -> Path:
        self._validate_atlas_name(atlas)
        return self.metrics_qc_dir / f"{self.stem(atlas)}_graph.json"

    def graph_raw_dir(self) -> Path:
        return self.raw_tool_dir("python_igraph_graph_metrics")

    def graph_raw_log(self, atlas: str) -> Path:
        return self.graph_raw_dir() / f"{self.stem(atlas)}_graph_python_igraph.log"

    def graph_adjacency_dir(self, atlas: str, density: float) -> Path:
        pct = int(round(density * 100))
        return self.graph_raw_dir() / f"{self.stem(atlas)}_density-{pct}"

    # ── Gradient metrics ──────────────────────────────────────────

    @property
    def gradient_dir(self) -> Path:
        return self.metrics_task_dir / "gradient"

    def gradient_roi_output(self, atlas: str) -> Path:
        self._validate_atlas_name(atlas)
        return self.gradient_dir / f"{self.stem(atlas)}_gradient_roi.tsv"

    def gradient_run_summary_output(self, atlas: str) -> Path:
        self._validate_atlas_name(atlas)
        return self.gradient_dir / f"{self.stem(atlas)}_gradient_summary.tsv"

    def gradient_network_output(self, atlas: str) -> Path:
        self._validate_atlas_name(atlas)
        return self.gradient_dir / f"{self.stem(atlas)}_gradient_network.tsv"

    def gradient_metadata_json_output(self, atlas: str) -> Path:
        self._validate_atlas_name(atlas)
        return self.metrics_qc_dir / f"{self.stem(atlas)}_gradient.json"

    # ── dirs ────────────────────────────────────────────────────────

    def ensure_dirs(self) -> None:
        for directory in [
            self.raw_tool_dir("afni_3dNetCorr"),
            self.timeseries_dir,
            self.connectivity_dir,
            self.maps_dir,
            self.regional_dir,
            self.graph_dir,
            self.gradient_dir,
            self.metrics_qc_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    # ── validators ──────────────────────────────────────────────────

    @staticmethod
    def _validate_tool_name(tool_name: str) -> None:
        if "/" in tool_name or tool_name in {"", ".", ".."}:
            raise ValueError(f"Invalid metrics tool name: {tool_name}")

    @staticmethod
    def _validate_atlas_name(atlas: str) -> None:
        if not atlas or any(char in atlas for char in "/\\"):
            raise ValueError(f"Invalid atlas name for generated metrics filename: {atlas}")

    @staticmethod
    def _validate_metric_name(metric: str) -> None:
        if not metric or any(char in metric for char in "/\\"):
            raise ValueError(f"Invalid metric name: {metric}")


@dataclass(frozen=True)
class GroupMaskPaths:
    prep_dir: Path

    @property
    def group_mask_dir(self) -> Path:
        return self.prep_dir / "group_mask"

    @property
    def raw_3dmask_tool_dir(self) -> Path:
        return self.group_mask_dir / "raw" / "afni_3dmask_tool"

    @property
    def qc_dir(self) -> Path:
        return self.group_mask_dir / "qc"

    @property
    def thr70_mask(self) -> Path:
        return self.group_mask_dir / "grp_mask_thr70.nii.gz"

    @property
    def thr100_mask(self) -> Path:
        return self.group_mask_dir / "grp_mask_thr100.nii.gz"

    @property
    def inputs_tsv(self) -> Path:
        return self.group_mask_dir / "group_mask_inputs.tsv"

    @property
    def log_path(self) -> Path:
        return self.raw_3dmask_tool_dir / "grp_mask_3dmask_tool.log"

    @property
    def qc_snapshot(self) -> Path:
        return self.qc_dir / "grp_mask_thr100_slicer.png"

    def ensure_dirs(self) -> None:
        for directory in [self.group_mask_dir, self.raw_3dmask_tool_dir, self.qc_dir]:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DatasetMetricPaths:
    """Paths for dataset-level metrics aggregation under ``Prep/derivatives/metrics/``."""

    prep_dir: Path

    @property
    def derivatives_metrics_dir(self) -> Path:
        return self.prep_dir / "derivatives" / "metrics"

    @property
    def inclusion_tsv(self) -> Path:
        return self.derivatives_metrics_dir / "participants_inclusion.tsv"

    @property
    def participants_metrics_qc_tsv(self) -> Path:
        return self.derivatives_metrics_dir / "participants_metrics_qc.tsv"

    def fc_long_tsv(self, atlas: str) -> Path:
        self._validate_atlas_name(atlas)
        return self.derivatives_metrics_dir / f"all_fc_long_atlas-{atlas}.tsv"

    @property
    def connectivity_summary_tsv(self) -> Path:
        return self.derivatives_metrics_dir / "all_connectivity_summary.tsv"

    @property
    def roi_timeseries_qc_tsv(self) -> Path:
        return self.derivatives_metrics_dir / "all_roi_timeseries_qc.tsv"

    @property
    def manifest_json(self) -> Path:
        return self.derivatives_metrics_dir / "metrics_dataset_manifest.json"

    @property
    def combat_manifest_json(self) -> Path:
        return self.derivatives_metrics_dir / "fc_combat_manifest.json"

    @property
    def combat_groups_tsv(self) -> Path:
        return self.derivatives_metrics_dir / "fc_combat_groups.tsv"

    @property
    def gradient_roi_tsv(self) -> Path:
        return self.derivatives_metrics_dir / "all_gradient_roi.tsv"

    @property
    def gradient_summary_tsv(self) -> Path:
        return self.derivatives_metrics_dir / "all_gradient_summary.tsv"

    @property
    def gradient_network_tsv(self) -> Path:
        return self.derivatives_metrics_dir / "all_gradient_network.tsv"

    @property
    def gradient_reference_tsv(self) -> Path:
        return self.derivatives_metrics_dir / "all_gradient_references.tsv"

    def ensure_dirs(self) -> None:
        self.derivatives_metrics_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_atlas_name(atlas: str) -> None:
        if not atlas or any(char in atlas for char in "/\\"):
            raise ValueError(f"Invalid atlas name for dataset metrics filename: {atlas}")
