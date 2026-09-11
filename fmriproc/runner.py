from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .manifest import append_log, append_status
from .models import SelectedInputs, StepDefinition, StepResult
from .paths import PrepPaths
from .step_args import _slice_timing_enabled, resolve_step_args


class Runner:
    def __init__(
        self,
        paths: PrepPaths,
        inputs: SelectedInputs,
        config: dict[str, Any],
        dry_run: bool = False,
        template_path: str = "",
    ) -> None:
        self.paths = paths
        self.inputs = inputs
        self.config = config
        self.dry_run = dry_run
        self.template_path = template_path
        self._dry_run_previewed_steps: set[str] = set()
        self.paths.ensure_dirs()
        (self.paths.anat_log_dir / "done").mkdir(parents=True, exist_ok=True)
        (self.paths.log_task_dir / "done").mkdir(parents=True, exist_ok=True)

    def run_step(self, step: StepDefinition, force: bool = False) -> StepResult:
        output = self._primary_output(step)
        done_dir = self._done_dir(step)
        done_path = done_dir / f"{step.step_id}.json"
        work_dir = self.paths.work_dir / step.step_id
        command: list[str] = []

        try:
            command = self._build_command(step)

            if self._is_samseg_mask_step(step):
                if self.dry_run:
                    result = StepResult(step.step_id, "dry_run", command, "samseg_masks_from_segmentation_step")
                    self._record(step, result)
                    self._dry_run_previewed_steps.add(step.step_id)
                    return result
                for mask_name in ("t1_brain_mask.nii.gz", "t1_wm_mask.nii.gz", "t1_csf_mask.nii.gz"):
                    mask_path = self.paths.masks_dir / mask_name
                    if not self._primary_output_is_valid(mask_path):
                        raise RuntimeError(f"SAMSEG mask step missing required mask: {mask_path}")
                output.write_text("done\n", encoding="utf-8")
                result = StepResult(step.step_id, "completed", command, "samseg_masks_already_written")
                self._record(step, result)
                return result

            # Check if this step should be skipped by config (e.g., slice_timing disabled).
            if step.optional and not self._enabled_by_config(step):
                result = StepResult(step.step_id, "skipped", command, "skipped_by_config")
                self._record(step, result)
                if self.dry_run:
                    self._dry_run_previewed_steps.add(step.step_id)
                return result

            # Only real done records (not dry-run) satisfy skip logic.
            if self._primary_output_is_valid(output) and done_path.exists() and not force and not self._done_record_is_dry_run(done_path):
                result = StepResult(step.step_id, "skipped", command, "output_exists_with_done_record")
                self._record(step, result)
                return result

            missing_dependencies = self._missing_dependencies(step)
            if missing_dependencies:
                raise RuntimeError(f"Missing dependency for {step.step_id}: {', '.join(missing_dependencies)}")

            if force and not self.dry_run:
                if done_path.exists():
                    done_path.unlink()
                if output.exists():
                    output.unlink()
                self._clean_sidecar_outputs(step)
                if work_dir.exists():
                    shutil.rmtree(work_dir)

            work_dir.mkdir(parents=True, exist_ok=True)
            start = datetime.now(timezone.utc)

            if self.dry_run:
                pass
            else:
                log_file = work_dir / f"{step.step_id}.log"
                with log_file.open("w", encoding="utf-8") as f:
                    try:
                        subprocess.run(command, check=True, stdout=f, stderr=subprocess.STDOUT)
                    except subprocess.CalledProcessError as exc:
                        try:
                            with log_file.open("r", encoding="utf-8") as lf:
                                lines = lf.readlines()
                                tail = "".join(lines[-20:])
                        except Exception:
                            tail = "Could not read log."
                        raise RuntimeError(f"Command failed with exit code {exc.returncode}. Log tail:\n{tail}") from exc

                if self._is_samseg_segmentation_step(step):
                    from .segmentation import (
                        derive_masks_from_samseg,
                        requested_segmentation_backend,
                        write_segmentation_provenance,
                    )

                    samseg_dir = self.paths.anat_dir / "samseg"
                    derive_masks_from_samseg(samseg_dir / "seg.mgz", self.paths.masks_dir)
                    write_segmentation_provenance(
                        self.paths.qc_output("segmentation_backend.json"),
                        requested=requested_segmentation_backend(self.config),
                        used="samseg",
                        source_t1=self.inputs.t1.path,
                        samseg_dir=samseg_dir,
                    )
                    output.write_text("done\n", encoding="utf-8")

                self._validate_primary_output(output)

            end = datetime.now(timezone.utc)
            if not self.dry_run:
                done_path.write_text(
                    json.dumps(
                        {
                            "step_id": step.step_id,
                            "output": str(output),
                            "start": start.isoformat(),
                            "end": end.isoformat(),
                            "dry_run": False,
                        },
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
            if self.config.get("cleanup", {}).get("intermediate", True) and work_dir.exists():
                shutil.rmtree(work_dir)
            result_status = "dry_run" if self.dry_run else "completed"
            result = StepResult(step.step_id, result_status, command, "")
            self._record(step, result)
            if self.dry_run:
                self._dry_run_previewed_steps.add(step.step_id)
            return result
        except Exception as exc:
            result = StepResult(step.step_id, "failed", command, str(exc))
            self._record(step, result)
            if isinstance(exc, RuntimeError):
                raise
            raise RuntimeError(str(exc)) from exc

    def _clean_sidecar_outputs(self, step: StepDefinition) -> None:
        """Remove known sidecar/transform files for this step (force re-run cleanup)."""
        reg_dir = self.paths.reg_dir
        # ANTs registration outputs — separate prefixes to avoid cross-step deletion.
        an_t_prefixes: dict[str, str] = {
            "t1_s04_space_mni152_affine": "ants_t1_to_mni_affine_",
            "t1_s05_space_mni152_nonlinear": "ants_t1_to_mni_nonlinear_",
            "bold_s06_epi_to_anat": "ants_epi_to_t1_affine_",
        }
        if step.step_id in an_t_prefixes:
            prefix = str(reg_dir / an_t_prefixes[step.step_id])
            for suffix in (
                "Warped.nii.gz", "0GenericAffine.mat", "1Warp.nii.gz",
                "1InverseWarp.nii.gz", "InverseWarped.nii.gz",
            ):
                p = Path(prefix + suffix)
                if p.exists():
                    p.unlink()
        # AFNI registration outputs.
        if step.step_id in ("t1_s04_space_mni152_affine", "t1_s05_space_mni152_nonlinear"):
            for fn in ("anat_s01_to_mni152_affine.1D", "anat_s02_to_mni152_warp.nii.gz"):
                p = reg_dir / fn
                if p.exists():
                    p.unlink()
        if step.step_id == "bold_s06_epi_to_anat":
            for pattern in ("*_align_mat.aff12.1D", "*_align_reg_mat.aff12.1D",
                            "*_align+orig.HEAD", "*_align+orig.BRIK*"):
                for p in reg_dir.glob(pattern):
                    p.unlink()

    def _done_dir(self, step: StepDefinition) -> Path:
        if step.module == "anat":
            return self.paths.anat_log_dir / "done"
        return self.paths.log_task_dir / "done"

    def _missing_dependencies(self, step: StepDefinition) -> list[str]:
        missing: list[str] = []
        for dependency_id in self._effective_dependencies(step):
            if self.dry_run and dependency_id in self._dry_run_previewed_steps:
                continue

            dependency = self._step_by_id(dependency_id)
            dependency_output = self._primary_output(dependency)
            if self._primary_output_is_valid(dependency_output):
                continue
            missing.append(f"{dependency_id} ({dependency_output})")
        return missing

    def _effective_dependencies(self, step: StepDefinition) -> tuple[str, ...]:
        backend = self.config.get("runtime", {}).get("backend", "afni")
        dependencies = list(step.dependencies)
        # Only AFNI/ANTs backend needs post-motion routing via dynamic step IDs.
        if backend != "fsl":
            if step.step_id == "bold_s05_epi_ref_unifize":
                post_motion = "bold_s04_slice_timing" if _slice_timing_enabled(self.config) else "bold_s03_motion_correction"
                dependencies = [dep for dep in dependencies if dep != "bold_s03_motion_correction"]
                dependencies.append(post_motion)
            elif step.step_id == "bold_s08_nuisance_regression":
                post_motion = "bold_s04_slice_timing" if _slice_timing_enabled(self.config) else "bold_s03_motion_correction"
                if post_motion not in dependencies:
                    dependencies.append(post_motion)
        return tuple(dict.fromkeys(dependencies))

    def _step_by_id(self, step_id: str) -> StepDefinition:
        backend = self.config.get("runtime", {}).get("backend", "afni")
        if backend == "fsl":
            from .fsl_steps import FSL_ANAT_STEPS, FSL_BOLD_STEPS

            for step in (*FSL_ANAT_STEPS, *FSL_BOLD_STEPS):
                if step.step_id == step_id:
                    return step
        else:
            from .steps import ANAT_STEPS, BOLD_STEPS

            for step in (*ANAT_STEPS, *BOLD_STEPS):
                if step.step_id == step_id:
                    return step
        raise ValueError(f"Unknown dependency step: {step_id}")

    @staticmethod
    def _done_record_is_dry_run(done_path: Path) -> bool:
        try:
            record = json.loads(done_path.read_text(encoding="utf-8"))
            return bool(record.get("dry_run", False))
        except Exception:
            return False

    def _enabled_by_config(self, step: StepDefinition) -> bool:
        if step.step_id in ("bold_s04_slice_timing", "bold_s05_slice_timing"):
            return _slice_timing_enabled(self.config)
        return True

    def _primary_output(self, step: StepDefinition) -> Path:
        if step.module == "anat":
            return self.paths.anat_output(step.primary_output)
        return self.paths.bold_output(step.primary_output)

    @staticmethod
    def _primary_output_is_valid(output: Path) -> bool:
        if not output.exists():
            return False
        if output.name.endswith((".nii", ".nii.gz")):
            return output.stat().st_size > 0
        return True

    def _validate_primary_output(self, output: Path) -> None:
        if not output.exists():
            raise RuntimeError(f"Step completed but primary output was not produced: {output}")
        if output.name.endswith((".nii", ".nii.gz")) and output.stat().st_size == 0:
            raise RuntimeError(f"Step completed but primary output is empty: {output}")

    def _actual_segmentation_backend(self) -> str:
        import shutil as _shutil

        from .segmentation import resolve_segmentation_backend

        backend = self.config.get("runtime", {}).get("backend", "afni")
        return resolve_segmentation_backend(self.config, backend, _shutil.which("run_samseg"))

    def _is_samseg_segmentation_step(self, step: StepDefinition) -> bool:
        return (
            step.step_id in {"t1_s06_segmentation", "t1_s03_fast_segmentation"}
            and self._actual_segmentation_backend() == "samseg"
        )

    def _is_samseg_mask_step(self, step: StepDefinition) -> bool:
        return (
            step.step_id in {"t1_s07_masks", "t1_s04_masks"}
            and self._actual_segmentation_backend() == "samseg"
        )

    def _build_command(self, step: StepDefinition) -> list[str]:
        backend = self.config.get("runtime", {}).get("backend", "afni")
        if self._is_samseg_segmentation_step(step):
            from .segmentation import build_run_samseg_command

            threads = int(self.config.get("anat", {}).get("samseg", {}).get("threads", 1))
            return build_run_samseg_command(
                self.inputs.t1.path,
                self.paths.anat_dir / "samseg",
                threads=threads,
            )
        if self._is_samseg_mask_step(step):
            return ["samseg_masks_to_pipeline", str(self.paths.masks_dir)]

        if backend == "fsl":
            from .fsl_step_args import resolve_fsl_step_args

            args = resolve_fsl_step_args(step, self.paths, self.inputs, self.config, self.template_path)
            script = Path(__file__).parent / "bash" / "fsl_steps.sh"
            return ["bash", str(script), step.bash_function] + args

        reg_backend = self._resolve_registration_backend()
        args = resolve_step_args(step, self.paths, self.inputs, self.config, self.template_path, reg_backend)
        script_name, func_name = self._select_script_and_function(step, reg_backend)
        script = Path(__file__).parent / "bash" / script_name
        return ["bash", str(script), func_name] + args

    def _resolve_registration_backend(self) -> str:
        import shutil as _shutil

        reg_backend = self.config.get("runtime", {}).get("registration_backend", "ants_if_available")
        if reg_backend == "ants":
            return "ants"
        if reg_backend == "afni":
            return "afni"
        # ants_if_available: need BOTH tools; fall back to AFNI if either is missing.
        has_ants = (
            _shutil.which("antsRegistrationSyN.sh") is not None
            and _shutil.which("antsApplyTransforms") is not None
        )
        return "ants" if has_ants else "afni"

    def _select_script_and_function(self, step: StepDefinition, reg_backend: str) -> tuple[str, str]:
        if reg_backend == "afni":
            return ("afni_steps.sh", step.bash_function)

        # ANTs backend maps registration + mask steps to ANTs equivalents.
        ant_func_map: dict[str, str] = {
            "t1_s04_space_mni152_affine": "ants_t1_to_mni_affine",
            "t1_s05_space_mni152_nonlinear": "ants_t1_to_mni_nonlinear",
            "bold_s06_epi_to_anat": "ants_epi_to_t1_affine",
            "bold_s07_masks_to_epi": "ants_masks_to_epi",
            "bold_s10_space_mni152": "ants_apply_epi_to_mni",
        }
        if step.step_id in ant_func_map:
            return ("ants_steps.sh", ant_func_map[step.step_id])
        return ("afni_steps.sh", step.bash_function)

    def _record(self, step: StepDefinition, result: StepResult) -> None:
        append_log(self.paths.log_task_dir / "commands.log", " ".join(result.command))
        append_status(
            self.paths.log_task_dir / "status.jsonl",
            {
                "subject": self.paths.subject,
                "session": self.paths.session,
                "task": self.paths.task,
                "step_id": step.step_id,
                "status": result.status,
                "reason": result.reason,
                "command": result.command,
                "dry_run": self.dry_run,
            },
        )
