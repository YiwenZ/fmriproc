from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any

from .config import load_config


def _reject_unsafe_path(value: str, label: str) -> None:
    """Reject paths that contain shell-sensitive characters."""
    unsafe = {" ", "\t", "\n", "\r", "'", '"', "`", "$", "(", ")", ";", "|", "&", "<", ">"}
    for char in unsafe:
        if char in value:
            raise ValueError(f"{label} contains unsafe character {char!r}: {value}")


def build_bsub_command(
    subject: str,
    session: str,
    task: str,
    bids_dir: str,
    prep_dir: str,
    config: str,
    hpc: dict[str, Any],
) -> list[str]:
    _reject_unsafe_path(subject, "subject")
    _reject_unsafe_path(session, "session")
    _reject_unsafe_path(task, "task")
    _reject_unsafe_path(bids_dir, "bids_dir")
    _reject_unsafe_path(prep_dir, "prep_dir")
    _reject_unsafe_path(config, "config")

    job_name = f"fmriproc_{subject}_{session}_task_{task}"
    run_command = (
        f"fmriproc preprocess --bids-dir {bids_dir} --prep-dir {prep_dir} "
        f"--subject {subject} --session {session} --task {task} --config {config}"
    )
    cmd = [
        "bsub",
        "-J", job_name,
        "-q", str(hpc["queue"]),
        "-G", str(hpc["group"]),
        "-n", str(hpc["n_cores"]),
        "-R", f"select[mem>{hpc['memory_mb']}] rusage[mem={hpc['memory_mb']}] span[hosts=1]",
    ]
    cmd.extend(shlex.split(run_command))
    return cmd


def build_sbatch_command(
    subject: str,
    session: str,
    task: str,
    bids_dir: str,
    prep_dir: str,
    config: str,
    hpc: dict[str, Any],
) -> list[str]:
    """Build sbatch command for SLURM scheduler."""
    _reject_unsafe_path(subject, "subject")
    _reject_unsafe_path(session, "session")
    _reject_unsafe_path(task, "task")
    _reject_unsafe_path(bids_dir, "bids_dir")
    _reject_unsafe_path(prep_dir, "prep_dir")
    _reject_unsafe_path(config, "config")

    job_name = f"fmriproc_{subject}_{session}_task_{task}"
    run_command = (
        f"fmriproc preprocess --bids-dir {bids_dir} --prep-dir {prep_dir} "
        f"--subject {subject} --session {session} --task {task} --config {config}"
    )
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--cpus-per-task={hpc['n_cores']}",
        f"--mem={hpc['memory_mb']}",
    ]
    if hpc.get("queue"):
        cmd.append(f"--partition={hpc['queue']}")
    if hpc.get("group"):
        cmd.append(f"--account={hpc['group']}")
    cmd.append(f"--wrap={run_command}")
    return cmd


def submit_hpc_jobs(args) -> int:
    """Submit batch jobs to HPC scheduler (LSF or SLURM)."""
    cfg = load_config(args.config)
    hpc = cfg["hpc"]

    # Read scheduler from args if explicitly specified, otherwise fall back to config.
    scheduler = getattr(args, "scheduler", None) or hpc.get("scheduler", "lsf")

    if scheduler not in ("lsf", "slurm"):
        raise ValueError(f"Unsupported scheduler: {scheduler}. Must be 'lsf' or 'slurm'.")

    subjects = [
        line.strip()
        for line in Path(args.subjects_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sessions = [value.strip() for value in args.sessions.split(",") if value.strip()]
    tasks = [value.strip() for value in args.tasks.split(",") if value.strip()]

    for subject in subjects:
        for session in sessions:
            for task in tasks:
                if scheduler == "slurm":
                    command = build_sbatch_command(
                        subject, session, task, args.bids_dir, args.prep_dir, args.config, hpc
                    )
                else:
                    command = build_bsub_command(
                        subject, session, task, args.bids_dir, args.prep_dir, args.config, hpc
                    )

                if args.dry_run:
                    print(shlex.join(command))
                else:
                    subprocess.run(command, check=True)
    return 0
