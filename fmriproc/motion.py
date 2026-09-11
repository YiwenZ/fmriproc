from __future__ import annotations

import math
from statistics import mean, median


def compute_power_fd(motion_rows: list[list[float]], radius_mm: float = 50.0) -> list[float]:
    """Compute Power FD from AFNI 3dvolreg motion parameters.

    AFNI 3dvolreg -1Dfile columns (6 per row):
      [roll(deg), pitch(deg), yaw(deg), dS(mm), dL(mm), dP(mm)]

    Power FD = sum(|delta translation|) + radius * sum(|delta rotation| * pi/180)
    """
    if not motion_rows:
        return []

    fd = [0.0]
    for previous, current in zip(motion_rows, motion_rows[1:]):
        diffs = [abs(c - p) for p, c in zip(previous, current)]
        # Columns 3-5 are translations (mm).
        translation = sum(diffs[3:6])
        # Columns 0-2 are rotations (degrees); convert to radians then multiply by radius.
        rotation = sum(value * (math.pi / 180.0) * radius_mm for value in diffs[0:3])
        fd.append(translation + rotation)
    return fd


def summarize_motion(
    subject: str,
    session: str,
    task: str,
    n_volumes_raw: int,
    n_dummy_removed: int,
    motion_rows: list[list[float]],
) -> dict[str, object]:
    fd = compute_power_fd(motion_rows)
    translations = [abs(value) for row in motion_rows for value in row[3:6]]
    rotations = [abs(value) for row in motion_rows for value in row[0:3]]
    n_used = len(motion_rows)

    return {
        "subject": subject,
        "session": session,
        "task": task,
        "n_volumes_raw": n_volumes_raw,
        "n_dummy_removed": n_dummy_removed,
        "n_volumes_used": n_used,
        "mean_fd_power": mean(fd) if fd else 0.0,
        "median_fd_power": median(fd) if fd else 0.0,
        "max_fd_power": max(fd) if fd else 0.0,
        "n_fd_power_gt_0_2": sum(value > 0.2 for value in fd),
        "n_fd_power_gt_0_5": sum(value > 0.5 for value in fd),
        "percent_fd_power_gt_0_2": (sum(value > 0.2 for value in fd) / n_used * 100.0) if n_used else 0.0,
        "percent_fd_power_gt_0_5": (sum(value > 0.5 for value in fd) / n_used * 100.0) if n_used else 0.0,
        "mean_abs_translation": mean(translations) if translations else 0.0,
        "max_abs_translation": max(translations) if translations else 0.0,
        "mean_abs_rotation": mean(rotations) if rotations else 0.0,
        "max_abs_rotation": max(rotations) if rotations else 0.0,
    }


def read_motion_1d(path) -> list[list[float]]:
    rows: list[list[float]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            rows.append([float(value) for value in stripped.split()[:6]])
    return rows


def plot_motion_parameters(par_path, output_path, backend: str) -> None:
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    rows = read_motion_1d(par_path)
    if not rows:
        raise ValueError(f"No motion rows found: {par_path}")

    rotations = [[row[i] for row in rows] for i in range(3)]
    translations = [[row[i] for row in rows] for i in range(3, 6)]
    rot_unit = "rad" if backend == "fsl" else "deg"
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for values, label in zip(rotations, ["rot_x", "rot_y", "rot_z"]):
        axes[0].plot(values, label=label)
    axes[0].set_ylabel(f"rotation ({rot_unit})")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.25)

    for values, label in zip(translations, ["trans_x", "trans_y", "trans_z"]):
        axes[1].plot(values, label=label)
    axes[1].set_ylabel("translation (mm)")
    axes[1].set_xlabel("volume")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.25)

    fig.suptitle("Motion parameters")
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)


def summarize_fsl_motion(par_path) -> dict[str, object]:
    """Summarize FSL MCFLIRT motion from a .par file.

    FSL .par columns: rot_x(rad), rot_y(rad), rot_z(rad), trans_x(mm), trans_y(mm), trans_z(mm).
    Rotations are already in radians — no deg→rad conversion needed.
    """
    rows = read_motion_1d(par_path)
    translations = [abs(value) for row in rows for value in row[3:6]]
    rotations = [abs(value) for row in rows for value in row[0:3]]
    n_used = len(rows)

    # Compute FD: sum(|Δtrans|) + radius * sum(|Δrot|) with rotations already in rad.
    fd: list[float] = [0.0]
    radius_mm = 50.0
    for previous, current in zip(rows, rows[1:]):
        diffs = [abs(c - p) for p, c in zip(previous, current)]
        trans = sum(diffs[3:6])
        rot = sum(value * radius_mm for value in diffs[0:3])
        fd.append(trans + rot)

    return {
        "n_volumes": n_used,
        "mean_abs_rotation_rad": mean(rotations) if rotations else 0.0,
        "max_abs_rotation_rad": max(rotations) if rotations else 0.0,
        "mean_abs_translation_mm": mean(translations) if translations else 0.0,
        "max_abs_translation_mm": max(translations) if translations else 0.0,
        "mean_fd_mm": mean(fd) if fd else 0.0,
        "max_fd_mm": max(fd) if fd else 0.0,
        "n_fd_gt_0_2": sum(v > 0.2 for v in fd),
        "n_fd_gt_0_5": sum(v > 0.5 for v in fd),
    }
