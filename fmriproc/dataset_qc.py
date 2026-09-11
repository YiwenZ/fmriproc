from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from .bids_scan import BidsIndex, choose_duplicate_bold


def write_dataset_qc(index: BidsIndex, prep_dir: Path, expected_volumes: dict[str, int]) -> None:
    qc_dir = Path(prep_dir) / "dataset_qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    _write_inventory(index, qc_dir / "subject_session_inventory.tsv")
    _write_volume_summary(index, expected_volumes, qc_dir)
    _write_duplicate_report(index, qc_dir / "duplicate_run_report.tsv")
    _write_missing_report(index, qc_dir / "missing_inputs_report.tsv")


def _write_inventory(index: BidsIndex, path: Path) -> None:
    sessions = sorted({(item.subject, item.session or "") for item in index.t1w + index.bold + index.sbref + index.fmap_ap + index.fmap_pa})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject", "session", "has_t1", "n_bold", "n_sbref", "has_fmap_ap", "has_fmap_pa"], delimiter="\t")
        writer.writeheader()
        for subject, session in sessions:
            writer.writerow(
                {
                    "subject": subject,
                    "session": session,
                    "has_t1": str(any(item.subject == subject and item.session == session for item in index.t1w)),
                    "n_bold": sum(item.subject == subject and item.session == session for item in index.bold),
                    "n_sbref": sum(item.subject == subject and item.session == session for item in index.sbref),
                    "has_fmap_ap": str(any(item.subject == subject and item.session == session for item in index.fmap_ap)),
                    "has_fmap_pa": str(any(item.subject == subject and item.session == session for item in index.fmap_pa)),
                }
            )


def _write_volume_summary(index: BidsIndex, expected_volumes: dict[str, int], qc_dir: Path) -> None:
    volumes_by_task: dict[str, list[int]] = defaultdict(list)
    for bold in index.bold:
        if bold.task and bold.volume_count is not None:
            volumes_by_task[bold.task].append(bold.volume_count)

    suggested: dict[str, int] = {}
    with (qc_dir / "task_volume_summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["task", "n_runs", "configured_expected_volumes", "suggested_expected_volumes",
                        "volume_counts", "volume_mismatch"],
            delimiter="\t",
        )
        writer.writeheader()
        for task in sorted(volumes_by_task):
            counts = volumes_by_task[task]
            majority = Counter(counts).most_common(1)[0][0]
            suggested[task] = majority
            configured = expected_volumes.get(task)
            if configured is not None and configured != majority:
                mismatch = f"configured={configured} majority={majority}"
            elif configured is not None and any(c != configured for c in counts):
                mismatch = f"configured={configured} but some runs differ"
            elif configured is None:
                mismatch = "not_configured"
            else:
                mismatch = ""
            writer.writerow(
                {
                    "task": task,
                    "n_runs": len(counts),
                    "configured_expected_volumes": configured if configured is not None else "",
                    "suggested_expected_volumes": majority,
                    "volume_counts": ",".join(str(value) for value in sorted(counts)),
                    "volume_mismatch": mismatch,
                }
            )

    with (qc_dir / "suggested_expected_volumes.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"input_selection": {"expected_volumes": suggested}}, handle, sort_keys=True)


def _write_duplicate_report(index: BidsIndex, path: Path) -> None:
    grouped: dict[tuple[str, str, str], list] = defaultdict(list)
    for bold in index.bold:
        if bold.session and bold.task:
            grouped[(bold.subject, bold.session, bold.task)].append(bold)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["subject", "session", "task", "n_candidates",
                        "selected_path", "selected_volume_count", "selection_reason",
                        "candidate_paths", "candidate_volume_counts"],
            delimiter="\t",
        )
        writer.writeheader()
        for (subject, session, task), candidates in sorted(grouped.items()):
            selected, _, reason = choose_duplicate_bold(candidates)
            writer.writerow(
                {
                    "subject": subject,
                    "session": session,
                    "task": task,
                    "n_candidates": len(candidates),
                    "selected_path": selected.path,
                    "selected_volume_count": selected.volume_count,
                    "selection_reason": reason,
                    "candidate_paths": ";".join(str(item.path) for item in sorted(candidates, key=lambda item: item.path.name)),
                    "candidate_volume_counts": ";".join(str(item.volume_count) for item in sorted(candidates, key=lambda item: item.path.name)),
                }
            )


def _write_missing_report(index: BidsIndex, path: Path) -> None:
    sessions = sorted({(item.subject, item.session or "") for item in index.bold + index.t1w})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject", "session", "missing"], delimiter="\t")
        writer.writeheader()
        for subject, session in sessions:
            missing: list[str] = []
            if not any(item.subject == subject and item.session == session for item in index.t1w):
                missing.append("T1w")
            if not any(item.subject == subject and item.session == session for item in index.bold):
                missing.append("bold")
            if not any(item.subject == subject and item.session == session for item in index.sbref):
                missing.append("sbref")
            if not any(item.subject == subject and item.session == session for item in index.fmap_ap):
                missing.append("fmap_AP")
            if not any(item.subject == subject and item.session == session for item in index.fmap_pa):
                missing.append("fmap_PA")
            writer.writerow({"subject": subject, "session": session, "missing": ",".join(missing)})
