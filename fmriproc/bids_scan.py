from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

from .models import BidsFile


@dataclass
class BidsIndex:
    t1w: list[BidsFile]
    bold: list[BidsFile]
    sbref: list[BidsFile]
    fmap_ap: list[BidsFile]
    fmap_pa: list[BidsFile]


ENTITY_RE = re.compile(r"(?P<key>sub|ses|task|run|dir)-(?P<value>[^_]+)")


def parse_bids_file(path: Path) -> BidsFile:
    name = path.name
    if name.endswith(".nii.gz"):
        stem = name[:-7]
    else:
        stem = path.stem

    parts = stem.split("_")
    suffix = parts[-1]
    entities = {match.group("key"): match.group("value") for part in parts for match in [ENTITY_RE.fullmatch(part)] if match}

    datatype = path.parent.name if path.parent.name in {"anat", "func", "fmap"} else ""
    subject = f"sub-{entities['sub']}" if "sub" in entities else ""
    session = f"ses-{entities['ses']}" if "ses" in entities else None

    return BidsFile(
        path=path,
        subject=subject,
        session=session,
        suffix=suffix,
        datatype=datatype,
        task=entities.get("task"),
        run=entities.get("run"),
        direction=entities.get("dir"),
    )


def default_volume_counter(path: Path) -> int | None:
    try:
        import nibabel as nib
    except ImportError:
        return None

    try:
        img = nib.load(str(path))
    except Exception:
        return None

    shape = img.shape
    return int(shape[3]) if len(shape) >= 4 else 1


def scan_bids(bids_dir: Path, volume_counter: Callable[[Path], int | None] = default_volume_counter) -> BidsIndex:
    t1w: list[BidsFile] = []
    bold: list[BidsFile] = []
    sbref: list[BidsFile] = []
    fmap_ap: list[BidsFile] = []
    fmap_pa: list[BidsFile] = []

    for path in sorted(Path(bids_dir).rglob("*.nii.gz")):
        item = parse_bids_file(path)
        if item.datatype == "anat" and item.suffix == "T1w":
            t1w.append(item)
        elif item.datatype == "func" and item.suffix == "bold":
            bold.append(replace(item, volume_count=volume_counter(path)))
        elif item.datatype == "func" and item.suffix == "sbref":
            sbref.append(item)
        elif item.datatype == "fmap" and item.suffix == "epi" and item.direction == "AP":
            fmap_ap.append(item)
        elif item.datatype == "fmap" and item.suffix == "epi" and item.direction == "PA":
            fmap_pa.append(item)

    return BidsIndex(t1w=t1w, bold=bold, sbref=sbref, fmap_ap=fmap_ap, fmap_pa=fmap_pa)


def choose_duplicate_bold(candidates: Iterable[BidsFile]) -> tuple[BidsFile, tuple[BidsFile, ...], str]:
    items = list(candidates)
    if not items:
        raise ValueError("No BOLD candidates provided")
    if len(items) == 1:
        return items[0], (), "single_candidate"

    def run_number(item: BidsFile) -> int:
        if item.run is None:
            return -1
        try:
            return int(item.run)
        except ValueError:
            return -1

    max_vol = max(item.volume_count if item.volume_count is not None else -1 for item in items)
    max_items = [item for item in items if (item.volume_count if item.volume_count is not None else -1) == max_vol]
    selected = sorted(max_items, key=lambda item: (run_number(item), item.path.name))[-1]
    unselected = tuple(item for item in items if item != selected)
    reason = "latest_among_max_volume_runs" if len(max_items) > 1 else "max_volume"
    return selected, unselected, reason


def resolve_t1_for_session(subject: str, session: str, t1_files: Iterable[BidsFile]) -> tuple[BidsFile, str]:
    subject_t1 = [item for item in t1_files if item.subject == subject and item.session is not None]
    current = [item for item in subject_t1 if item.session == session]
    if current:
        return sorted(current, key=lambda item: item.path.name)[-1], "current_session"

    target_number = _session_number(session)
    later = sorted(
        [item for item in subject_t1 if _session_number(item.session) > target_number],
        key=lambda item: _session_number(item.session),
    )
    if later:
        return later[0], "fallback_later_session"

    earlier = sorted(
        [item for item in subject_t1 if _session_number(item.session) < target_number],
        key=lambda item: _session_number(item.session),
        reverse=True,
    )
    if earlier:
        return earlier[0], "fallback_earlier_session"

    raise ValueError(f"No T1w found for {subject} using fallback policy later_then_earlier")


def _session_number(session: str | None) -> int:
    if session is None:
        return -1
    match = re.search(r"(\d+)$", session)
    return int(match.group(1)) if match else -1


def read_bids_json(nii_path: Path) -> dict:
    """Read the BIDS JSON sidecar for a .nii.gz file.

    Returns an empty dict if the sidecar does not exist.
    """
    import json

    name = nii_path.name
    if name.endswith(".nii.gz"):
        json_name = name[:-7] + ".json"
    else:
        json_name = nii_path.stem + ".json"
    json_path = nii_path.parent / json_name
    if not json_path.exists():
        return {}
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
