from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class BidsFile:
    path: Path
    subject: str
    session: str | None
    suffix: str
    datatype: str
    task: str | None = None
    run: str | None = None
    direction: str | None = None
    volume_count: int | None = None


@dataclass(frozen=True)
class SelectedInputs:
    subject: str
    session: str
    task: str
    bold: BidsFile
    t1: BidsFile
    sbref: BidsFile | None = None
    fmap_ap: BidsFile | None = None
    fmap_pa: BidsFile | None = None
    unselected_bold: tuple[BidsFile, ...] = ()
    t1_fallback_reason: str = "current_session"
    bold_selection_reason: str = "single_candidate"


@dataclass(frozen=True)
class StepDefinition:
    step_id: str
    module: Literal["anat", "bold"]
    bash_function: str
    primary_output: str
    dependencies: tuple[str, ...] = ()
    optional: bool = False


@dataclass
class StepResult:
    step_id: str
    status: Literal["completed", "skipped", "failed", "dry_run"]
    command: list[str] = field(default_factory=list)
    reason: str = ""
