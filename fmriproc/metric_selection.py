"""Parse and validate metric family selections (``fc``, ``alff``, ``falff``, ``reho``).

Shared between ``fmriproc features`` and ``fmriproc features-dataset``.
"""

from __future__ import annotations

from typing import Any

VALID_METRICS = frozenset({"fc", "alff", "falff", "reho", "combat", "graph", "gradient"})
REGIONAL_METRICS = frozenset({"alff", "falff", "reho"})


def parse_metrics(value: str | None, cfg: dict[str, Any]) -> list[str]:
    """Return a validated, deduplicated list of metric family names.

    *value* is the comma-separated ``--metrics`` argument (may be ``None`` or
    empty).  Falls back to ``metrics.default`` in *cfg*, then to ``["fc"]``.
    """
    raw = (value or "").strip()
    if raw:
        names = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        names = cfg.get("metrics", {}).get("default", ["fc"])

    seen: set[str] = set()
    validated: list[str] = []
    for name in names:
        if name not in VALID_METRICS:
            raise ValueError(
                f"Unknown metric family: {name!r}. Valid values: {', '.join(sorted(VALID_METRICS))}"
            )
        if name not in seen:
            seen.add(name)
            validated.append(name)

    if "combat" in validated:
        if not ((cfg.get("metrics") or {}).get("combat") or {}).get("enabled", False):
            raise ValueError("combat metric requested but metrics.combat.enabled is false")

    return validated
