from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AtlasSpec:
    name: str
    atlas_path_candidates: tuple[Path, ...]
    label_path: Path | None
    space: str
    has_network_labels: bool
    default: bool = False

    def resolve_path(self, resolution: str = "auto") -> Path:
        existing = [path for path in self.atlas_path_candidates if path.exists()]
        if not existing:
            candidates = ", ".join(str(path) for path in self.atlas_path_candidates)
            raise FileNotFoundError(f"Atlas resources for {self.name} not found. Candidates: {candidates}")
        if resolution == "auto":
            return existing[0]
        wanted = f"_{resolution}.nii"
        for path in existing:
            if wanted in path.name:
                return path
        raise FileNotFoundError(f"Atlas {self.name} has no {resolution} candidate. Available: {existing}")


def resource_root() -> Path:
    return Path(__file__).parent / "resources" / "atlas"


def built_in_atlases() -> dict[str, AtlasSpec]:
    root = resource_root()
    yeo = root / "Yeo_MNI"
    gordon = root / "Gordon"
    return {
        "Schaefer400_7Networks": AtlasSpec(
            name="Schaefer400_7Networks",
            atlas_path_candidates=(
                yeo / "Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz",
                yeo / "Schaefer2018_3mm" / "Schaefer2018_400Parcels_7Networks_order_FSLMNI152_3mm.nii.gz",
                yeo / "Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz",
            ),
            label_path=yeo / "freeview_lut" / "Schaefer2018_400Parcels_7Networks_order.txt",
            space="MNI152",
            has_network_labels=True,
            default=True,
        ),
        "Schaefer400_17Networks": AtlasSpec(
            name="Schaefer400_17Networks",
            atlas_path_candidates=(
                yeo / "Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz",
                yeo / "Schaefer2018_3mm" / "Schaefer2018_400Parcels_17Networks_order_FSLMNI152_3mm.nii.gz",
                yeo / "Schaefer2018_400Parcels_17Networks_order_FSLMNI152_1mm.nii.gz",
            ),
            label_path=yeo / "freeview_lut" / "Schaefer2018_400Parcels_17Networks_order.txt",
            space="MNI152",
            has_network_labels=True,
            default=False,
        ),
        "Gordon": AtlasSpec(
            name="Gordon",
            atlas_path_candidates=(
                gordon / "atlas-Gordon_space-MNI152NLin6Asym_dseg.nii.gz",
            ),
            label_path=gordon / "atlas-Gordon_dseg.tsv",
            space="MNI152NLin6Asym",
            has_network_labels=False,
            default=True,
        ),
    }


def parse_atlas_names(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def atlas_names_from_config(cfg: dict[str, Any]) -> list[str]:
    configured = cfg.get("metrics", {}).get("atlases", {}).get("default")
    if configured:
        return list(configured)
    return [name for name, spec in built_in_atlases().items() if spec.default]


def resolve_atlases(names: list[str]) -> list[AtlasSpec]:
    registry = built_in_atlases()
    missing = [name for name in names if name not in registry]
    if missing:
        valid = ", ".join(sorted(registry))
        raise ValueError(f"Unknown atlas name(s): {', '.join(missing)}. Valid atlas names: {valid}")
    return [registry[name] for name in names]
