from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import rc_context
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from .atlas_registry import resource_root


GORDON_NETWORK_COLORS = {
    "Default": "#ff1f2c",
    "ParietoOccip": "#fffdce",
    "FrontoParietal": "#ffe833",
    "Salience": "#252525",
    "CinguloOperc": "#8b1d72",
    "MedialParietal": "#a15be2",
    "DorsalAttn": "#009f24",
    "VentralAttn": "#00798c",
    "Visual": "#0e26a2",
    "SMhand": "#00b6d9",
    "SMmouth": "#ff8929",
    "Auditory": "#f80bba",
    "None": "#eeeeee",
}

YEO_7_NETWORK_COLORS = {
    "Vis": "#781286",
    "SomMot": "#4682b4",
    "DorsAttn": "#00760e",
    "SalVentAttn": "#c43afa",
    "Limbic": "#dcf8a4",
    "Cont": "#e69422",
    "Default": "#cd3e4e",
}

YEO_17_NETWORK_COLORS = {
    "VisCent": "#781685",
    "VisPeri": "#f90603",
    "SomMotA": "#4384b0",
    "SomMotB": "#29d0a1",
    "DorsAttnA": "#499e3b",
    "DorsAttnB": "#06770d",
    "SalVentAttnA": "#c139fb",
    "SalVentAttnB": "#fe99d6",
    "LimbicB": "#7b8839",
    "LimbicA": "#dafa99",
    "ContA": "#e69a2d",
    "ContB": "#8a334e",
    "ContC": "#758db0",
    "DefaultA": "#fdfe08",
    "DefaultB": "#d03b50",
    "DefaultC": "#000283",
    "TempPar": "#0f30fd",
}

THRESHOLD_BAND_GRAY = "#bdbdbd"


@dataclass(frozen=True)
class AtlasLabels:
    atlas: str
    labels: list[str]
    networks: list[str]
    colors: list[str]
    color_source: str


@dataclass(frozen=True)
class MatrixPlotResult:
    output_path: Path
    atlas: str
    n_rois: int
    mode: str
    threshold: float | None


def read_square_matrix(path: Path) -> tuple[np.ndarray, list[str]]:
    path = Path(path)
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    frame = pd.read_csv(path, sep=sep)
    if frame.empty:
        raise ValueError(f"Matrix file is empty: {path}")

    first_col = frame.columns[0]
    labels = frame[first_col].astype(str).tolist()
    values = frame.drop(columns=[first_col]).to_numpy(dtype=float)
    if values.shape[0] != values.shape[1]:
        raise ValueError(f"Matrix must be square after removing first column: {path}")
    if len(labels) != values.shape[0]:
        raise ValueError(f"Matrix row labels do not match row count: {path}")
    return values, labels


def apply_matrix_threshold(matrix: np.ndarray, threshold: float | None, mode: str = "none") -> np.ndarray:
    mode = mode.lower()
    valid_modes = {"none", "positive", "negative", "abs", "both"}
    if mode not in valid_modes:
        raise ValueError(f"Unknown threshold mode: {mode}. Valid modes: {sorted(valid_modes)}")
    result = np.array(matrix, dtype=float, copy=True)
    if threshold is None or mode == "none":
        return result

    if mode == "positive":
        keep = result >= threshold
    elif mode == "negative":
        keep = result <= -threshold
    elif mode == "abs":
        keep = np.abs(result) >= threshold
    else:
        keep = (result >= threshold) | (result <= -threshold)
    result[~keep] = np.nan
    return result


def load_atlas_labels(atlas: str) -> AtlasLabels:
    if atlas == "Gordon":
        return _load_gordon_labels(atlas)

    match = re.fullmatch(r"Schaefer(\d+)_(7|17)Networks", atlas)
    if match:
        n_parcels = int(match.group(1))
        n_networks = int(match.group(2))
        return _load_schaefer_labels(atlas, n_parcels, n_networks)

    raise ValueError(f"Unsupported atlas for matrix plotting: {atlas}")


def plot_atlas_matrix(
    matrix_path: Path,
    atlas: str,
    output_path: Path,
    threshold: float | None = 0,
    mode: str = "both",
    title: str | None = None,
    hide_diagonal: bool = False,
    transparent: bool = True,
    show_colorbar: bool = False,
    block_linewidth: float = 0.85,
    hemisphere_linewidth: float = 1.1,
    colorbar_fraction: float = 0.036,
    colorbar_pad: float = 0.02,
    font_family: str = "Times New Roman",
    dpi: int = 300,
    order: str = "original",
    networks: list[str] | None = None,
) -> MatrixPlotResult:
    matrix, matrix_labels = read_square_matrix(Path(matrix_path))
    atlas_labels = load_atlas_labels(atlas)
    labels, roi_networks, colors = _align_labels(matrix_labels, atlas_labels)
    if networks:
        filter_indices = _network_filter_indices(roi_networks, networks)
        labels = [labels[index] for index in filter_indices]
        roi_networks = [roi_networks[index] for index in filter_indices]
        colors = [colors[index] for index in filter_indices]
        matrix = matrix[np.ix_(filter_indices, filter_indices)]
    order_indices = _ordered_indices(labels, roi_networks, order=order)
    labels = [labels[index] for index in order_indices]
    roi_networks = [roi_networks[index] for index in order_indices]
    colors = [colors[index] for index in order_indices]
    matrix = matrix[np.ix_(order_indices, order_indices)]
    plotted = apply_matrix_threshold(matrix, threshold, mode)
    if hide_diagonal:
        np.fill_diagonal(plotted, np.nan)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _draw_matrix(
        plotted,
        labels,
        roi_networks,
        colors,
        output_path,
        title=title,
        threshold=threshold,
        transparent=transparent,
        show_colorbar=show_colorbar,
        block_linewidth=block_linewidth,
        hemisphere_linewidth=hemisphere_linewidth,
        colorbar_fraction=colorbar_fraction,
        colorbar_pad=colorbar_pad,
        font_family=font_family,
        dpi=dpi,
    )
    return MatrixPlotResult(
        output_path=output_path,
        atlas=atlas,
        n_rois=len(labels),
        mode=mode,
        threshold=threshold,
    )


def _network_filter_indices(networks: list[str], requested: list[str] | None) -> list[int]:
    if not requested:
        return list(range(len(networks)))

    requested_keys: set[str] = set()
    for name in requested:
        requested_keys.add(_network_key(name))

    indices = [index for index, network in enumerate(networks) if _network_key(network) in requested_keys]
    if not indices:
        available = ", ".join(sorted(set(networks)))
        wanted = ", ".join(requested)
        raise ValueError(f"No ROIs matched requested networks: {wanted}. Available networks: {available}")
    return indices


def _network_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _ordered_indices(labels: list[str], networks: list[str], order: str) -> list[int]:
    if order == "original":
        return list(range(len(labels)))
    if order != "network":
        raise ValueError("Matrix order must be 'original' or 'network'")

    network_order: dict[str, int] = {}
    for name in list(GORDON_NETWORK_COLORS) + list(YEO_7_NETWORK_COLORS) + list(YEO_17_NETWORK_COLORS):
        network_order.setdefault(name, len(network_order))

    return sorted(
        range(len(labels)),
        key=lambda index: (
            _hemisphere_rank(labels[index]),
            network_order.get(networks[index], 10_000),
            _roi_number(labels[index]),
            labels[index],
        ),
    )


def _hemisphere_rank(label: str) -> int:
    if label.startswith("L_") or "_LH_" in label:
        return 0
    if label.startswith("R_") or "_RH_" in label:
        return 1
    return 2


def _roi_number(label: str) -> int:
    match = re.search(r"_(\d+)$", label)
    return int(match.group(1)) if match else 0


def _load_gordon_labels(atlas: str) -> AtlasLabels:
    path = resource_root() / "Gordon" / "atlas-Gordon_dseg.tsv"
    labels: list[str] = []
    networks: list[str] = []
    colors: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            network = row["network_label"]
            labels.append(row["label"])
            networks.append(network)
            colors.append(GORDON_NETWORK_COLORS.get(network, "#eeeeee"))
    return AtlasLabels(atlas=atlas, labels=labels, networks=networks, colors=colors, color_source="fmriproc_default")


def _load_schaefer_labels(atlas: str, n_parcels: int, n_networks: int) -> AtlasLabels:
    lut_path = (
        resource_root()
        / "Yeo_MNI"
        / "freeview_lut"
        / f"Schaefer2018_{n_parcels}Parcels_{n_networks}Networks_order.txt"
    )
    if not lut_path.exists():
        raise FileNotFoundError(f"Schaefer LUT not found: {lut_path}")

    labels: list[str] = []
    networks: list[str] = []
    colors: list[str] = []
    with lut_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            label = parts[1]
            network = _schaefer_network_from_label(label, n_networks)
            labels.append(label)
            networks.append(network)
            if n_networks == 7:
                colors.append(YEO_7_NETWORK_COLORS.get(network, "#eeeeee"))
            elif n_networks == 17:
                colors.append(YEO_17_NETWORK_COLORS.get(network, "#eeeeee"))
            else:
                colors.append(_rgb_to_hex(parts[2], parts[3], parts[4]))

    color_source = "fmriproc_yeo7" if n_networks == 7 else "fmriproc_yeo17"
    return AtlasLabels(atlas=atlas, labels=labels, networks=networks, colors=colors, color_source=color_source)


def _schaefer_network_from_label(label: str, n_networks: int) -> str:
    parts = label.split("_")
    if n_networks == 7 and len(parts) >= 3:
        return parts[2]
    if n_networks == 17 and len(parts) >= 3:
        # Collapse 17-network sublabels into the first subnetwork token for block drawing.
        return parts[2]
    return "Unknown"


def _rgb_to_hex(red: str, green: str, blue: str) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(red), int(green), int(blue))


def _align_labels(matrix_labels: list[str], atlas_labels: AtlasLabels) -> tuple[list[str], list[str], list[str]]:
    lookup: dict[str, tuple[str, str]] = {
        label: (network, color)
        for label, network, color in zip(atlas_labels.labels, atlas_labels.networks, atlas_labels.colors)
    }
    if all(label in lookup for label in matrix_labels):
        networks = [lookup[label][0] for label in matrix_labels]
        colors = [lookup[label][1] for label in matrix_labels]
        return matrix_labels, networks, colors

    if len(matrix_labels) <= len(atlas_labels.labels):
        networks = atlas_labels.networks[: len(matrix_labels)]
        colors = atlas_labels.colors[: len(matrix_labels)]
        return matrix_labels, networks, colors

    raise ValueError(
        f"Matrix has {len(matrix_labels)} labels but atlas {atlas_labels.atlas} has {len(atlas_labels.labels)} labels"
    )


def _draw_matrix(
    matrix: np.ndarray,
    labels: list[str],
    networks: list[str],
    colors: list[str],
    output_path: Path,
    title: str | None,
    threshold: float | None,
    transparent: bool,
    show_colorbar: bool,
    block_linewidth: float,
    hemisphere_linewidth: float,
    colorbar_fraction: float,
    colorbar_pad: float,
    font_family: str,
    dpi: int,
) -> None:
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        vmin, vmax = -1.0, 1.0
    else:
        vmax = float(np.nanmax(np.abs(finite)))
        vmin = -vmax
        if vmax == 0:
            vmin, vmax = -1.0, 1.0

    with rc_context({"font.family": "serif", "font.serif": [font_family, "Times", "DejaVu Serif"]}):
        n_rois = matrix.shape[0]
        fig, ax = plt.subplots(figsize=(8, 7), dpi=dpi)

        cmap, norm = _build_diverging_cmap(vmin, vmax, threshold)
        cmap.set_bad("#f5f5f5")
        image = ax.imshow(
            matrix,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            extent=(0, n_rois, n_rois, 0),
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if title:
            ax.set_title(title, fontsize=10, pad=10)

        blocks = _network_blocks(networks, colors)
        _draw_embedded_network_bars(ax, blocks, n_rois)
        if len(blocks) <= 40:
            _draw_network_pair_boxes(ax, blocks, linewidth=block_linewidth)
        _draw_hemisphere_split(ax, labels, linewidth=hemisphere_linewidth)

        bar_width = 0.05 * n_rois
        ax.set_xlim(-bar_width, n_rois)
        ax.set_ylim(n_rois, -bar_width)
        if show_colorbar:
            cbar = fig.colorbar(image, ax=ax, fraction=colorbar_fraction, pad=colorbar_pad)
            cbar.ax.tick_params(labelsize=7)
            cbar.outline.set_visible(False)
            for spine in cbar.ax.spines.values():
                spine.set_visible(False)
        fig.savefig(output_path, bbox_inches="tight", transparent=transparent)
        plt.close(fig)


def _build_diverging_cmap(
    vmin: float,
    vmax: float,
    threshold: float | None,
) -> tuple[mcolors.Colormap, mcolors.TwoSlopeNorm]:
    absmax = max(abs(vmin), abs(vmax), 1e-9)
    norm = mcolors.TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax)
    base = plt.get_cmap("coolwarm")
    values = base(np.linspace(0, 1, 256))

    if threshold is not None and threshold > 0:
        clipped_threshold = min(float(threshold), absmax)
        normalized_low = norm(-clipped_threshold)
        normalized_high = norm(clipped_threshold)
        positions = np.linspace(0, 1, 256)
        gray_band = (positions >= normalized_low) & (positions <= normalized_high)
        values[gray_band] = mcolors.to_rgba(THRESHOLD_BAND_GRAY)

    cmap = mcolors.ListedColormap(values, name="fmriproc_diverging")
    return cmap, norm


def _hemisphere_split_index(labels: list[str]) -> int | None:
    for index, label in enumerate(labels):
        if label.startswith("R_") or "_RH_" in label:
            return index
    return None


def _draw_hemisphere_split(ax: Any, labels: list[str], linewidth: float) -> None:
    split = _hemisphere_split_index(labels)
    if split is None or split <= 0:
        return
    ax.axhline(split, color="black", linewidth=linewidth)
    ax.axvline(split, color="black", linewidth=linewidth)


def _network_blocks(networks: list[str], colors: list[str]) -> list[tuple[int, int, str, str]]:
    blocks: list[tuple[int, int, str, str]] = []
    if not networks:
        return blocks
    start = 0
    current_network = networks[0]
    current_color = colors[0]
    for index, network in enumerate(networks[1:], start=1):
        if network != current_network:
            blocks.append((start, index, current_network, current_color))
            start = index
            current_network = network
            current_color = colors[index]
    blocks.append((start, len(networks), current_network, current_color))
    return blocks


def _draw_embedded_network_bars(ax: Any, blocks: list[tuple[int, int, str, str]], n_rois: int) -> None:
    bar_width = 0.05 * n_rois
    for start, end, _, color in blocks:
        width = end - start
        ax.add_patch(
            Rectangle(
                (start, -bar_width),
                width,
                bar_width,
                facecolor=color,
                edgecolor="gray",
                linewidth=0.25,
                clip_on=False,
            )
        )
        ax.add_patch(
            Rectangle(
                (-bar_width, start),
                bar_width,
                width,
                facecolor=color,
                edgecolor="gray",
                linewidth=0.25,
                clip_on=False,
            )
        )


def _draw_network_pair_boxes(ax: Any, blocks: list[tuple[int, int, str, str]], linewidth: float) -> None:
    grouped: dict[str, list[tuple[int, int]]] = {}
    for start, end, network, _ in blocks:
        grouped.setdefault(network, []).append((start, end))

    for ranges in grouped.values():
        for row_start, row_end in ranges:
            for col_start, col_end in ranges:
                ax.add_patch(
                    Rectangle(
                        (col_start, row_start),
                        col_end - col_start,
                        row_end - row_start,
                        linewidth=linewidth,
                        edgecolor="#303030",
                        facecolor="none",
                    )
                )
