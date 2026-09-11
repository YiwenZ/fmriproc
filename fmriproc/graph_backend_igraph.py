"""Graph-theory metrics backend using python-igraph.

Reads a run-level Fisher-z FC matrix, builds undirected unweighted binary
graphs at configured density thresholds, computes global graph metrics and
small-world normalisation, and writes adjacency matrices, threshold-wise and
summary tables, and a JSON metadata file.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def compute_graph_metrics(
    fc_fisherz_path: Path,
    fc_metadata_json_path: Path,
    atlas: str,
    subject: str,
    session: str,
    task: str,
    thresholds_path: Path,
    summary_path: Path,
    metadata_out_path: Path,
    raw_dir: Path,
    log_path: Path,
    cfg: dict[str, Any],
) -> None:
    """Run the full graph-metrics pipeline for one subject-session-task-atlas.

    Parameters
    ----------
    fc_fisherz_path:
        Path to the Fisher-z FC matrix TSV.
    fc_metadata_json_path:
        Path to the FC metadata JSON (for provenance).
    atlas:
        Atlas name.
    subject, session, task:
        Run identifiers.
    thresholds_path, summary_path, metadata_out_path:
        Output paths for the threshold table, summary table, and JSON metadata.
    raw_dir:
        Directory for raw adjacency matrices and log.
    cfg:
        Full pipeline config (reads ``metrics.graph``).
    """
    graph_cfg = cfg.get("metrics", {}).get("graph", {})
    backend = graph_cfg.get("backend", "python_igraph")
    if backend != "python_igraph":
        raise RuntimeError(
            f"Unsupported graph backend: {backend!r}. "
            f"Only 'python_igraph' is available."
        )
    densities = graph_cfg.get("densities", [0.10, 0.15, 0.20, 0.25])
    n_random = int(graph_cfg.get("random_networks", 100))
    base_seed = int(graph_cfg.get("seed", 112))
    seed_strategy = graph_cfg.get("seed_strategy", "run_hash")
    require_leiden = bool(graph_cfg.get("require_leiden", False))
    keep_adjacency = bool(graph_cfg.get("keep_adjacency", True))

    _check_igraph_available()

    # Read and validate the Fisher-z matrix.
    labels, matrix = _read_fisherz_matrix(fc_fisherz_path)
    _validate_square_symmetric(matrix, fc_fisherz_path)
    n_rois = matrix.shape[0]
    n_possible = n_rois * (n_rois - 1) // 2

    # Extract upper-triangle positive edges (z > 0).
    triu_indices = np.triu_indices(n_rois, k=1)
    z_values = matrix[triu_indices]
    positive_mask = z_values > 0
    if not np.any(positive_mask):
        raise RuntimeError(
            f"No positive (z > 0) edges in {fc_fisherz_path}. "
            f"Cannot construct graph at any density."
        )
    pos_indices = (triu_indices[0][positive_mask], triu_indices[1][positive_mask])
    pos_z = z_values[positive_mask]
    sort_order = np.argsort(pos_z)[::-1]  # descending

    # Effective seed.
    effective_seed = _effective_seed(base_seed, seed_strategy, subject, session, task, atlas)

    # Compute per-density metrics.
    threshold_rows: list[dict[str, Any]] = []
    summary_metrics: dict[str, list[float]] = {
        "clustering_coefficient": [],
        "transitivity_global": [],
        "characteristic_path_length": [],
        "global_efficiency": [],
        "modularity": [],
        "normalized_clustering": [],
        "normalized_path_length": [],
        "small_worldness": [],
    }
    community_method = ""
    n_fragmented = 0
    max_cc = 0
    min_plpc = 1.0
    plpc_values: list[float] = []
    thresholds_included: list[str] = []

    for density in densities:
        n_edges = math.ceil(density * n_possible)
        if n_edges > len(pos_z):
            raise RuntimeError(
                f"Insufficient positive edges for density {density}: "
                f"need {n_edges}, have {len(pos_z)} positive edges. "
                f"Source: {fc_fisherz_path}"
            )

        # Select top edges, build adjacency.
        top_n = sort_order[:n_edges]
        adj = np.zeros((n_rois, n_rois), dtype=np.int32)
        adj[pos_indices[0][top_n], pos_indices[1][top_n]] = 1
        adj[pos_indices[1][top_n], pos_indices[0][top_n]] = 1  # undirected

        realized_density = float(np.sum(adj)) / (2 * n_possible)

        # Write adjacency if requested.
        pct = int(round(density * 100))
        if keep_adjacency:
            adj_dir = raw_dir / f"{atlas}_density-{pct}"
            adj_dir.mkdir(parents=True, exist_ok=True)
            _write_adjacency_tsv(adj_dir / "adjacency.tsv", adj, labels)

        # Compute metrics.
        row, cm = _compute_threshold_metrics(
            adj, labels, density, n_rois, n_edges, realized_density,
            subject, session, task, atlas, effective_seed, n_random,
            require_leiden,
        )

        # Track community method (should be consistent across thresholds).
        community_method = community_method or cm

        # Track fragmentation.
        connected_components = int(row.get("connected_components", 1))
        plpc = float(row.get("path_length_pair_coverage", 1.0))
        if connected_components > 1:
            n_fragmented += 1
        max_cc = max(max_cc, connected_components)
        min_plpc = min(min_plpc, plpc)
        plpc_values.append(plpc)

        threshold_rows.append(row)
        thresholds_included.append(f"{density:.2f}")
        for key in summary_metrics:
            val = row.get(key)
            if val is not None:
                try:
                    fv = float(val)
                    if not math.isnan(fv):
                        summary_metrics[key].append(fv)
                except (ValueError, TypeError):
                    pass

    # Compute summary row.
    summary_row = _build_summary_row(
        subject, session, task, atlas, densities, n_rois, summary_metrics,
        community_method, n_fragmented, max_cc, min_plpc, plpc_values,
        thresholds_included,
    )

    # Write outputs.
    _write_threshold_tsv(thresholds_path, threshold_rows)
    _write_summary_tsv(summary_path, summary_row)

    # Resolve versions.
    import igraph
    try:
        import leidenalg
        leiden_ver = leidenalg.__version__
    except ImportError:
        leiden_ver = None

    metadata = {
        "subject": subject,
        "session": session,
        "task": task,
        "atlas": atlas,
        "backend": backend,
        "python_version": __import__("sys").version,
        "igraph_version": igraph.__version__,
        "leidenalg_version": leiden_ver,
        "community_method": community_method,
        "fallback_reason": _fallback_reason(require_leiden, leiden_ver, community_method),
        "base_seed": base_seed,
        "effective_seed": effective_seed,
        "seed_strategy": seed_strategy,
        "densities": densities,
        "random_networks": n_random,
        "require_leiden": require_leiden,
        "keep_adjacency": keep_adjacency,
        "fc_variant": cfg.get("metrics", {}).get("input", {}).get("fc_variant", "raw"),
        "fc_source": str(fc_fisherz_path),
        "fc_metadata_source": str(fc_metadata_json_path),
        "n_rois": n_rois,
        "n_possible_edges": n_possible,
        "n_positive_edges": int(len(pos_z)),
        "n_fragmented_thresholds": n_fragmented,
        "max_connected_components": max_cc,
        "min_path_length_pair_coverage": min_plpc,
        "thresholds_included": thresholds_included,
    }
    metadata_out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_out_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    # Write log.
    _write_log(log_path, metadata)


# ---------------------------------------------------------------------------
# Matrix I/O & validation
# ---------------------------------------------------------------------------


def _read_fisherz_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    """Read a Fisher-z FC matrix TSV, return (labels, matrix)."""
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, [])
        rows = []
        for row in reader:
            rows.append([float(v) for v in row[1:]])  # skip label column
    labels = header[1:] if len(header) > 1 else [f"ROI_{i:03d}" for i in range(len(rows))]
    return labels, np.array(rows, dtype=float)


def _validate_square_symmetric(matrix: np.ndarray, source: Path) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Fisher-z matrix is not square: {source} (shape={matrix.shape})")
    if not np.allclose(matrix, matrix.T, atol=1e-9):
        max_diff = float(np.max(np.abs(matrix - matrix.T)))
        if max_diff > 1e-6:
            raise ValueError(
                f"Fisher-z matrix is asymmetric beyond tolerance: {source} (max|diff|={max_diff:.2e})"
            )


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------


def _effective_seed(
    base_seed: int, strategy: str, subject: str, session: str, task: str, atlas: str
) -> int:
    """Derive a deterministic run-specific seed from the base seed."""
    if strategy == "run_hash":
        tag = f"{subject}_{session}_{task}_{atlas}"
        extra = int(hashlib.md5(tag.encode()).hexdigest()[:8], 16)
        return base_seed + (extra % 1000000)
    return base_seed


# ---------------------------------------------------------------------------
# Per-threshold metrics
# ---------------------------------------------------------------------------


def _compute_threshold_metrics(
    adj: np.ndarray,
    labels: list[str],
    density: float,
    n_rois: int,
    n_edges: int,
    realized_density: float,
    subject: str,
    session: str,
    task: str,
    atlas: str,
    seed: int,
    n_random: int,
    require_leiden: bool,
) -> tuple[dict[str, Any], str]:
    """Compute global graph metrics for one density threshold."""
    import igraph

    g = igraph.Graph.Adjacency(adj.tolist(), mode="undirected")

    # Basic metrics.
    cc = float(g.transitivity_avglocal_undirected(mode="zero"))  # mean local CC, isolates→0
    trans_global = float(g.transitivity_undirected())  # global transitivity

    # Efficiency and path length.
    n = g.vcount()
    n_pairs = n * (n - 1) // 2
    dist_matrix = np.array(g.distances(weights=None))
    iu = np.triu_indices(n, k=1)
    all_dists = dist_matrix[iu]
    finite_mask = np.isfinite(all_dists) & (all_dists > 0)
    finite_dists = all_dists[finite_mask]
    plpc = len(finite_dists) / n_pairs if n_pairs > 0 else 0.0
    if len(finite_dists) == 0:
        raise RuntimeError(f"No finite off-diagonal distances for density {density}")

    # Global efficiency: unreachable pairs contribute 0 (1/∞ = 0),
    # divided by all possible pairs.
    eff = float(np.sum(1.0 / finite_dists)) / n_pairs if n_pairs > 0 else 0.0
    cpl = float(np.mean(finite_dists))  # characteristic path length (finite pairs only)

    # Connected components.
    comps = g.connected_components()
    n_comps = len(comps)

    # Community detection.
    community_method, modularity = _detect_communities(g, require_leiden)

    # Random graphs for normalization — seed igraph's RNG with our numpy generator.
    rng = np.random.default_rng(seed)
    _seed_igraph_rng(rng)
    rand_ccs = []
    rand_cpls = []
    for _ in range(n_random):
        rg = igraph.GraphBase.Erdos_Renyi(n=n, m=n_edges, directed=False, loops=False)
        rand_ccs.append(float(rg.transitivity_avglocal_undirected(mode="zero")))
        # Path length for random graph.
        r_dist_matrix = np.array(rg.distances(weights=None))
        r_iu = np.triu_indices(n, k=1)
        r_all = r_dist_matrix[r_iu]
        r_finite = r_all[np.isfinite(r_all) & (r_all > 0)]
        rand_cpls.append(float(np.mean(r_finite)) if len(r_finite) > 0 else float("inf"))

    mean_rand_cc = float(np.mean(rand_ccs))
    mean_rand_cpl = float(np.mean(rand_cpls))

    norm_cc = cc / mean_rand_cc if mean_rand_cc > 0 else 0.0
    norm_cpl = cpl / mean_rand_cpl if mean_rand_cpl > 0 else float("inf")
    sw = norm_cc / norm_cpl if norm_cpl > 0 else 0.0

    row: dict[str, Any] = {
        "subject": subject,
        "session": session,
        "task": task,
        "atlas": atlas,
        "density": f"{density:.2f}",
        "n_rois": n_rois,
        "n_edges": n_edges,
        "realized_density": f"{realized_density:.6f}",
        "clustering_coefficient": f"{cc:.6f}",
        "transitivity_global": f"{trans_global:.6f}",
        "characteristic_path_length": f"{cpl:.6f}",
        "global_efficiency": f"{eff:.6f}",
        "modularity": f"{modularity:.6f}",
        "normalized_clustering": f"{norm_cc:.6f}",
        "normalized_path_length": f"{norm_cpl:.6f}",
        "small_worldness": f"{sw:.6f}",
        "community_method": community_method,
        "n_random": n_random,
        "connected_components": n_comps,
        "path_length_pair_coverage": f"{plpc:.6f}",
        "_mean_rand_cc": f"{mean_rand_cc:.6f}",
        "_mean_rand_cpl": f"{mean_rand_cpl:.6f}",
    }
    return row, community_method


def _detect_communities(g, require_leiden: bool) -> tuple[str, float]:
    """Detect communities and compute modularity. Returns (method, modularity)."""
    try:
        import leidenalg

        partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
        return ("leiden", float(g.modularity(partition)))
    except ImportError:
        if require_leiden:
            raise RuntimeError("leidenalg is required (require_leiden: true) but not installed")
    except Exception:
        if require_leiden:
            raise

    # Fallback: Louvain → fastgreedy
    try:
        partition = g.community_multilevel(weights=None)
        return ("louvain", float(g.modularity(partition)))
    except Exception:
        try:
            partition = g.community_fastgreedy(weights=None).as_clustering()
            return ("fastgreedy", float(g.modularity(partition)))
        except Exception:
            raise RuntimeError("No community detection method available")


def _fallback_reason(require_leiden: bool, leiden_ver: str | None, method: str) -> str:
    if require_leiden:
        return ""
    if method == "leiden":
        return ""
    try:
        import leidenalg
        return ""
    except ImportError:
        return f"leidenalg not available; used {method}"
    return f"leiden failed; used {method}"


# ---------------------------------------------------------------------------
# Summary row
# ---------------------------------------------------------------------------


def _build_summary_row(
    subject: str, session: str, task: str, atlas: str,
    densities: list[float], n_rois: int,
    summary_metrics: dict[str, list[float]],
    community_method: str,
    n_fragmented: int, max_cc: int, min_plpc: float,
    plpc_values: list[float],
    thresholds_included: list[str],
) -> dict[str, Any]:
    """Build the cross-threshold summary row."""
    row: dict[str, Any] = {
        "subject": subject,
        "session": session,
        "task": task,
        "atlas": atlas,
        "densities": ",".join(f"{d:.2f}" for d in densities),
        "n_thresholds": len(densities),
        "n_rois": n_rois,
    }
    for key, vals in summary_metrics.items():
        row[key] = f"{float(np.mean(vals)):.6f}" if vals else "nan"
    row["community_method"] = community_method
    row["n_fragmented_thresholds"] = n_fragmented
    row["max_connected_components"] = max_cc
    row["min_path_length_pair_coverage"] = f"{min_plpc:.6f}"
    row["mean_path_length_pair_coverage"] = f"{float(np.mean(plpc_values)):.6f}" if plpc_values else "nan"
    n_failed = len(densities) - len(thresholds_included)
    row["n_failed_thresholds"] = n_failed
    row["thresholds_included"] = ",".join(thresholds_included)
    return row


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _write_adjacency_tsv(path: Path, adj: np.ndarray, labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["roi_i", "roi_j", "connected"])
        triu = np.triu_indices(adj.shape[0], k=1)
        for i, j in zip(triu[0], triu[1]):
            if adj[i, j]:
                writer.writerow([labels[i], labels[j], 1])
            else:
                writer.writerow([labels[i], labels[j], 0])


def _write_threshold_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary_tsv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerow(row)


def _write_log(path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{k}: {v}" for k, v in sorted(metadata.items())]
    path.write_text("\n".join(lines), encoding="utf-8")


def _seed_igraph_rng(rng: np.random.Generator) -> None:
    """Seed igraph's internal random number generator from a numpy Generator."""
    import igraph

    class _IgraphRNG:
        def random(self) -> float:
            return float(rng.random())

        def randint(self, a: int, b: int) -> int:
            return int(rng.integers(a, b + 1))

        def gauss(self, mu: float, sigma: float) -> float:
            return float(rng.normal(mu, sigma))

    igraph.set_random_number_generator(_IgraphRNG())


def _check_igraph_available() -> None:
    try:
        import igraph  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "python-igraph is required for graph metrics. "
            "Install it with: pip install python-igraph"
        )
