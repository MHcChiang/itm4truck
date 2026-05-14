"""Urban/rural zoning via H3 spatial region growing.

Pipeline:
  1. Seed Selection  — res-N cells with population above a minimum threshold
  2. Expansion       — k-ring buffer with h3.grid_disk to fill internal gaps
  3. Clustering      — BFS connected components on the expanded urban set
  4. Town Grid       — fine-res children sampled from the population raster
"""

from collections import deque
from typing import Dict, List, Set, Tuple

import h3
import numpy as np
import pandas as pd
import rasterio

DEFAULT_MIN_ACTIVE_POP = 0.1


def select_seeds(df_base: pd.DataFrame, min_pop: float = DEFAULT_MIN_ACTIVE_POP) -> List[str]:
    """Return base-layer cells whose raw pixel population exceeds a minimum threshold.

    Args:
        df_base: Base-layer dataframe from ``generate_base_layer``. Must have
            ``h3_index`` and ``raw_pixel_pop`` columns.
        min_pop: Minimum raw pixel population for a cell to qualify as a seed.

    Returns:
        List of H3 cell IDs.
    """
    return df_base[df_base["raw_pixel_pop"] > min_pop]["h3_index"].tolist()


def expand_seeds(seeds: List[str], k: int = 1) -> Set[str]:
    """Expand seed cells outward by k rings using H3 grid_disk.

    The expansion connects nearby seed cells into continuous town zones and
    heals internal gaps such as parks or industrial areas that appear empty in
    raw population data.

    Args:
        seeds: Seed cell IDs from ``select_seeds``.
        k: Number of rings to expand (1 ring ≈ one cell-width buffer).

    Returns:
        Set of H3 cell IDs covering the expanded town footprint.
    """
    expanded: Set[str] = set()
    for seed in seeds:
        expanded.update(h3.grid_disk(seed, k))
    return expanded


def _bfs_components(cells: Set[str]) -> Dict[str, int]:
    """Assign cluster IDs via BFS over H3 adjacency.

    Uses h3.grid_ring (edge neighbors only) so no self-exclusion check is needed.

    Args:
        cells: Set of H3 cell IDs to cluster.

    Returns:
        Dict mapping each cell ID to an integer cluster label (0-indexed).
    """
    label: Dict[str, int] = {}
    cluster_id = 0

    for start in cells:
        if start in label:
            continue
        queue = deque([start])
        label[start] = cluster_id
        while queue:
            current = queue.popleft()
            for neighbor in h3.grid_ring(current, 1):
                if neighbor in cells and neighbor not in label:
                    label[neighbor] = cluster_id
                    queue.append(neighbor)
        cluster_id += 1

    return label


def classify_zones(
    df_base: pd.DataFrame,
    expansion_rings: int = 1,
    min_pop: float = DEFAULT_MIN_ACTIVE_POP,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Classify base-layer H3 cells into town clusters and rural remainder.

    Steps:
      1. Seed selection — base cells with ``raw_pixel_pop > min_pop``.
      2. Expansion — ``expansion_rings``-ring buffer via ``h3.grid_disk``.
      3. Intersection — restrict expanded set to cells present in ``df_base``.
      4. Clustering — BFS connected components on the restricted town set.

    Args:
        df_base: Base-layer dataframe from ``generate_base_layer``.
            Must have ``h3_index``, ``pop``, and ``raw_pixel_pop`` columns.
        expansion_rings: Number of H3 rings to grow each seed outward.
        min_pop: Minimum raw pixel population used to identify seed cells.

    Returns:
        A tuple ``(df_urban, df_rural, df_clusters)``:

        - ``df_urban``: Rows from ``df_base`` classified as town, with an
          added ``cluster_id`` (int) column.
        - ``df_rural``: Rows from ``df_base`` classified as rural (no cluster).
        - ``df_clusters``: Summary table — ``cluster_id``, ``n_cells``,
          ``total_pop``, ``centroid_lat``, ``centroid_lon``.
    """
    print("--- Zone Classification ---")

    seeds = select_seeds(df_base, min_pop)
    print(f"Seeds (populated cells): {len(seeds)}")

    expanded = expand_seeds(seeds, k=expansion_rings)
    print(f"Expanded town footprint (before intersection): {len(expanded)} cells")

    base_cells = set(df_base["h3_index"])
    urban_cells = expanded & base_cells
    rural_cells = base_cells - urban_cells
    print(f"Town cells (intersected with base layer): {len(urban_cells)}")
    print(f"Rural cells: {len(rural_cells)}")

    cluster_map = _bfs_components(urban_cells)
    n_clusters = len(set(cluster_map.values()))
    print(f"Town clusters found: {n_clusters}")

    df_urban = df_base[df_base["h3_index"].isin(urban_cells)].copy()
    df_urban["cluster_id"] = df_urban["h3_index"].map(cluster_map)

    df_rural = df_base[df_base["h3_index"].isin(rural_cells)].copy()

    df_clusters = _build_cluster_summary(df_urban)

    return df_urban, df_rural, df_clusters


def _build_cluster_summary(df_urban: pd.DataFrame) -> pd.DataFrame:
    """Compute per-cluster statistics from the town dataframe.

    Args:
        df_urban: Town cells dataframe with ``cluster_id``, ``h3_index``, and
            ``pop`` columns.

    Returns:
        DataFrame with columns ``cluster_id``, ``n_cells``, ``total_pop``,
        ``centroid_lat``, ``centroid_lon`` — one row per cluster, sorted by
        descending ``total_pop``.
    """
    rows = []
    for cid, group in df_urban.groupby("cluster_id"):
        latlngs = np.array([h3.cell_to_latlng(cell) for cell in group["h3_index"]])
        centroid_lat, centroid_lon = latlngs.mean(axis=0)
        rows.append(
            {
                "cluster_id": cid,
                "n_cells": len(group),
                "total_pop": group["pop"].sum(),
                "centroid_lat": centroid_lat,
                "centroid_lon": centroid_lon,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("total_pop", ascending=False).reset_index(drop=True)
    return df


def generate_town_grid(
    df_urban: pd.DataFrame,
    tif_path: str,
    refined_res: int = 8,
) -> pd.DataFrame:
    """Generate a fine-grained population grid for all town cells.

    For each town cell, enumerates its children at ``refined_res`` and samples
    the population raster in a single batched call. Only children with positive
    population are kept.

    Args:
        df_urban: Town cells dataframe with ``h3_index`` and ``cluster_id``
            columns (output of ``classify_zones``).
        tif_path: Path to the population GeoTIFF raster.
        refined_res: H3 resolution for child cells (default 8).

    Returns:
        DataFrame with columns ``h3_index``, ``pop``, ``res``, ``parent``, and
        ``cluster_id``.
    """
    print(f"--- Generating Town Grid (Res {refined_res}) ---")
    cluster_lookup = df_urban.set_index("h3_index")["cluster_id"].to_dict()

    all_children: List[str] = []
    all_coords: List[Tuple[float, float]] = []
    all_parents: List[str] = []
    all_cluster_ids: List[int] = []

    for parent_hex, cluster_id in cluster_lookup.items():
        children = list(h3.cell_to_children(parent_hex, refined_res))
        coords = [(lon, lat) for lat, lon in (h3.cell_to_latlng(c) for c in children)]
        all_children.extend(children)
        all_coords.extend(coords)
        all_parents.extend([parent_hex] * len(children))
        all_cluster_ids.extend([cluster_id] * len(children))

    refined_data = []
    with rasterio.open(tif_path) as src:
        samples = [v[0] for v in src.sample(all_coords)]

    for child, pop, parent, cid in zip(all_children, samples, all_parents, all_cluster_ids):
        if pop > 0:
            refined_data.append(
                {
                    "h3_index": child,
                    "pop": float(pop),
                    "res": refined_res,
                    "parent": parent,
                    "cluster_id": cid,
                }
            )

    df_town_grid = pd.DataFrame(refined_data)
    print(f"Town grid complete: {len(df_town_grid)} hexes.")
    return df_town_grid
