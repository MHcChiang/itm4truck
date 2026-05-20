"""Visualize processed pipeline outputs as interactive Folium maps.

Loads pre-computed CSVs from processed_data/ and generates:
  - h3_zones.html   — town areas, fine grid, anomalies, candidates, scatter
  - coverage.html   — HIFLD/demand tower coverage over rural demand cells
  - pop_dist.html   — base-layer population heatmap

Run process_pop.py and precal_itm.py first to generate the input files.

Usage:
    python viz_processed.py
    python viz_processed.py --config config.yaml --zones-file my_zones.html
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_config
from viz.viz_pop import (
    add_anomaly_layer,
    add_candidates_layer,
    add_coverage_layer,
    add_scatter_layer,
    add_town_areas_layer,
    add_town_grid_layer,
    create_zone_map,
    finalize_map,
    visualize_pop_distribution,
)


def parse_args(cfg: dict) -> argparse.Namespace:
    """Parse CLI arguments with config.yaml values as defaults."""
    parser = argparse.ArgumentParser(
        description="Generate Folium maps from processed pipeline outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml", help="Project YAML config file")

    files = parser.add_argument_group("Files")
    files.add_argument("--out-dir",       help="Directory containing processed CSVs")
    files.add_argument("--fig-dir",       help="Output directory for HTML maps")
    files.add_argument("--scatter-file",  help="Parquet with raw measurement points (lat, lon)")
    files.add_argument("--zones-file",    default="h3_zones.html",
                       help="Output filename for the zone map")
    files.add_argument("--coverage-file", default="coverage.html",
                       help="Output filename for the coverage map")
    files.add_argument("--pop-dist-file", default="pop_dist.html",
                       help="Output filename for the population distribution map")

    parser.set_defaults(
        out_dir      = cfg["paths"]["out_dir"],
        fig_dir      = cfg["paths"]["fig_dir"],
        scatter_file = cfg["paths"]["scatter_file"],
    )
    return parser.parse_args()


def _load_csv_if_exists(path: Path) -> pd.DataFrame:
    """Return the CSV as a DataFrame, or an empty DataFrame if the file is missing."""
    if path.exists():
        return pd.read_csv(path)
    print(f"File not found, skipping: {path}")
    return pd.DataFrame()


def _load_coverage_npz(path: Path):
    """Load coverage_matrix.npz and return (coverage, candidate_h3, demand_h3).

    Returns (None, None, None) if the file is missing.
    """
    if not path.exists():
        print(f"Coverage matrix not found, skipping: {path}")
        return None, None, None
    npz = np.load(str(path), allow_pickle=True)
    coverage     = npz["coverage"]               # (n_cand, n_dem) uint8
    candidate_h3 = npz["candidates"].tolist()    # list of H3 strings
    demand_h3    = npz["demand"].tolist()         # list of H3 strings
    print(f"Coverage matrix loaded: {coverage.shape[0]} candidates × {coverage.shape[1]} demand cells")
    return coverage, candidate_h3, demand_h3


def main() -> None:
    """Load processed data and render all maps."""
    cfg  = load_config()
    args = parse_args(cfg)

    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(exist_ok=True)

    # --- Required data ---
    df_base      = _load_csv_if_exists(out_dir / "base_layer.csv")
    df_urban     = _load_csv_if_exists(out_dir / "urban_cells.csv")
    df_clusters  = _load_csv_if_exists(out_dir / "clusters.csv")
    df_town_grid = _load_csv_if_exists(out_dir / "town_grid.csv")

    # Reconstruct cluster summary from urban cells if clusters.csv is missing
    if df_clusters.empty and not df_urban.empty and "cluster_id" in df_urban.columns:
        print("clusters.csv not found — reconstructing from urban_cells.csv")
        df_clusters = (
            df_urban.groupby("cluster_id", as_index=False)
            .agg(n_cells=("h3_index", "count"), total_pop=("pop", "sum"))
            .sort_values("total_pop", ascending=False)
            .reset_index(drop=True)
        )

    if df_urban.empty:
        raise FileNotFoundError(
            f"urban_cells.csv not found in {out_dir}. Run process_pop.py first."
        )

    # --- Optional layers ---
    df_anomaly    = _load_csv_if_exists(out_dir / "anomaly_counts.csv")
    df_candidates = _load_csv_if_exists(out_dir / "rural_candidates.csv")

    df_scatter = pd.DataFrame()
    scatter_path = Path(args.scatter_file)
    if scatter_path.exists():
        df_scatter = pd.read_parquet(str(scatter_path), columns=["lat", "lon"])
        print(f"Scatter points loaded: {len(df_scatter)}")
    else:
        print(f"Scatter file not found, skipping: {scatter_path}")

    coverage, candidate_h3, demand_h3 = _load_coverage_npz(out_dir / "coverage_matrix.npz")

    # --- Map 1: Zone map ---
    zone_map = create_zone_map(df_urban)
    add_town_areas_layer(zone_map, df_urban, df_clusters)
    add_town_grid_layer(zone_map, df_town_grid)
    if not df_anomaly.empty:
        add_anomaly_layer(zone_map, df_anomaly)
    if not df_scatter.empty:
        add_scatter_layer(zone_map, df_scatter)
    if not df_candidates.empty:
        add_candidates_layer(zone_map, df_candidates)
    finalize_map(zone_map, str(fig_dir / args.zones_file))

    # --- Map 2: Coverage map ---
    if coverage is not None and not df_candidates.empty:
        cov_map = create_zone_map(df_urban)
        # HIFLD coverage (on by default)
        add_coverage_layer(
            cov_map, coverage, candidate_h3, demand_h3,
            df_candidates, source_filter="hifld", show=True,
        )
        # Demand-proposed coverage (off by default — toggle to compare)
        if (df_candidates["source"] == "demand").any():
            add_coverage_layer(
                cov_map, coverage, candidate_h3, demand_h3,
                df_candidates, source_filter="demand", show=False,
            )
        # Overlay candidate tower markers for context
        add_candidates_layer(cov_map, df_candidates, show=True)
        finalize_map(cov_map, str(fig_dir / args.coverage_file))

    # --- Map 3: Population distribution ---
    if not df_base.empty:
        visualize_pop_distribution(df_base, output_file=str(fig_dir / args.pop_dist_file))


if __name__ == "__main__":
    main()
