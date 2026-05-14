import argparse
from pathlib import Path

from DHS.pop import generate_base_layer
from DHS.viz_pop import visualize_pop_distribution, visualize_town_zones
from DHS.zone import classify_zones, generate_town_grid


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Population demand hotspot pipeline: zone classification + grid generation."
    )

    # Bounding box
    bbox = parser.add_argument_group("Bounding Box")
    bbox.add_argument("--north", type=float, default=38.457485, help="North latitude bound")
    bbox.add_argument("--south", type=float, default=33.464998, help="South latitude bound")
    bbox.add_argument("--east",  type=float, default=-77.000644, help="East longitude bound")
    bbox.add_argument("--west",  type=float, default=-87.998634, help="West longitude bound")

    # Grid resolution
    grid = parser.add_argument_group("Grid Resolution")
    grid.add_argument(
        "--base-res", type=int, default=5,
        help="H3 resolution for the base (low-res) population layer (default: 5, ~110 km edge)",
    )
    grid.add_argument(
        "--refined-res", type=int, default=7,
        help="H3 resolution for the fine-grained town grid (default: 7, ~5 km edge)",
    )

    # Zone classification
    zone = parser.add_argument_group("Zone Classification")
    zone.add_argument(
        "--min-core-pop", type=float, default=10.0,
        help=(
            "Min raw pixel population (people / 100m² pixel) for a cell to be a town seed. "
            "Higher = fewer, more separated town clusters (default: 10)"
        ),
    )
    zone.add_argument(
        "--expansion-rings", type=int, default=1,
        help="H3 rings to expand from each town seed (0 = no buffer, default: 1)",
    )

    # Files
    files = parser.add_argument_group("Files")
    files.add_argument(
        "--pop-file", default="usa_pop_2026_CN_100m_R2025A_v1.tif",
        help="Population GeoTIFF filename inside ./pop_data/ (default: usa_pop_2026_CN_100m_R2025A_v1.tif)",
    )
    files.add_argument("--fig-dir",        default="fig",          help="Output directory for HTML maps (default: fig)")
    files.add_argument("--zones-file",     default="h3_zones.html", help="Zone map filename (default: h3_zones.html)")
    files.add_argument("--pop-dist-file",  default="pop_dist.html", help="Population distribution map filename (default: pop_dist.html)")
    files.add_argument("--data-dir",       default="processed_data",          help="Output directory for CSV grid data (default: processed_data)")

    return parser.parse_args()


def get_project_root() -> Path:
    """Return the repository root whether the script is run from root or DHS."""
    project_root = Path.cwd()
    if not (project_root / "DHS").exists():
        project_root = project_root.parent
    return project_root


def main() -> None:
    """Run the hierarchical zone classification and town-grid pipeline."""
    args = parse_args()

    project_root = get_project_root()
    pop_path = project_root / "pop_data" / args.pop_file
    fig_dir = project_root / args.fig_dir
    data_dir = project_root / args.data_dir
    fig_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    if not pop_path.exists():
        raise FileNotFoundError(f"Population raster not found: {pop_path}")

    bbox = {"north": args.north, "south": args.south, "east": args.east, "west": args.west}

    # Step 1: Base population layer — one sampled population value per low-res hex.
    df_base = generate_base_layer(str(pop_path), bbox, base_res=args.base_res)
    print(f"Base layer cells: {len(df_base)}")

    # Step 2: Zone classification — cluster populated cells into town vs rural.
    df_urban, df_rural, df_clusters = classify_zones(
        df_base, expansion_rings=args.expansion_rings, min_pop=args.min_core_pop
    )
    print(f"\nUrban cells : {len(df_urban)}")
    print(f"Rural cells : {len(df_rural)}")
    print(f"\nCluster summary (top 10 by population):")
    print(df_clusters.head(10).to_string(index=False))

    # Step 3: Fine-grained town grid within classified town areas.
    df_town_grid = generate_town_grid(df_urban, str(pop_path), refined_res=args.refined_res)

    # Step 4a: Zone map — town clusters (low res) + town grid (high res).
    visualize_town_zones(
        df_urban, df_town_grid, df_clusters,
        output_file=str(fig_dir / args.zones_file),
    )

    # Step 4b: Population distribution — all base-layer cells colored by population.
    visualize_pop_distribution(df_base, output_file=str(fig_dir / args.pop_dist_file))

    # Step 5: Save grid data as CSV for GA input.
    df_base.to_csv(data_dir / "base_layer.csv", index=False)
    df_urban.to_csv(data_dir / "urban_cells.csv", index=False)
    df_town_grid.to_csv(data_dir / "town_grid.csv", index=False)
    print(f"\nGrid data saved to {data_dir}/")


if __name__ == "__main__":
    main()
