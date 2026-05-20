"""Population demand hotspot pipeline — zone classification and grid generation.

Runs the full data processing pipeline and saves results to processed_data/.
For visualization, run viz_processed.py after this script completes.

Usage:
    python process_pop.py
    python process_pop.py --config config.yaml --min-core-pop 5.0
"""

import argparse
from pathlib import Path

import pandas as pd

from config import load_config
from candidates.candidates import build_rural_candidates
from DHS.anomaly import count_anomaly_per_cell
from DHS.pop import generate_base_layer
from DHS.zone import classify_zones, generate_town_grid


def parse_args(cfg: dict) -> argparse.Namespace:
    """Parse command-line arguments, using config.yaml values as defaults."""
    parser = argparse.ArgumentParser(
        description="Population demand hotspot pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml", help="Project YAML config file")

    bbox = parser.add_argument_group("Bounding Box")
    bbox.add_argument("--north", type=float)
    bbox.add_argument("--south", type=float)
    bbox.add_argument("--east",  type=float)
    bbox.add_argument("--west",  type=float)

    grid = parser.add_argument_group("Grid Resolution")
    grid.add_argument("--base-res",    type=int, help="H3 base resolution for rural layer")
    grid.add_argument("--refined-res", type=int, help="H3 resolution for town fine grid")

    zone = parser.add_argument_group("Zone Classification")
    zone.add_argument("--min-core-pop",    type=float, default=10.0,
                      help="Min raw pixel pop for a cell to be a town seed")
    zone.add_argument("--expansion-rings", type=int, default=1,
                      help="H3 rings to expand from each town seed")

    files = parser.add_argument_group("Files")
    files.add_argument("--pop-file",     help="Population GeoTIFF inside ./data/")
    files.add_argument("--anomaly-file", help="Parquet with anomaly points (lat, lon)")
    files.add_argument("--hifld-csv",    help="HIFLD cellular tower CSV")
    files.add_argument("--out-dir",      help="Output directory for CSV results")

    # Apply config values as defaults so CLI args still override
    parser.set_defaults(
        north        = cfg["region"]["north"],
        south        = cfg["region"]["south"],
        east         = cfg["region"]["east"],
        west         = cfg["region"]["west"],
        base_res     = cfg["h3"]["base_res"],
        refined_res  = cfg["h3"]["refined_res"],
        pop_file     = cfg["paths"]["pop_file"],
        anomaly_file = cfg["paths"]["anomaly_file"],
        hifld_csv    = cfg["paths"]["hifld_csv"],
        out_dir      = cfg["paths"]["out_dir"],
    )

    return parser.parse_args()


def get_project_root() -> Path:
    """Return the repository root whether the script is run from root or a subdirectory."""
    root = Path.cwd()
    if not (root / "DHS").exists():
        root = root.parent
    return root


def main() -> None:
    """Run the hierarchical zone classification and town-grid pipeline."""
    cfg = load_config()
    args = parse_args(cfg)

    project_root = get_project_root()
    pop_path  = project_root / args.pop_file
    data_dir  = project_root / args.out_dir
    data_dir.mkdir(exist_ok=True)

    if not pop_path.exists():
        raise FileNotFoundError(f"Population raster not found: {pop_path}")

    bbox = {"north": args.north, "south": args.south, "east": args.east, "west": args.west}

    # Step 1: Base population layer
    df_base = generate_base_layer(str(pop_path), bbox, base_res=args.base_res)
    print(f"Base layer cells: {len(df_base)}")

    # Step 2: Zone classification
    df_urban, df_rural, df_clusters = classify_zones(
        df_base, expansion_rings=args.expansion_rings, min_pop=args.min_core_pop
    )
    print(f"Urban cells : {len(df_urban)}")
    print(f"Rural cells : {len(df_rural)}")
    print(df_clusters.head(10).to_string(index=False))

    # Step 3: Fine-grained town grid
    df_town_grid = generate_town_grid(df_urban, str(pop_path), refined_res=args.refined_res)

    # Step 4: Anomaly counts per base-layer cell
    anomaly_path = project_root / args.anomaly_file
    df_anomaly = None
    if anomaly_path.exists():
        df_anomaly = count_anomaly_per_cell(
            str(anomaly_path), df_base["h3_index"].tolist(), args.base_res
        )
    else:
        print(f"Anomaly file not found, skipping: {anomaly_path}")

    # Step 5: Rural candidate towers (HIFLD + demand hotspots)
    hifld_path = project_root / args.hifld_csv
    if hifld_path.exists():
        build_rural_candidates(
            hifld_csv=str(hifld_path),
            df_rural=df_rural,
            df_anomaly=df_anomaly,
            bbox=bbox,
            output_path=str(data_dir / "rural_candidates.csv"),
        )
    else:
        print(f"HIFLD CSV not found, skipping candidates: {hifld_path}")

    # Step 6: Save all grid data
    df_urban["grid_type"] = "urban"
    df_rural["grid_type"] = "rural"
    df_base_tagged = df_base.merge(
        pd.concat([df_urban[["h3_index", "grid_type"]], df_rural[["h3_index", "grid_type"]]]),
        on="h3_index", how="left",
    )

    df_base_tagged.to_csv(data_dir / "base_layer.csv",    index=False)
    df_urban.to_csv(      data_dir / "urban_cells.csv",   index=False)
    df_rural.to_csv(      data_dir / "rural_cells.csv",   index=False)
    df_town_grid.to_csv(  data_dir / "town_grid.csv",     index=False)
    df_clusters.to_csv(   data_dir / "clusters.csv",      index=False)
    if df_anomaly is not None:
        df_anomaly.to_csv(data_dir / "anomaly_counts.csv", index=False)

    print(f"\nAll outputs saved to {data_dir}/")
    print("Run viz_processed.py to generate maps.")


if __name__ == "__main__":
    main()
