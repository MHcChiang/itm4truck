import sys
import os
import argparse

# Dynamically inject the local itmlogic repository paths
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'itmlogic'))
src_path = os.path.join(repo_path, 'src')
scripts_path = os.path.join(repo_path, 'scripts')

for p in [repo_path, src_path, scripts_path]:
    if p not in sys.path:
        sys.path.insert(0, p)

from itm.dem_downloader import download_and_crop
from itm.signal_estimator import estimate_grid_signal, estimate_hex_grid_signal
from itm.visualizer import plot_signal_distribution, plot_hex_signal_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate Signal Strength")
    parser.add_argument('--h3', action='store_true', help='Use H3 hex grid instead of square grid')
    parser.add_argument(
        '--save-path',
        default='/figure',
        help='Directory where the generated figure should be saved',
    )
    args = parser.parse_args()

    target_area = {
        "west": -81.60,
        "south": 35.38, 
        "east": -81.35, 
        "north": 35.52
        }

    tx_coords = (-81.47, 35.45) # Longitude, Latitude
    
    # Base configuration for ITM Logic
    base_params = {
        'fmhz': 700.0,     # Frequency (MHz)
        'hg': [30.0, 2.0], # Antenna height [TX, RX] (meters)
        'ipol': 1          # Polarization (1=Vertical)
    }

    # Step 1: Download DEM
    dem_path = download_and_crop(**target_area, output_dir="data/dem_data", filename="target_area_dem.tif")
    if not dem_path:
        return

    os.makedirs(args.save_path, exist_ok=True)

    if args.h3:
        # Step 2: Compute hex grid RSSI
        resolution = 9
        hex_rssi = estimate_hex_grid_signal(
            dem_path,
            tx_coords,
            target_area,
            resolution,
            base_params,
        )

        # Step 3: Visualize
        output_path = os.path.join(args.save_path, "hex_signal_coverage.png")
        plot_hex_signal_map(dem_path, tx_coords, hex_rssi, target_area, output_path)
    else:
        # Step 2: Compute grid RSSI
        # Adjust grid_size_n and step_deg to determine the computation density
        grid_size_n = 100
        step_deg = 0.001 
        lon_grid, lat_grid, rssi_matrix = estimate_grid_signal(
            dem_path,
            tx_coords,
            grid_size_n,
            step_deg,
            base_params,
            target_area=target_area,
        )

        # Step 3: Visualize
        output_path = os.path.join(args.save_path, "signal_coverage.png")
        plot_signal_distribution(
            dem_path,
            tx_coords,
            lon_grid,
            lat_grid,
            rssi_matrix,
            target_area,
            output_path,
        )

if __name__ == "__main__":
    main()