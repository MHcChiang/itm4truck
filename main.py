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

from src.dem_downloader import download_and_crop
from src.signal_estimator import estimate_grid_signal, estimate_hex_grid_signal
from src.visualizer import plot_signal_distribution, plot_hex_signal_map

def main():
    parser = argparse.ArgumentParser(description="Estimate Signal Strength")
    parser.add_argument('--h3', action='store_true', help='Use H3 hex grid instead of square grid')
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
    dem_path = download_and_crop(**target_area, output_dir="dem_data", filename="target_area_dem.tif")
    if not dem_path:
        return

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
        plot_hex_signal_map(dem_path, tx_coords, hex_rssi, target_area)
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
        plot_signal_distribution(dem_path, tx_coords, lon_grid, lat_grid, rssi_matrix, target_area)

if __name__ == "__main__":
    main()