"""Pre-compute ITM coverage matrix for the rural Maximum Coverage Problem.

Design choices:
- Distance pruning: pairs beyond max_coverage_km skip ITM (skips ~85–90% of pairs).
- Windowed DEM: each worker loads only the patch needed for one candidate
  (~60 km square, ~3.5 MB) instead of holding the full 90 m raster in RAM.
- ProcessPoolExecutor: one task = one candidate (all its in-range demand pairs).
  True CPU parallelism — avoids GIL limitation of ThreadPoolExecutor for
  CPU-bound itmlogic calls.

Coverage matrix output (n_candidates × n_demand):
  - Rows  → candidate towers from rural_candidates.csv (Res-8 hilltop coords)
  - Cols  → demand cells from rural_cells.csv (Res-6 centers)
  - Value → 1 if RSSI ≥ threshold, 0 otherwise

Usage:
    python precal_itm.py
    python precal_itm.py --config config.yaml --max-coverage-km 30
"""

import argparse
import contextlib
import io
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

# Inject itmlogic paths before importing signal_estimator.
# Mirrors the same sys.path setup in main.py so workers also resolve
# 'scripts.*' imports when ProcessPoolExecutor re-imports this module.
_ITMLOGIC_ROOT = Path(__file__).parent / "itmlogic"
for _p in [_ITMLOGIC_ROOT, _ITMLOGIC_ROOT / "src", _ITMLOGIC_ROOT / "scripts"]:
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)

import h3
import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.windows import Window, from_bounds
from tqdm import tqdm

from config import load_config
from itm.dem_downloader import download_and_crop
from itm.signal_estimator import calculate_memory_p2p_rssi

_GEOD = pyproj.Geod(ellps="WGS84")


# ---------------------------------------------------------------------------
# Distance pruning
# ---------------------------------------------------------------------------

def _within_range_mask(
    tx_lat: float,
    tx_lon: float,
    rx_lats: np.ndarray,
    rx_lons: np.ndarray,
    max_km: float,
) -> np.ndarray:
    """Boolean mask: True where geodesic TX→RX distance ≤ max_km.

    Vectorised over all demand cells in one pyproj call.

    Args:
        tx_lat: Transmitter latitude.
        tx_lon: Transmitter longitude.
        rx_lats: Receiver latitudes, shape (N,).
        rx_lons: Receiver longitudes, shape (N,).
        max_km: Maximum coverage radius in km.

    Returns:
        Boolean array of shape (N,).
    """
    _, _, dists_m = _GEOD.inv(
        np.full(len(rx_lons), tx_lon),
        np.full(len(rx_lats), tx_lat),
        rx_lons,
        rx_lats,
    )
    return np.asarray(dists_m) / 1000.0 <= max_km


# ---------------------------------------------------------------------------
# Windowed DEM loading
# ---------------------------------------------------------------------------

def _load_dem_window(
    dem_path: str,
    center_lat: float,
    center_lon: float,
    radius_km: float,
) -> Tuple[np.ndarray, object]:
    """Load a square DEM patch centred on (center_lat, center_lon).

    Converts radius_km to approximate degree margins (flat-earth approximation
    is accurate to <1% within the bbox latitudes) and reads only that window
    from the raster, keeping per-worker memory ~3–4 MB instead of ~266 MB.

    Args:
        dem_path: Path to the 90 m DEM GeoTIFF.
        center_lat: Latitude of the candidate tower.
        center_lon: Longitude of the candidate tower.
        radius_km: Half-width of the patch to load.

    Returns:
        Tuple of (dem_array, inv_affine_transform) ready for
        ``calculate_memory_p2p_rssi``.
    """
    lat_margin = radius_km / 111.0
    lon_margin = radius_km / (111.0 * math.cos(math.radians(center_lat)))

    west  = center_lon - lon_margin
    east  = center_lon + lon_margin
    south = center_lat - lat_margin
    north = center_lat + lat_margin

    with rasterio.open(dem_path) as src:
        win = from_bounds(west, south, east, north, src.transform)
        # Clamp to valid raster extent — handles candidates near the bbox edge
        col0 = max(0, int(math.floor(win.col_off)))
        row0 = max(0, int(math.floor(win.row_off)))
        col1 = min(src.width,  int(math.ceil(win.col_off + win.width)))
        row1 = min(src.height, int(math.ceil(win.row_off + win.height)))

        # Candidate is fully outside the DEM — return a flat zero patch
        if col1 <= col0 or row1 <= row0:
            return np.zeros((3, 3), dtype=np.float64), ~src.transform

        actual_win    = Window(col0, row0, col1 - col0, row1 - row0)
        dem_slice     = src.read(1, window=actual_win).astype(np.float64)
        win_transform = src.window_transform(actual_win)
        nodata        = src.nodata

    if nodata is not None:
        dem_slice[dem_slice == nodata] = 0.0
    dem_slice[dem_slice < -1000] = 0.0

    return dem_slice, ~win_transform


# ---------------------------------------------------------------------------
# Per-candidate worker — must be top-level for ProcessPoolExecutor pickling
# ---------------------------------------------------------------------------

def _process_candidate(args: tuple) -> Tuple[int, np.ndarray]:
    """Compute RSSI from one candidate tower to all its in-range demand cells.

    Loads only the DEM window required for this candidate's coverage radius,
    then runs ITM p2p for every in-range (candidate, demand) pair sequentially.
    Called in a subprocess by ProcessPoolExecutor.

    Args:
        args: Packed tuple — see ``precompute_coverage`` for fields.

    Returns:
        Tuple of (candidate_index, rssi_row) where rssi_row has shape (n_dem,).
        Out-of-range cells remain np.nan.
    """
    (
        i, tx_lat, tx_lon,
        j_indices, rx_lats_sub, rx_lons_sub,
        n_dem, dem_path, dem_radius_km,
        base_params, tx_power_dbm,
    ) = args

    dem_array, inv_transform = _load_dem_window(dem_path, tx_lat, tx_lon, dem_radius_km)

    rssi_row = np.full(n_dem, np.nan, dtype=np.float32)
    _sink = io.StringIO()
    with contextlib.redirect_stdout(_sink):
        for j, rx_lat, rx_lon in zip(j_indices, rx_lats_sub, rx_lons_sub):
            rssi = calculate_memory_p2p_rssi(
                tx_lon, tx_lat,
                float(rx_lon), float(rx_lat),
                dem_array, inv_transform,
                base_params, tx_power_dbm,
            )
            rssi_row[int(j)] = rssi

    return i, rssi_row


# ---------------------------------------------------------------------------
# Core precompute
# ---------------------------------------------------------------------------

def precompute_coverage(
    candidates: pd.DataFrame,
    demand: pd.DataFrame,
    dem_path: str,
    base_params: dict,
    tx_power_dbm: float = 43.0,
    rssi_threshold: float = -90.0,
    max_coverage_km: float = 30.0,
    max_workers: Optional[int] = None,
    output_path: str = "processed_data/coverage_matrix.npz",
) -> np.ndarray:
    """Build binary coverage matrix [n_candidates × n_demand] via ITM p2p.

    Args:
        candidates: DataFrame with ``h3_index``, ``lat``, ``lon`` columns
            (Res-8 hilltop locations from rural_candidates.csv).
        demand: DataFrame with ``h3_index`` column
            (Res-6 rural demand cells from rural_cells.csv).
        dem_path: Path to DEM GeoTIFF (90 m recommended for full bbox).
        base_params: ITM parameter dict — ``fmhz``, ``hg``, ``ipol``.
        tx_power_dbm: Transmit power in dBm.
        rssi_threshold: Minimum RSSI (dBm) for a cell to count as covered.
        max_coverage_km: Geodesic distance cutoff; farther pairs skip ITM.
        max_workers: Number of processes (default: os.cpu_count()).
        output_path: Where to save the .npz coverage matrix.

    Returns:
        Binary uint8 array of shape (n_candidates, n_demand).
    """
    n_cand = len(candidates)
    n_dem  = len(demand)

    # Demand cell receiver coordinates (Res-6 cell centers)
    demand_latlons = [h3.cell_to_latlng(c) for c in demand["h3_index"]]
    rx_lats = np.array([ll[0] for ll in demand_latlons])
    rx_lons = np.array([ll[1] for ll in demand_latlons])

    # DEM window: coverage radius + 10% so path endpoints always fall inside
    dem_radius_km = max_coverage_km * 1.1

    # Build one task per candidate — pass only the in-range demand subset
    # to minimise pickle traffic across process boundaries
    cand_records = candidates.to_dict("records")
    tasks: List[tuple] = []
    total_itm = 0

    for i, cand in enumerate(cand_records):
        tx_lat, tx_lon = cand["lat"], cand["lon"]
        in_range = _within_range_mask(tx_lat, tx_lon, rx_lats, rx_lons, max_coverage_km)
        j_idx = np.where(in_range)[0].astype(int)
        total_itm += len(j_idx)
        tasks.append((
            i, tx_lat, tx_lon,
            j_idx,
            rx_lats[j_idx],   # only the subset (~90 floats, not ~7000)
            rx_lons[j_idx],
            n_dem, dem_path, dem_radius_km,
            base_params, tx_power_dbm,
        ))

    total_pairs = n_cand * n_dem
    skipped     = total_pairs - total_itm
    workers     = max_workers or os.cpu_count() or 4

    print(f"Coverage matrix : {n_cand} candidates × {n_dem} demand = {total_pairs:,} pairs")
    print(f"Pruned (>{max_coverage_km} km)  : {skipped:,} ({skipped / total_pairs:.1%})")
    print(f"ITM calls needed: {total_itm:,} ({total_itm / total_pairs:.1%})")
    print(f"Workers         : {workers} processes  |  Tasks: {len(tasks)} (one per candidate)")

    rssi_matrix = np.full((n_cand, n_dem), np.nan, dtype=np.float32)

    completed = 0
    next_milestone = 10  # print a log line every 10%

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_candidate, t): t for t in tasks}
        with tqdm(total=len(tasks), desc="ITM precompute", unit="cand") as pbar:
            for future in as_completed(futures):
                i, rssi_row = future.result()
                rssi_matrix[i] = rssi_row
                completed += 1
                pbar.update(1)
                pct = completed / len(tasks) * 100
                if pct >= next_milestone:
                    fd   = pbar.format_dict
                    rate = fd.get("rate") or 0
                    eta  = (len(tasks) - completed) / rate if rate > 0 else 0
                    pbar.write(
                        f"  [{pct:5.1f}%] {completed}/{len(tasks)} candidates done"
                        f"  |  elapsed {fd['elapsed']:.0f}s"
                        f"  |  ETA ~{eta:.0f}s"
                    )
                    next_milestone += 10

    # Threshold — NaN and out-of-range cells both stay 0
    coverage = np.zeros((n_cand, n_dem), dtype=np.uint8)
    coverage[rssi_matrix >= rssi_threshold] = 1

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        coverage=coverage,
        rssi=rssi_matrix,
        candidates=np.array(candidates["h3_index"].values, dtype=str),
        demand=np.array(demand["h3_index"].values, dtype=str),
        rssi_threshold=np.float32(rssi_threshold),
    )

    avg_cov = coverage.mean(axis=1).mean()
    print(f"Saved to {output_path}")
    print(f"Average demand cells covered per candidate: {avg_cov:.1%}")
    return coverage


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Pre-compute ITM coverage matrix for rural MCP GA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml", help="Project YAML config file")

    files = parser.add_argument_group("Files (override config.yaml paths)")
    files.add_argument("--candidates", default=None, help="Candidate tower CSV")
    files.add_argument("--demand",     default=None, help="Rural demand cells CSV")
    files.add_argument("--output",     default=None, help="Output .npz path")
    files.add_argument("--dem",        default=None, help="DEM GeoTIFF (downloaded if absent)")

    bbox = parser.add_argument_group("Bounding Box (used only if DEM must be downloaded)")
    bbox.add_argument("--north", type=float, default=None)
    bbox.add_argument("--south", type=float, default=None)
    bbox.add_argument("--east",  type=float, default=None)
    bbox.add_argument("--west",  type=float, default=None)
    bbox.add_argument("--dem-resolution", type=int, default=None,
                      help="DEM resolution in metres for download")

    rf = parser.add_argument_group("RF / ITM Parameters (override config.yaml rf)")
    rf.add_argument("--freq-mhz",       type=float, default=None)
    rf.add_argument("--tx-height",      type=float, default=None, help="TX antenna height (m)")
    rf.add_argument("--rx-height",      type=float, default=None, help="RX antenna height (m)")
    rf.add_argument("--tx-power",       type=float, default=None, help="TX power (dBm)")
    rf.add_argument("--rssi-threshold", type=float, default=None, help="Coverage threshold (dBm)")
    rf.add_argument("--max-coverage-km", type=float, default=None,
                    help="Distance pruning cutoff (km)")
    rf.add_argument("--workers", type=int, default=None,
                    help="Process count (default: os.cpu_count())")

    return parser.parse_args()


def main() -> None:
    """Entry point for ITM coverage precomputation."""
    args = parse_args()
    cfg  = load_config(args.config)

    def _get(cli_val, *cfg_keys):
        if cli_val is not None:
            return cli_val
        node = cfg
        for key in cfg_keys:
            node = node[key]
        return node

    dem_path  = Path(_get(args.dem, "dem", "path"))
    dem_res   = _get(args.dem_resolution, "dem", "resolution")
    out_dir   = cfg["paths"]["out_dir"]

    candidates_path = args.candidates or f"{out_dir}/rural_candidates.csv"
    demand_path     = args.demand     or f"{out_dir}/rural_cells.csv"
    output_path     = args.output     or f"{out_dir}/coverage_matrix.npz"

    freq_mhz    = _get(args.freq_mhz,       "rf", "freq_mhz")
    tx_height   = _get(args.tx_height,       "rf", "tx_height_m")
    rx_height   = _get(args.rx_height,       "rf", "rx_height_m")
    tx_power    = _get(args.tx_power,        "rf", "tx_power_dbm")
    rssi_thresh = _get(args.rssi_threshold,  "rf", "rssi_threshold_dbm")
    max_cov_km  = _get(args.max_coverage_km, "rf", "max_coverage_km")

    north = _get(args.north, "region", "north")
    south = _get(args.south, "region", "south")
    east  = _get(args.east,  "region", "east")
    west  = _get(args.west,  "region", "west")

    # Download DEM if not present; re-download if it doesn't cover the bbox
    if dem_path.exists():
        with rasterio.open(str(dem_path)) as src:
            b = src.bounds
        if b.left > west or b.right < east or b.bottom > south or b.top < north:
            print(
                f"WARNING: existing DEM covers only "
                f"W={b.left:.3f} S={b.bottom:.3f} E={b.right:.3f} N={b.top:.3f}\n"
                f"  Required bbox: W={west} S={south} E={east} N={north}\n"
                f"  Deleting and re-downloading at {dem_res} m ..."
            )
            dem_path.unlink()

    if not dem_path.exists():
        print(f"Downloading DEM at {dem_res} m for full bbox ...")
        result = download_and_crop(
            west=west, south=south, east=east, north=north,
            output_dir=str(dem_path.parent),
            filename=dem_path.name,
            resolution=dem_res,
        )
        if not result:
            sys.exit("DEM download failed — aborting.")

    candidates = pd.read_csv(candidates_path)
    demand     = pd.read_csv(demand_path)
    print(f"Candidates : {len(candidates)}")
    print(f"Demand cells: {len(demand)}")

    itm_params = {
        "fmhz": freq_mhz,
        "hg":   [tx_height, rx_height],
        "ipol": 1,
    }

    precompute_coverage(
        candidates=candidates,
        demand=demand,
        dem_path=str(dem_path),
        base_params=itm_params,
        tx_power_dbm=tx_power,
        rssi_threshold=rssi_thresh,
        max_coverage_km=max_cov_km,
        max_workers=args.workers,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
