import numpy as np
import copy
import concurrent.futures
import os

import pyproj
import rasterio
from scripts.terrain_module import terrain_p2p
from scripts.terrain_module import determine_num_samples
from scripts.p2p import itmlogic_p2p
from tqdm import tqdm

GEOD = pyproj.Geod(ellps="WGS84")


def calculate_single_p2p_rssi(
    tx_coords: tuple,
    rx_coords: tuple,
    dem_path: str,
    base_params: dict,
    tx_power_dbm: float = 43.0,
) -> tuple:
    """
    Calculate RSSI and path loss for one TX-RX pair using terrain_p2p.

    Args:
        tx_coords: Transmitter coordinates as (longitude, latitude).
        rx_coords: Receiver coordinates as (longitude, latitude).
        dem_path: Path to the DEM raster.
        base_params: Base ITM parameter dictionary (for example fmhz, hg, ipol).
        tx_power_dbm: Transmit power in dBm.

    Returns:
        A tuple (rssi, path_loss). Returns (None, None) if the required ITM
        reliability/confidence result is unavailable.
    """
    transmitter = {
        'type': 'Feature',
        'geometry': {'type': 'Point', 'coordinates': tx_coords}
    }
    receiver = {
        'type': 'Feature',
        'geometry': {'type': 'Point', 'coordinates': rx_coords}
    }
    line = {
        'type': 'Feature',
        'geometry': {
            'type': 'LineString',
            'coordinates': [transmitter['geometry']['coordinates'], receiver['geometry']['coordinates']]
        }
    }
    
    # Extract terrain profile and distance
    terrain_profile, distance_km, _ = terrain_p2p(dem_path, line)
    
    # Ensure fresh parameters for each call
    params = copy.deepcopy(base_params)
    params['d'] = distance_km
    
    results = itmlogic_p2p(params, terrain_profile)
    
    path_loss = None
    for res in results:
        if res['reliability_level_%'] == 50 and res['confidence_level_%'] == 50:
            path_loss = res['propagation_loss_dB']
            break
            
    if path_loss is not None:
        rssi = tx_power_dbm - path_loss
        return rssi, path_loss
    return None, None


def memory_terrain_p2p(
    tx_lon: float,
    tx_lat: float,
    rx_lon: float,
    rx_lat: float,
    dem_array: np.ndarray,
    inv_transform,
) -> tuple:
    """
    Extract terrain profile fully in RAM for one TX-RX path.

    Args:
        tx_lon: Transmitter longitude.
        tx_lat: Transmitter latitude.
        rx_lon: Receiver longitude.
        rx_lat: Receiver latitude.
        dem_array: DEM raster as numpy array.
        inv_transform: Inverse raster affine transform.

    Returns:
        A tuple (surface_profile_list, distance_km), where
        surface_profile_list is a list of sampled elevations in meters.
    """
    _, _, distance_m = GEOD.inv(tx_lon, tx_lat, rx_lon, rx_lat)
    distance_km = distance_m / 1000.0

    num_samples = determine_num_samples(distance_m)
    lons = np.linspace(tx_lon, rx_lon, num_samples)
    lats = np.linspace(tx_lat, rx_lat, num_samples)

    cols, rows = inv_transform * (lons, lats)
    cols = np.clip(np.round(cols).astype(int), 0, dem_array.shape[1] - 1)
    rows = np.clip(np.round(rows).astype(int), 0, dem_array.shape[0] - 1)

    surface_profile = dem_array[rows, cols]
    return surface_profile.tolist(), distance_km


def calculate_memory_p2p_rssi(
    tx_lon: float,
    tx_lat: float,
    rx_lon: float,
    rx_lat: float,
    dem_array: np.ndarray,
    inv_transform,
    base_params: dict,
    tx_power_dbm: float = 43.0,
) -> float:
    """
    Calculate one RSSI value using in-memory terrain extraction.

    Args:
        tx_lon: Transmitter longitude.
        tx_lat: Transmitter latitude.
        rx_lon: Receiver longitude.
        rx_lat: Receiver latitude.
        dem_array: DEM raster in memory.
        inv_transform: Inverse affine transform for DEM indexing.
        base_params: Base ITM parameter dictionary.
        tx_power_dbm: Transmit power in dBm.

    Returns:
        RSSI in dBm as a float. Returns -30.0 when TX and RX locations are
        effectively identical. Returns np.nan if the RSSI cannot be computed.
    """
    if abs(tx_lon - rx_lon) < 1e-6 and abs(tx_lat - rx_lat) < 1e-6:
        return -30.0

    try:
        terrain_profile, distance_km = memory_terrain_p2p(
            tx_lon,
            tx_lat,
            rx_lon,
            rx_lat,
            dem_array,
            inv_transform,
        )
        current_params = copy.deepcopy(base_params)
        current_params["d"] = distance_km

        results = itmlogic_p2p(current_params, terrain_profile)
        for result in results:
            if (
                result["reliability_level_%"] == 50
                and result["confidence_level_%"] == 50
            ):
                return tx_power_dbm - result["propagation_loss_dB"]
    except Exception:
        return np.nan

    return np.nan


def process_grid_point_memory(args: tuple) -> tuple:
    """
    Worker wrapper for threaded RSSI calculation.

    Args:
        args: Packed tuple containing grid indices, TX/RX coordinates, DEM data,
            inverse transform, ITM parameters, and transmit power.

    Returns:
        A tuple (i, j, rssi) where i and j are grid indices.
    """
    (
        i,
        j,
        tx_lon,
        tx_lat,
        rx_lon,
        rx_lat,
        dem_array,
        inv_transform,
        base_params,
        tx_power_dbm,
    ) = args
    rssi = calculate_memory_p2p_rssi(
        tx_lon,
        tx_lat,
        rx_lon,
        rx_lat,
        dem_array,
        inv_transform,
        base_params,
        tx_power_dbm,
    )
    return i, j, rssi


def estimate_grid_signal(
    dem_path: str,
    center_coords: tuple,
    grid_size_n: int,
    step_deg: float,
    base_params: dict,
    tx_power_dbm: float = 43.0,
    target_area: dict = None,
) -> tuple:
    """
    Estimate RSSI over an n-by-n grid around a central base station.

    Args:
        dem_path: Path to the DEM raster used for terrain sampling.
        center_coords: Base station coordinates as (longitude, latitude).
        grid_size_n: Grid size per dimension (n creates an n-by-n grid).
        step_deg: Angular step between adjacent grid points in degrees.
        base_params: Base ITM parameter dictionary.
        tx_power_dbm: Transmit power in dBm.
        target_area: Optional dict with keys west, south, east, north. When
            provided, the grid is generated to fully span these bounds.

    Returns:
        A tuple (lon_grid, lat_grid, rssi_matrix):
        - lon_grid: 1D longitude coordinates.
        - lat_grid: 1D latitude coordinates.
        - rssi_matrix: 2D RSSI matrix in dBm. Uncomputable cells are np.nan.
    """
    tx_lon, tx_lat = center_coords

    if grid_size_n < 2:
        raise ValueError("grid_size_n must be at least 2.")

    # Prefer target-area bounds when provided so the grid fully fills area.
    if target_area is not None:
        lon_grid = np.linspace(
            float(target_area["west"]),
            float(target_area["east"]),
            grid_size_n,
        )
        lat_grid = np.linspace(
            float(target_area["south"]),
            float(target_area["north"]),
            grid_size_n,
        )
    else:
        half_grid = grid_size_n // 2
        lon_grid = np.linspace(
            tx_lon - half_grid * step_deg,
            tx_lon + half_grid * step_deg,
            grid_size_n,
        )
        lat_grid = np.linspace(
            tx_lat - half_grid * step_deg,
            tx_lat + half_grid * step_deg,
            grid_size_n,
        )
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    rssi_matrix = np.full((grid_size_n, grid_size_n), np.nan)

    with rasterio.open(dem_path) as src:
        dem_array = src.read(1).astype(float)
        inv_transform = ~src.transform
        nodata = src.nodata

    if nodata is not None:
        dem_array[dem_array == nodata] = 0.0
    dem_array[dem_array < -1000] = 0.0

    tasks = []
    for i in range(grid_size_n):
        for j in range(grid_size_n):
            tasks.append(
                (
                    i,
                    j,
                    tx_lon,
                    tx_lat,
                    float(lon_mesh[i, j]),
                    float(lat_mesh[i, j]),
                    dem_array,
                    inv_transform,
                    base_params,
                    tx_power_dbm,
                )
            )

    print(f"Calculating ITM logic over a {grid_size_n}x{grid_size_n} grid...")
    max_workers = os.cpu_count() * 2 if os.cpu_count() else 4
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_grid_point_memory, task): task for task in tasks
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(tasks),
            desc="Calculating RSSI",
        ):
            i, j, rssi = future.result()
            rssi_matrix[i, j] = rssi

    return lon_grid, lat_grid, rssi_matrix


def build_hex_grid(target_area: dict, resolution: int = 9) -> list:
    """
    Build a list of hexagonal cell IDs covering the specified target area.

    Args:
        target_area: Dictionary with keys 'west', 'south', 'east', 'north'.
        resolution: H3 resolution level. 9=0.1km^2

    Returns:
        A list of H3 cell string IDs.
    """
    import h3

    exterior = [
        (float(target_area["north"]), float(target_area["west"])),
        (float(target_area["north"]), float(target_area["east"])),
        (float(target_area["south"]), float(target_area["east"])),
        (float(target_area["south"]), float(target_area["west"])),
        (float(target_area["north"]), float(target_area["west"])),
    ]

    try:
        # h3 v4.x
        polygon = h3.LatLngPoly(exterior)
        cells = h3.polygon_to_cells(polygon, resolution)
    except AttributeError:
        # h3 v3.x fallback
        geojson = {
            "type": "Polygon",
            "coordinates": [[[lng, lat] for lat, lng in exterior]],
        }
        cells = h3.polyfill(geojson, resolution, geo_json_conformant=True)
    
    return list(cells)


def convert_dem_to_hex(dem_array: np.ndarray, inv_transform, hex_cells: list) -> dict:
    """
    Convert square DEM to hex DEM by sampling elevation at each hex cell center.

    Args:
        dem_array: DEM raster as numpy array.
        inv_transform: Inverse raster affine transform.
        hex_cells: List of H3 cell IDs.

    Returns:
        A dictionary mapping hex_id to elevation.
    """
    import h3

    hex_dem = {}
    for cell in hex_cells:
        try:
            lat, lon = h3.cell_to_latlng(cell)
        except AttributeError:
            lat, lon = h3.h3_to_geo(cell)

        cols, rows = inv_transform * (lon, lat)
        col = int(np.clip(np.round(cols), 0, dem_array.shape[1] - 1))
        row = int(np.clip(np.round(rows), 0, dem_array.shape[0] - 1))

        hex_dem[cell] = float(dem_array[row, col])

    return hex_dem


def calculate_hex_p2p_rssi(
    tx_lon: float,
    tx_lat: float,
    rx_cell: str,
    hex_dem: dict,
    base_params: dict,
    tx_power_dbm: float = 43.0,
) -> float:
    import h3

    try:
        lat, lon = h3.cell_to_latlng(rx_cell)
    except AttributeError:
        lat, lon = h3.h3_to_geo(rx_cell)

    if abs(tx_lon - lon) < 1e-6 and abs(tx_lat - lat) < 1e-6:
        return -30.0

    _, _, distance_m = GEOD.inv(tx_lon, tx_lat, lon, lat)
    distance_km = distance_m / 1000.0

    try:
        res = h3.get_resolution(rx_cell) if hasattr(h3, 'get_resolution') else h3.h3_get_resolution(rx_cell)
        
        try:
            tx_cell = h3.latlng_to_cell(tx_lat, tx_lon, res)
        except AttributeError:
            tx_cell = h3.geo_to_h3(tx_lat, tx_lon, res)

        try:
            path_cells = h3.grid_path_cells(tx_cell, rx_cell)
        except AttributeError:
            path_cells = h3.h3_line(tx_cell, rx_cell)

        surface_profile = [hex_dem.get(c, 0.0) for c in path_cells]

        current_params = copy.deepcopy(base_params)
        current_params["d"] = distance_km

        results = itmlogic_p2p(current_params, surface_profile)
        for result in results:
            if (
                result["reliability_level_%"] == 50
                and result["confidence_level_%"] == 50
            ):
                return tx_power_dbm - result["propagation_loss_dB"]
    except Exception:
        return np.nan

    return np.nan


def process_hex_grid_point(args: tuple) -> tuple:
    (
        rx_cell,
        tx_lon,
        tx_lat,
        hex_dem,
        base_params,
        tx_power_dbm,
    ) = args

    rssi = calculate_hex_p2p_rssi(
        tx_lon, tx_lat, rx_cell, hex_dem, base_params, tx_power_dbm
    )
    return rx_cell, rssi


def estimate_hex_grid_signal(
    dem_path: str,
    center_coords: tuple,
    target_area: dict,
    resolution: int,
    base_params: dict,
    tx_power_dbm: float = 43.0,
) -> dict:
    """
    Estimate RSSI over an H3 hexagonal grid.

    Args:
        dem_path: Path to the DEM raster used for terrain sampling.
        center_coords: Base station coordinates as (longitude, latitude).
        target_area: Dictionary with keys 'west', 'south', 'east', 'north'.
        resolution: H3 resolution level.
        base_params: Base ITM parameter dictionary.
        tx_power_dbm: Transmit power in dBm.

    Returns:
        A dictionary mapping H3 cell IDs to RSSI values.
    """
    import h3

    tx_lon, tx_lat = center_coords

    print(f"Building hex grid at resolution {resolution}...")
    hex_cells = build_hex_grid(target_area, resolution)

    with rasterio.open(dem_path) as src:
        dem_array = src.read(1).astype(float)
        inv_transform = ~src.transform
        nodata = src.nodata

    if nodata is not None:
        dem_array[dem_array == nodata] = 0.0
    dem_array[dem_array < -1000] = 0.0

    print("Converting square DEM to hex DEM...")
    hex_dem = convert_dem_to_hex(dem_array, inv_transform, hex_cells)

    try:
        tx_cell = h3.latlng_to_cell(tx_lat, tx_lon, resolution)
    except AttributeError:
        tx_cell = h3.geo_to_h3(tx_lat, tx_lon, resolution)

    if tx_cell not in hex_dem:
        cols, rows = inv_transform * (tx_lon, tx_lat)
        col = int(np.clip(np.round(cols), 0, dem_array.shape[1] - 1))
        row = int(np.clip(np.round(rows), 0, dem_array.shape[0] - 1))
        hex_dem[tx_cell] = float(dem_array[row, col])

    tasks = [
        (
            cell,
            tx_lon,
            tx_lat,
            hex_dem,
            base_params,
            tx_power_dbm,
        )
        for cell in hex_cells
    ]

    hex_rssi = {}
    print(f"Calculating ITM logic over {len(hex_cells)} hex cells...")
    max_workers = os.cpu_count() * 2 if os.cpu_count() else 4
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_hex_grid_point, task): task for task in tasks
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(tasks),
            desc="Calculating Hex RSSI",
        ):
            rx_cell, rssi = future.result()
            hex_rssi[rx_cell] = rssi

    return hex_rssi
