import os
import py3dep
import rioxarray
from shapely.geometry import box

def download_and_crop(
    west: float,
    south: float,
    east: float,
    north: float,
    output_dir: str = "data/dem_data",
    filename: str = "target_area_dem.tif",
    resolution: int = 30,
) -> str:
    """Download DEM data from USGS 3DEP and save as a GeoTIFF.

    Args:
        west: West longitude bound.
        south: South latitude bound.
        east: East longitude bound.
        north: North latitude bound.
        output_dir: Directory to save the GeoTIFF.
        filename: Output filename.
        resolution: DEM resolution in metres (default 30 m).
            Coarser values (e.g. 90) download faster for large areas;
            match to the finest H3 child resolution you plan to sample.

    Returns:
        Path to the saved GeoTIFF, or empty string on failure.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, filename)
    print(f"Downloading DEM ({resolution} m) for W:{west} S:{south} E:{east} N:{north}")

    try:
        geom = box(west, south, east, north)
        dem = py3dep.get_dem(geom, resolution=resolution, crs="EPSG:4326")
        dem = dem.rio.reproject("EPSG:4326")
        dem.rio.to_raster(output_path)
        print(f"DEM saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"DEM download failed: {e}")
        return ""