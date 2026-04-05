import os
import py3dep
import rioxarray
from shapely.geometry import box

def download_and_crop(west: float, south: float, east: float, north: float, output_dir: str = "dem_data", filename: str = "target_area_dem.tif") -> str:
    """
    Downloads DEM data from USGS using the py3dep library and saves it as a GeoTIFF.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, filename)
    print(f"Starting download for bounds: W:{west}, S:{south}, E:{east}, N:{north}")
    
    try:
        # Create a bounding box for the target area
        geom = box(west, south, east, north)
        
        # Fetch the DEM at 30m resolution using EPSG:4326 (WGS84)
        dem = py3dep.get_dem(geom, resolution=30, crs="EPSG:4326")
        
        # Save the downloaded data as a GeoTIFF file (required for itmlogic)
        dem = dem.rio.reproject("EPSG:4326")
        dem.rio.to_raster(output_path)
        
        print(f"Success: File saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error during DEM download: {e}")
        return ""