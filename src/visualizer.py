import io
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend — safe for Colab and headless envs
import matplotlib.pyplot as plt
import rasterio

def plot_signal_distribution(dem_path: str, tx_coords: tuple, lon_grid: np.ndarray, lat_grid: np.ndarray, rssi_matrix: np.ndarray, target_area: dict):
    """
    Overlays the calculated RSSI heatmap onto the downloaded topographic DEM context.
    """
    print("Generating advanced GIS visualization...")
    plt.figure(figsize=(12, 10))
    
    # Load DEM data for background
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1).astype(float)
        dem_extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        dem_data[dem_data < -1000] = np.nan
        
    # Plot DEM as Grayscale background
    plt.imshow(dem_data, extent=dem_extent, cmap='Greys_r', origin='upper', alpha=0.6)
    
    # Plot Topographic Contours
    dem_x = np.linspace(dem_extent[0], dem_extent[1], dem_data.shape[1])
    dem_y = np.linspace(dem_extent[3], dem_extent[2], dem_data.shape[0])
    dem_X, dem_Y = np.meshgrid(dem_x, dem_y)
    
    contour_dem = plt.contour(dem_X, dem_Y, dem_data, levels=10, colors='black', alpha=0.4, linewidths=0.8)
    plt.clabel(contour_dem, inline=True, fontsize=9, fmt='%1.0f m')
    
    # Overlay the RSSI Heatmap
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(alpha=0.0)
    masked_rssi = np.ma.masked_invalid(rssi_matrix)
    mesh = plt.pcolormesh(
        lon_grid,
        lat_grid,
        masked_rssi,
        cmap=cmap,
        shading='auto',
        alpha=0.5,
        vmin=-120,
        vmax=-50,
    )
    
    # Plot Base Station Tower
    plt.scatter(tx_coords[0], tx_coords[1], color='blue', marker='^', s=200, edgecolors='white', linewidth=2, label='Base Station')
    
    plt.colorbar(mesh, label='RSSI (dBm)')
    plt.title('Integrated Signal & Terrain Coverage (700MHz)', fontsize=14, fontweight='bold')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    # set_aspect with 'datalim' adjusts the data range instead of fighting xlim/ylim
    plt.gca().set_aspect('equal', adjustable='datalim')

    if target_area:
        plt.xlim(target_area["west"], target_area["east"])
        plt.ylim(target_area["south"], target_area["north"])

    plt.legend(loc='upper right')
    plt.tight_layout()

    # Display inline (Colab/Jupyter) or save to file (headless)
    try:
        from IPython.display import display, Image as IPImage
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        display(IPImage(buf.read()))
    except ImportError:
        plt.savefig('signal_coverage.png', dpi=150, bbox_inches='tight')
        print("Plot saved to signal_coverage.png")
    finally:
        plt.close()


def plot_hex_signal_map(dem_path: str, tx_coords: tuple, hex_rssi: dict, target_area: dict):
    """
    Overlays the calculated Hex RSSI heatmap onto the downloaded topographic DEM context.
    """
    import h3
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    print("Generating advanced GIS visualization (Hex Grid)...")
    plt.figure(figsize=(12, 10))
    
    # Load DEM data for background
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1).astype(float)
        dem_extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        dem_data[dem_data < -1000] = np.nan
        
    # Plot DEM as Grayscale background
    plt.imshow(dem_data, extent=dem_extent, cmap='Greys_r', origin='upper', alpha=0.6)
    
    # Plot Topographic Contours
    dem_x = np.linspace(dem_extent[0], dem_extent[1], dem_data.shape[1])
    dem_y = np.linspace(dem_extent[3], dem_extent[2], dem_data.shape[0])
    dem_X, dem_Y = np.meshgrid(dem_x, dem_y)
    
    contour_dem = plt.contour(dem_X, dem_Y, dem_data, levels=10, colors='black', alpha=0.4, linewidths=0.8)
    plt.clabel(contour_dem, inline=True, fontsize=9, fmt='%1.0f m')

    # Overlay the RSSI Hexes
    patches = []
    colors = []
    for cell, rssi in hex_rssi.items():
        if np.isnan(rssi):
            continue
        boundary = h3.cell_to_boundary(cell)
        
        # h3 returns (lat, lng), we need (lng, lat) for matplotlib
        polygon = Polygon([(lng, lat) for lat, lng in boundary], closed=True)
        patches.append(polygon)
        colors.append(rssi)

    if patches:
        cmap = plt.cm.RdYlGn.copy()
        cmap.set_bad(alpha=0.0)
        p = PatchCollection(patches, cmap=cmap, alpha=0.5, edgecolor='none')
        p.set_array(np.array(colors))
        p.set_clim(vmin=-120, vmax=-50)
        plt.gca().add_collection(p)
        plt.colorbar(p, label='RSSI (dBm)')

    # Plot Base Station Tower
    plt.scatter(tx_coords[0], tx_coords[1], color='blue', marker='^', s=200, edgecolors='white', linewidth=2, label='Base Station')
    
    plt.title('Integrated Signal & Terrain Coverage (Hex Grid - 700MHz)', fontsize=14, fontweight='bold')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.gca().set_aspect('equal', adjustable='datalim')

    if target_area:
        plt.xlim(target_area["west"], target_area["east"])
        plt.ylim(target_area["south"], target_area["north"])

    plt.legend(loc='upper right')
    plt.tight_layout()

    try:
        from IPython.display import display, Image as IPImage
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        display(IPImage(buf.read()))
    except ImportError:
        plt.savefig('hex_signal_coverage.png', dpi=150, bbox_inches='tight')
        print("Plot saved to hex_signal_coverage.png")
    finally:
        plt.close()