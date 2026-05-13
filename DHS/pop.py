from typing import Dict, List

import h3
import numpy as np
import pandas as pd
import rasterio


DEFAULT_BASE_POP_SCALE = 3600.0
DEFAULT_MIN_ACTIVE_POP = 0.1


def get_h3_cells(bbox: Dict[str, float], resolution: int) -> List[str]:
    """Generate H3 cells covering a latitude/longitude bounding box.

    Args:
        bbox: Bounding box with ``north``, ``south``, ``east``, and ``west`` keys.
        resolution: H3 resolution to generate.

    Returns:
        H3 cell indexes covering the bounding box.
    """
    exterior = [
        (float(bbox["north"]), float(bbox["west"])),
        (float(bbox["north"]), float(bbox["east"])),
        (float(bbox["south"]), float(bbox["east"])),
        (float(bbox["south"]), float(bbox["west"])),
        (float(bbox["north"]), float(bbox["west"])),
    ]
    polygon = h3.LatLngPoly(exterior)

    return list(h3.polygon_to_cells(polygon, resolution))


def generate_base_layer(
    tif_path: str,
    bbox_dict: Dict[str, float],
    base_res: int = 6,
    pop_scale: float = DEFAULT_BASE_POP_SCALE,
) -> pd.DataFrame:
    """Generate the base H3 population layer for a bounding box.

    Args:
        tif_path: Path to the population GeoTIFF.
        bbox_dict: Bounding box with ``north``, ``south``, ``east``, and ``west`` keys.
        base_res: H3 resolution for the base layer.
        pop_scale: Multiplier used to approximate population over the larger base hex.

    Returns:
        Dataframe with ``h3_index``, ``pop``, ``res``, and ``raw_pixel_pop`` columns.
    """
    print(f"--- Generating Base Layer (Res {base_res}) ---")
    cells = get_h3_cells(bbox_dict, base_res)

    data = []
    with rasterio.open(tif_path) as src:
        coords = [h3.cell_to_latlng(cell)[::-1] for cell in cells]
        samples = np.array([value[0] for value in src.sample(coords)])
        samples = np.where(samples < 0, 0, samples)

        for index, pop_value in enumerate(samples):
            data.append(
                {
                    "h3_index": cells[index],
                    "pop": pop_value * pop_scale,
                    "res": base_res,
                    "raw_pixel_pop": pop_value,
                }
            )

    df_base = pd.DataFrame(data)
    print(f"Base Layer complete: {len(df_base)} hexes.")

    return df_base


def generate_refined_layer(
    tif_path: str,
    df_base: pd.DataFrame,
    percentile: float = 20,
    refined_res: int = 8,
    min_active_pop: float = DEFAULT_MIN_ACTIVE_POP,
) -> pd.DataFrame:
    """Generate a refined H3 population layer within active base-layer hot spots.

    Args:
        tif_path: Path to the population GeoTIFF.
        df_base: Base layer dataframe returned by ``generate_base_layer``.
        percentile: Percentile of active raw pixel population used as the hot spot threshold.
        refined_res: H3 resolution for refined child cells.
        min_active_pop: Minimum raw population used to identify active base cells.

    Returns:
        Dataframe with ``h3_index``, ``pop``, ``res``, and ``parent`` columns.
    """
    print(f"--- Generating Refined Layer (Res {refined_res}) ---")

    active_pops = df_base[df_base["raw_pixel_pop"] > min_active_pop]["raw_pixel_pop"]
    if len(active_pops) > 0:
        threshold = np.percentile(active_pops, percentile)
    else:
        threshold = min_active_pop

    print(f"Threshold for refinement ({percentile}th percentile): {threshold:.4f}")

    hot_spots = df_base[df_base["raw_pixel_pop"] >= threshold]["h3_index"].tolist()
    print(f"Found {len(hot_spots)} hot spots to refine.")

    refined_data = []
    with rasterio.open(tif_path) as src:
        for parent_hex in hot_spots:
            children = list(h3.cell_to_children(parent_hex, refined_res))
            child_coords = [h3.cell_to_latlng(child)[::-1] for child in children]
            child_samples = list(src.sample(child_coords))

            for index, child_hex in enumerate(children):
                child_pop = child_samples[index][0]
                if child_pop > 0:
                    refined_data.append(
                        {
                            "h3_index": child_hex,
                            "pop": child_pop,
                            "res": refined_res,
                            "parent": parent_hex,
                        }
                    )

    df_refined = pd.DataFrame(refined_data)
    print(f"Refined Layer complete: {len(df_refined)} hexes.")

    return df_refined
