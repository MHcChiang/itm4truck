from typing import Optional

import branca.colormap as cm
import folium
from folium import FeatureGroup, LayerControl
import h3
import pandas as pd


DEFAULT_MIN_POPULATION = 0.1
DEFAULT_COLORS = ["#0000ff", "#00ff00", "#ffff00", "#ff0000"]


def _get_layer_resolution(df: pd.DataFrame, fallback: str) -> str:
    """Return a displayable H3 resolution label from a dataframe."""
    if "res" not in df.columns or df.empty:
        return fallback

    resolutions = df["res"].dropna().unique()
    if len(resolutions) == 0:
        return fallback

    return str(resolutions[0])


def _get_population_colormap(
    df: pd.DataFrame,
    caption: str,
    min_population: float,
    default_max: float,
) -> cm.LinearColormap:
    """Create a population colormap based on active cells in a layer."""
    active_pop = df[df["pop"] > min_population]["pop"]
    max_population = active_pop.quantile(0.95) if not active_pop.empty else default_max
    colormap = cm.LinearColormap(
        DEFAULT_COLORS,
        vmin=min_population,
        vmax=max_population,
    )
    colormap.caption = caption

    return colormap


def add_filtered_hexes_to_map(
    df: pd.DataFrame,
    group: FeatureGroup,
    colormap: cm.LinearColormap,
    min_population: float = DEFAULT_MIN_POPULATION,
) -> None:
    """Add populated H3 cells from a dataframe to a Folium layer group.

    Args:
        df: Dataframe with ``h3_index`` and ``pop`` columns.
        group: Folium feature group to receive the H3 polygons.
        colormap: Colormap used to convert population values to fill colors.
        min_population: Minimum population required for a cell to be drawn.
    """
    populated_df = df[df["pop"] > min_population]

    for _, row in populated_df.iterrows():
        hex_id = row["h3_index"]
        population = row["pop"]
        boundary = h3.cell_to_boundary(hex_id)
        points = [[lat, lon] for lat, lon in boundary]

        folium.Polygon(
            locations=points,
            fill=True,
            fill_color=colormap(population),
            color="black",
            weight=0.3,
            fill_opacity=0.7,
            tooltip=f"H3: {hex_id}<br>Pop: {population:.1f}",
        ).add_to(group)


def visualize_hierarchical_heatmap_filtered(
    df_base: pd.DataFrame,
    df_refined: pd.DataFrame,
    output_file: str = "h3_heatmap_filtered.html",
    min_population: float = DEFAULT_MIN_POPULATION,
    zoom_start: int = 7,
    tiles: str = "CartoDB positron",
    base_layer_name: Optional[str] = None,
    refined_layer_name: Optional[str] = None,
) -> folium.Map:
    """Create and save a filtered hierarchical H3 population heatmap.

    Args:
        df_base: Base H3 layer dataframe returned by ``generate_base_layer``.
        df_refined: Refined H3 layer dataframe returned by ``generate_refined_layer``.
        output_file: HTML file path where the map will be saved.
        min_population: Minimum population required for a cell to be drawn.
        zoom_start: Initial Folium map zoom level.
        tiles: Folium tile layer name.
        base_layer_name: Optional display name for the base layer.
        refined_layer_name: Optional display name for the refined layer.

    Returns:
        Folium map containing base and refined population layers.
    """
    if df_base.empty:
        raise ValueError("df_base must contain at least one H3 cell to center the map.")

    sample_cell = df_base["h3_index"].iloc[0]
    center_lat, center_lng = h3.cell_to_latlng(sample_cell)
    map_obj = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=zoom_start,
        tiles=tiles,
    )

    base_res = _get_layer_resolution(df_base, "base")
    refined_res = _get_layer_resolution(df_refined, "refined")
    base_name = base_layer_name or f"Res {base_res} (Population Areas)"
    refined_name = refined_layer_name or f"Res {refined_res} (Hotspots Layer)"

    cmap_base = _get_population_colormap(
        df_base,
        "Base Layer Population (Pop > 0)",
        min_population,
        default_max=100,
    )
    cmap_refined = _get_population_colormap(
        df_refined,
        "Refined Layer Population (Pop > 0)",
        min_population,
        default_max=10,
    )

    fg_base = FeatureGroup(name=base_name, show=True)
    fg_refined = FeatureGroup(name=refined_name, show=True)

    print("Adding filtered base layer...")
    add_filtered_hexes_to_map(df_base, fg_base, cmap_base, min_population=min_population)

    print("Adding refined layer...")
    add_filtered_hexes_to_map(df_refined, fg_refined, cmap_refined, min_population=min_population)

    fg_base.add_to(map_obj)
    fg_refined.add_to(map_obj)
    map_obj.add_child(cmap_base)

    LayerControl().add_to(map_obj)

    map_obj.save(output_file)
    print(f"Filtered Hierarchical Heatmap saved to {output_file}")

    return map_obj
