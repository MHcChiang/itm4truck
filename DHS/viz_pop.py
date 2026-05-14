from typing import Dict, List, Optional

import branca.colormap as cm
import folium
from folium import FeatureGroup, LayerControl
import h3
import pandas as pd

_CLUSTER_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
    "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
]

DEFAULT_MIN_POPULATION = 0.1
DEFAULT_COLORS = ["#0000ff", "#00ff00", "#ffff00", "#ff0000"]


# --- Private helpers ---

def _hex_boundary_points(hex_id: str) -> List[List[float]]:
    """Return Folium-compatible [[lat, lon], ...] boundary for an H3 cell."""
    return [[lat, lon] for lat, lon in h3.cell_to_boundary(hex_id)]


def _build_rank_map(df_clusters: pd.DataFrame) -> Dict[int, int]:
    """Map cluster_id → population rank (1 = largest cluster).

    df_clusters must be sorted by descending total_pop (as returned by
    classify_zones), so iterrows() rank matches population order.
    """
    return {int(row["cluster_id"]): rank + 1 for rank, row in df_clusters.iterrows()}


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
    colormap = cm.LinearColormap(DEFAULT_COLORS, vmin=min_population, vmax=max_population)
    colormap.caption = caption
    return colormap


# --- Public functions ---

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
    for row in df[df["pop"] > min_population][["h3_index", "pop"]].to_dict("records"):
        folium.Polygon(
            locations=_hex_boundary_points(row["h3_index"]),
            fill=True,
            fill_color=colormap(row["pop"]),
            color="black",
            weight=0.3,
            fill_opacity=0.7,
            tooltip=f"H3: {row['h3_index']}<br>Pop: {row['pop']:.1f}",
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

    center_lat, center_lng = h3.cell_to_latlng(df_base["h3_index"].iloc[0])
    map_obj = folium.Map(location=[center_lat, center_lng], zoom_start=zoom_start, tiles=tiles)

    base_res = _get_layer_resolution(df_base, "base")
    refined_res = _get_layer_resolution(df_refined, "refined")
    base_name = base_layer_name or f"Res {base_res} (Population Areas)"
    refined_name = refined_layer_name or f"Res {refined_res} (Hotspots Layer)"

    cmap_base = _get_population_colormap(df_base, "Base Layer Population (Pop > 0)",
                                         min_population, default_max=100)
    cmap_refined = _get_population_colormap(df_refined, "Refined Layer Population (Pop > 0)",
                                            min_population, default_max=10)

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


def add_zone_layers(
    map_obj: folium.Map,
    df_urban: pd.DataFrame,
    df_rural: pd.DataFrame,
    df_clusters: pd.DataFrame,
    output_file: str = "h3_zones.html",
    show_rural: bool = True,
) -> folium.Map:
    """Overlay urban cluster and rural zone layers onto an existing Folium map.

    Each urban cluster is drawn as a distinct colored layer that can be toggled
    independently in the layer control. Rural cells share a single gray layer.

    Args:
        map_obj: Existing Folium map to add layers to.
        df_urban: Urban cells dataframe with ``h3_index``, ``pop``, and
            ``cluster_id`` columns (output of ``classify_zones``).
        df_rural: Rural cells dataframe with ``h3_index`` and ``pop`` columns.
        df_clusters: Cluster summary from ``classify_zones`` (sorted by
            descending ``total_pop`` so index 0 is the largest cluster).
        output_file: Path to save the updated map HTML.
        show_rural: Whether to show the rural layer by default.

    Returns:
        The modified Folium map.
    """
    rank_map = _build_rank_map(df_clusters)

    for row in df_clusters.to_dict("records"):
        cid = int(row["cluster_id"])
        rank = rank_map[cid]
        color = _CLUSTER_COLORS[rank % len(_CLUSTER_COLORS)]
        pop_k = row["total_pop"] / 1_000
        fg = FeatureGroup(name=f"Urban Cluster #{rank} ({pop_k:,.0f}k pop)", show=True)

        for cell in df_urban[df_urban["cluster_id"] == cid][["h3_index", "pop"]].to_dict("records"):
            folium.Polygon(
                locations=_hex_boundary_points(cell["h3_index"]),
                fill=True, fill_color=color, color=color, weight=0.5, fill_opacity=0.5,
                tooltip=f"Cluster #{rank} | Cell: {cell['h3_index']}<br>Pop: {cell['pop']:.1f}",
            ).add_to(fg)
        fg.add_to(map_obj)

    fg_rural = FeatureGroup(name="Rural Zone", show=show_rural)
    for cell in df_rural[df_rural["pop"] > DEFAULT_MIN_POPULATION][["h3_index", "pop"]].to_dict("records"):
        folium.Polygon(
            locations=_hex_boundary_points(cell["h3_index"]),
            fill=True, fill_color="#aaaaaa", color="#888888", weight=0.3, fill_opacity=0.3,
            tooltip=f"Rural | Pop: {cell['pop']:.1f}",
        ).add_to(fg_rural)
    fg_rural.add_to(map_obj)

    LayerControl(collapsed=False).add_to(map_obj)
    map_obj.save(output_file)
    print(f"Zone map saved to {output_file}")
    return map_obj


def visualize_town_zones(
    df_urban: pd.DataFrame,
    df_town_grid: pd.DataFrame,
    df_clusters: pd.DataFrame,
    output_file: str = "h3_zones.html",
    zoom_start: int = 7,
    tiles: str = "CartoDB positron",
) -> folium.Map:
    """Create a standalone zone map with town-area and town-grid layers.

    Produces a fresh Folium map with two toggleable layers:

    - **Town Areas**: base-res hexes colored by cluster identity.
    - **Town Grid**: fine-res child hexes colored by population intensity.

    Args:
        df_urban: Town cells dataframe with ``h3_index``, ``pop``, and
            ``cluster_id`` columns (output of ``classify_zones``).
        df_town_grid: Fine-res grid dataframe with ``h3_index`` and ``pop``
            columns (output of ``generate_town_grid``).
        df_clusters: Cluster summary sorted by descending ``total_pop``
            (output of ``classify_zones``).
        output_file: Path to write the HTML map.
        zoom_start: Initial map zoom level.
        tiles: Folium tile layer name.

    Returns:
        The saved Folium map object.
    """
    if df_urban.empty:
        raise ValueError("df_urban is empty — no town cells to map.")

    center_lat, center_lon = h3.cell_to_latlng(df_urban["h3_index"].iloc[0])
    map_obj = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles=tiles)

    rank_map = _build_rank_map(df_clusters)

    fg_towns = FeatureGroup(name="Town Areas (Res 6)", show=True)
    for cell in df_urban[["h3_index", "pop", "cluster_id"]].to_dict("records"):
        rank = rank_map[int(cell["cluster_id"])]
        color = _CLUSTER_COLORS[rank % len(_CLUSTER_COLORS)]
        folium.Polygon(
            locations=_hex_boundary_points(cell["h3_index"]),
            fill=True, fill_color=color, color=color, weight=0.8, fill_opacity=0.4,
            tooltip=f"Cluster #{rank} | Pop: {cell['pop']:.1f}",
        ).add_to(fg_towns)
    fg_towns.add_to(map_obj)

    fg_grid = FeatureGroup(name="Town Grid (High Res)", show=True)
    if not df_town_grid.empty:
        cmap_grid = _get_population_colormap(df_town_grid, "Town Grid Population",
                                             min_population=0, default_max=10)
        for cell in df_town_grid[["h3_index", "pop"]].to_dict("records"):
            folium.Polygon(
                locations=_hex_boundary_points(cell["h3_index"]),
                fill=True, fill_color=cmap_grid(max(cell["pop"], 0)),
                color="#333333", weight=0.2, fill_opacity=0.75,
                tooltip=f"Pop: {cell['pop']:.2f}",
            ).add_to(fg_grid)
        map_obj.add_child(cmap_grid)
    fg_grid.add_to(map_obj)

    LayerControl(collapsed=False).add_to(map_obj)
    map_obj.save(output_file)
    print(f"Zone map saved to {output_file}")
    return map_obj


def visualize_pop_distribution(
    df_base: pd.DataFrame,
    output_file: str = "fig/pop_dist.html",
    zoom_start: int = 6,
    tiles: str = "CartoDB positron",
) -> folium.Map:
    """Create a population heatmap showing all base-layer cells colored by population.

    Every cell in the base layer is drawn regardless of population value. Color
    encodes population intensity from the colormap minimum (blue) to the 95th
    percentile (red), so sparse rural cells and dense urban cells are both visible.

    Args:
        df_base: Base-layer dataframe with ``h3_index`` and ``pop`` columns.
        output_file: Path to write the HTML map.
        zoom_start: Initial map zoom level.
        tiles: Folium tile layer name.

    Returns:
        The saved Folium map object.
    """
    if df_base.empty:
        raise ValueError("df_base is empty — nothing to map.")

    center_lat, center_lon = h3.cell_to_latlng(df_base["h3_index"].iloc[0])
    map_obj = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles=tiles)

    cmap = _get_population_colormap(
        df_base, "Population (Base Layer)",
        min_population=0,
        default_max=float(df_base["pop"].quantile(0.95)) or 1.0,
    )

    fg = FeatureGroup(name="Population Distribution", show=True)
    for cell in df_base[["h3_index", "pop"]].to_dict("records"):
        pop = max(cell["pop"], 0)
        folium.Polygon(
            locations=_hex_boundary_points(cell["h3_index"]),
            fill=True, fill_color=cmap(pop), color="#555555", weight=0.3, fill_opacity=0.7,
            tooltip=f"Cell: {cell['h3_index']}<br>Pop: {pop:.1f}",
        ).add_to(fg)
    fg.add_to(map_obj)
    map_obj.add_child(cmap)

    LayerControl().add_to(map_obj)
    map_obj.save(output_file)
    print(f"Population distribution map saved to {output_file}")
    return map_obj
