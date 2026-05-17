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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Map lifecycle
# ---------------------------------------------------------------------------

def create_zone_map(
    df_h3: pd.DataFrame,
    zoom_start: int = 7,
    tiles: str = "CartoDB positron",
) -> folium.Map:
    """Create a new Folium map centered on the first cell in df_h3.

    Args:
        df_h3: Any dataframe with an ``h3_index`` column — used only for
            centering the map.
        zoom_start: Initial zoom level.
        tiles: Folium tile layer name.

    Returns:
        Empty Folium map ready to receive layers.
    """
    center_lat, center_lon = h3.cell_to_latlng(df_h3["h3_index"].iloc[0])
    return folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles=tiles)


def finalize_map(map_obj: folium.Map, output_file: str) -> folium.Map:
    """Add a LayerControl and save the map to HTML.

    Call this once after all layers have been added. Adding more layers
    after this call will not update the LayerControl.

    Args:
        map_obj: Folium map with all desired layers already added.
        output_file: Path to write the HTML file.

    Returns:
        The saved map object.
    """
    LayerControl(collapsed=False).add_to(map_obj)
    map_obj.save(output_file)
    print(f"Map saved to {output_file}")
    return map_obj


# ---------------------------------------------------------------------------
# Composable layer adders
# ---------------------------------------------------------------------------

def add_town_areas_layer(
    map_obj: folium.Map,
    df_urban: pd.DataFrame,
    df_clusters: pd.DataFrame,
    show: bool = True,
) -> folium.Map:
    """Add base-res town hexes colored by cluster identity.

    Args:
        map_obj: Target Folium map.
        df_urban: Town cells with ``h3_index``, ``pop``, ``cluster_id``.
        df_clusters: Cluster summary sorted by descending ``total_pop``.
        show: Whether to show this layer by default.

    Returns:
        The same map_obj for chaining.
    """
    rank_map = _build_rank_map(df_clusters)
    fg = FeatureGroup(name="Town Areas", show=show)
    for cell in df_urban[["h3_index", "pop", "cluster_id"]].to_dict("records"):
        rank = rank_map[int(cell["cluster_id"])]
        color = _CLUSTER_COLORS[rank % len(_CLUSTER_COLORS)]
        folium.Polygon(
            locations=_hex_boundary_points(cell["h3_index"]),
            fill=True, fill_color=color, color=color, weight=0.8, fill_opacity=0.4,
            tooltip=f"Cluster #{rank} | Pop: {cell['pop']:.1f}",
        ).add_to(fg)
    fg.add_to(map_obj)
    return map_obj


def add_town_grid_layer(
    map_obj: folium.Map,
    df_town_grid: pd.DataFrame,
    show: bool = True,
) -> folium.Map:
    """Add fine-res town hexes colored by population intensity.

    Args:
        map_obj: Target Folium map.
        df_town_grid: Fine-res grid with ``h3_index`` and ``pop`` columns.
        show: Whether to show this layer by default.

    Returns:
        The same map_obj for chaining.
    """
    if df_town_grid.empty:
        return map_obj
    cmap = _get_population_colormap(df_town_grid, "Town Grid Population",
                                    min_population=0, default_max=10)
    fg = FeatureGroup(name="Town Grid (High Res)", show=show)
    for cell in df_town_grid[["h3_index", "pop"]].to_dict("records"):
        folium.Polygon(
            locations=_hex_boundary_points(cell["h3_index"]),
            fill=True, fill_color=cmap(max(cell["pop"], 0)),
            color="#333333", weight=0.2, fill_opacity=0.75,
            tooltip=f"Pop: {cell['pop']:.2f}",
        ).add_to(fg)
    fg.add_to(map_obj)
    map_obj.add_child(cmap)
    return map_obj


def add_anomaly_layer(
    map_obj: folium.Map,
    df_anomaly: pd.DataFrame,
    show: bool = True,
) -> folium.Map:
    """Add signal anomaly density as a hex-grid layer.

    Only cells with anomaly_count > 0 are drawn.

    Args:
        map_obj: Target Folium map.
        df_anomaly: Dataframe with ``h3_index`` and ``anomaly_count`` columns
            (output of ``count_anomaly_per_cell``).
        show: Whether to show this layer by default.

    Returns:
        The same map_obj for chaining.
    """
    active = df_anomaly[df_anomaly["anomaly_count"] > 0]
    if active.empty:
        return map_obj
    cmap = _get_population_colormap(
        active.rename(columns={"anomaly_count": "pop"}),
        caption="Anomaly Count",
        min_population=0,
        default_max=float(active["anomaly_count"].quantile(0.95)) or 1.0,
    )
    fg = FeatureGroup(name="Signal Anomaly Density", show=show)
    for cell in active[["h3_index", "anomaly_count"]].to_dict("records"):
        folium.Polygon(
            locations=_hex_boundary_points(cell["h3_index"]),
            fill=True, fill_color=cmap(cell["anomaly_count"]),
            color="#ff4400", weight=0.6, fill_opacity=0.65,
            tooltip=f"Anomaly count: {cell['anomaly_count']}",
        ).add_to(fg)
    fg.add_to(map_obj)
    map_obj.add_child(cmap)
    return map_obj


def add_scatter_layer(
    map_obj: folium.Map,
    df_scatter: pd.DataFrame,
    show: bool = False,
) -> folium.Map:
    """Add raw measurement points as a scatter layer of circle markers.

    Args:
        map_obj: Target Folium map.
        df_scatter: Dataframe with ``lat`` and ``lon`` columns.
        show: Whether to show this layer by default (False — 19k points
            are slow to render; toggle on when needed).

    Returns:
        The same map_obj for chaining.
    """
    if df_scatter.empty:
        return map_obj
    fg = FeatureGroup(name="Measurement Points", show=show)
    for row in df_scatter[["lat", "lon"]].to_dict("records"):
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=2,
            color="#2266cc",
            fill=True,
            fill_color="#2266cc",
            fill_opacity=0.5,
            weight=0,
        ).add_to(fg)
    fg.add_to(map_obj)
    return map_obj


# ---------------------------------------------------------------------------
# Convenience compositor
# ---------------------------------------------------------------------------

def visualize_town_zones(
    df_urban: pd.DataFrame,
    df_town_grid: pd.DataFrame,
    df_clusters: pd.DataFrame,
    output_file: str = "h3_zones.html",
    zoom_start: int = 7,
    tiles: str = "CartoDB positron",
) -> folium.Map:
    """Compose and save the standard zone map (town areas + town grid).

    For additional layers (anomaly, scatter, etc.) call the individual
    ``add_*_layer`` functions on the returned map before saving, or build
    the map manually using ``create_zone_map`` / ``add_*_layer`` /
    ``finalize_map``.

    Args:
        df_urban: Town cells with ``h3_index``, ``pop``, ``cluster_id``.
        df_town_grid: Fine-res grid with ``h3_index`` and ``pop``.
        df_clusters: Cluster summary sorted by descending ``total_pop``.
        output_file: Path to write the HTML file.
        zoom_start: Initial map zoom level.
        tiles: Folium tile layer name.

    Returns:
        The saved Folium map object.
    """
    if df_urban.empty:
        raise ValueError("df_urban is empty — no town cells to map.")
    map_obj = create_zone_map(df_urban, zoom_start=zoom_start, tiles=tiles)
    add_town_areas_layer(map_obj, df_urban, df_clusters)
    add_town_grid_layer(map_obj, df_town_grid)
    return finalize_map(map_obj, output_file)


# ---------------------------------------------------------------------------
# Legacy / standalone visualizers
# ---------------------------------------------------------------------------

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
