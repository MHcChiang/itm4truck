"""Rural candidate site generation for the Maximum Coverage Problem.

Two candidate sources combined into one list:
  1. HIFLD existing cellular towers — filter to bbox, map to Res-5 H3 cells.
  2. Demand hotspots — rural Res-5 cells scored by normalized (pop + anomaly),
     refined to the highest-elevation Res-8 child via point-based elevation
     queries (py3dep.elevation_bycoords — no full raster download needed).
"""

from typing import Dict, List, Optional, Tuple

import h3
import numpy as np
import pandas as pd
import py3dep
import pyproj

_GEOD = pyproj.Geod(ellps="WGS84")


# ---------------------------------------------------------------------------
# Distance helper
# ---------------------------------------------------------------------------

def _min_dist_to_towers_km(
    demand_latlons: List[Tuple[float, float]],
    hifld_latlons: List[Tuple[float, float]],
) -> np.ndarray:
    """Minimum geodesic distance (km) from each demand cell to the nearest HIFLD tower.

    Vectorises over demand cells for each tower, so cost is O(N × M) with
    numpy ops rather than a pure-Python double loop.

    Args:
        demand_latlons: (lat, lon) of each demand cell center.
        hifld_latlons: (lat, lon) of each HIFLD tower.

    Returns:
        Array of shape (N,) — minimum distance in km to any HIFLD tower.
    """
    if not hifld_latlons:
        return np.full(len(demand_latlons), float("inf"))

    d_lats = np.array([ll[0] for ll in demand_latlons])
    d_lons = np.array([ll[1] for ll in demand_latlons])

    min_dists = np.full(len(demand_latlons), float("inf"))
    for h_lat, h_lon in hifld_latlons:
        _, _, dists_m = _GEOD.inv(
            d_lons, d_lats,
            np.full(len(d_lons), h_lon),
            np.full(len(d_lats), h_lat),
        )
        min_dists = np.minimum(min_dists, np.array(dists_m) / 1000.0)

    return min_dists


# ---------------------------------------------------------------------------
# Elevation helpers — point queries, no raster download
# ---------------------------------------------------------------------------

def _query_elevation(h3_cells: List[str]) -> Dict[str, float]:
    """Query USGS 3DEP elevation at each H3 cell center via py3dep.

    Uses ``py3dep.elevation_bycoords`` which fetches elevation at discrete
    coordinates only — no full raster is downloaded, so it scales to any
    bounding box size without memory issues.

    Args:
        h3_cells: H3 cell IDs whose centers to query.

    Returns:
        Dict mapping h3_index → elevation in metres (0.0 for failures).
    """
    # py3dep expects (lon, lat) tuples in EPSG:4326
    coords = [(lon, lat) for lat, lon in (h3.cell_to_latlng(c) for c in h3_cells)]
    elevations = py3dep.elevation_bycoords(coords, crs="EPSG:4326")
    return {
        cell: max(float(e), 0.0)
        for cell, e in zip(h3_cells, elevations)
    }


def _refine_to_hilltop(
    res5_cells: List[str],
    fine_res: int = 8,
) -> pd.DataFrame:
    """Within each Res-5 cell, find the Res-8 child with the highest elevation.

    Enumerates all Res-8 children (343 per parent) and queries elevation at
    every child center in one batched py3dep call, then keeps the
    highest-elevation child per parent as the actual tower location.

    Args:
        res5_cells: Res-5 H3 cell IDs selected as candidate sites.
        fine_res: Child resolution (default 8, ~0.5 km edge).

    Returns:
        DataFrame with ``h3_res5``, ``h3_index``, ``lat``, ``lon``,
        ``elevation`` columns.
    """
    all_children: List[str] = []
    all_parents: List[str] = []
    for cell in res5_cells:
        children = list(h3.cell_to_children(cell, fine_res))
        all_children.extend(children)
        all_parents.extend([cell] * len(children))

    print(f"Querying elevation for {len(all_children)} Res-{fine_res} children "
          f"across {len(res5_cells)} Res-5 cells...")
    elev_map = _query_elevation(all_children)

    df = pd.DataFrame({"h3_res5": all_parents, "h3_index": all_children})
    df["elevation"] = df["h3_index"].map(elev_map)
    df = df.sort_values("elevation", ascending=False).drop_duplicates("h3_res5")

    latlngs = [h3.cell_to_latlng(c) for c in df["h3_index"]]
    df["lat"] = [ll[0] for ll in latlngs]
    df["lon"] = [ll[1] for ll in latlngs]
    return df[["h3_res5", "h3_index", "lat", "lon", "elevation"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Source 1 — HIFLD existing towers
# ---------------------------------------------------------------------------

def load_hifld_in_bbox(
    csv_path: str,
    bbox: Dict[str, float],
    resolution: int = 5,
) -> pd.DataFrame:
    """Load HIFLD towers within the bounding box and map to H3 cells.

    Multiple towers that fall in the same H3 cell are collapsed to one row.

    Args:
        csv_path: Path to Cellular_Towers_in_US.csv.
        bbox: Dict with ``north``, ``south``, ``east``, ``west`` keys.
        resolution: H3 resolution (default 5, ~8.5 km edge).

    Returns:
        DataFrame with ``h3_index``, ``lat``, ``lon``, ``elevation``,
        ``demand``, ``source`` columns.
    """
    df = pd.read_csv(csv_path, usecols=["latdec", "londec"])
    df = df.dropna(subset=["latdec", "londec"])
    df = df[
        (df["latdec"] >= bbox["south"]) & (df["latdec"] <= bbox["north"]) &
        (df["londec"] >= bbox["west"])  & (df["londec"] <= bbox["east"])
    ].copy()

    df["h3_index"] = [
        h3.latlng_to_cell(lat, lon, resolution)
        for lat, lon in zip(df["latdec"], df["londec"])
    ]
    df = df.groupby("h3_index", as_index=False).first()
    df = df.rename(columns={"latdec": "lat", "londec": "lon"})
    df["source"] = "hifld"
    df["demand"] = float("nan")

    # Query elevation at the actual tower coordinates (not cell center).
    coords = [(lon, lat) for lat, lon in zip(df["lat"], df["lon"])]
    elevations = py3dep.elevation_bycoords(coords, crs="EPSG:4326")
    df["elevation"] = [max(float(e), 0.0) for e in elevations]

    print(f"HIFLD towers in bbox: {len(df)} unique Res-{resolution} cells.")
    return df[["h3_index", "lat", "lon", "elevation", "demand", "source"]]


# ---------------------------------------------------------------------------
# Source 2 — Demand-driven candidates
# ---------------------------------------------------------------------------

def propose_demand_candidates(
    df_rural: pd.DataFrame,
    df_anomaly: Optional[pd.DataFrame],
    df_hifld: pd.DataFrame,
    min_demand: float = 0.0,
    exclusion_km: float = 15.0,
    w_pop: float = 0.5,
    w_anomaly: float = 0.5,
    fine_res: int = 8,
) -> pd.DataFrame:
    """Propose new tower candidates from rural demand hotspot cells.

    Demand is a weighted sum of min-max normalised population and anomaly
    count. Cells within ``exclusion_km`` of any existing HIFLD tower are
    excluded using geodesic distance rather than H3 grid rings, giving
    finer control over the spacing threshold. Each surviving Res-5 cell is
    refined to its highest-elevation Res-8 child via py3dep point queries.

    Args:
        df_rural: Rural cells with ``h3_index`` and ``pop`` columns.
        df_anomaly: Optional anomaly counts with ``h3_index`` and
            ``anomaly_count``. If None, only population drives demand.
        df_hifld: HIFLD towers DataFrame with ``lat`` and ``lon`` columns
            (output of ``load_hifld_in_bbox``).
        min_demand: Minimum normalised demand (0–1 scale) to keep a cell.
        exclusion_km: Demand cells closer than this distance (km) to any
            HIFLD tower are excluded (default 15 km).
        w_pop: Weight for normalised population (default 0.5).
        w_anomaly: Weight for normalised anomaly count (default 0.5).
        fine_res: Child resolution for hilltop refinement (default 8).

    Returns:
        DataFrame with ``h3_index``, ``lat``, ``lon``, ``elevation``,
        ``demand``, ``source`` columns.
    """
    df = df_rural[["h3_index", "pop"]].copy()

    if df_anomaly is not None:
        df = df.merge(df_anomaly[["h3_index", "anomaly_count"]], on="h3_index", how="left")
        df["anomaly_count"] = df["anomaly_count"].fillna(0)
    else:
        df["anomaly_count"] = 0.0

    def _minmax(series: pd.Series) -> pd.Series:
        lo, hi = series.min(), series.max()
        return (series - lo) / (hi - lo) if hi > lo else pd.Series(0.0, index=series.index)

    df["demand"] = w_pop * _minmax(df["pop"]) + w_anomaly * _minmax(df["anomaly_count"])
    df = df[df["demand"] > min_demand].copy()
    print(f"Rural cells with demand > {min_demand}: {len(df)}")

    # Distance-based exclusion: drop demand cells within exclusion_km of any HIFLD tower.
    demand_latlons = [h3.cell_to_latlng(c) for c in df["h3_index"]]
    hifld_latlons  = list(zip(df_hifld["lat"], df_hifld["lon"]))
    min_dists = _min_dist_to_towers_km(demand_latlons, hifld_latlons)
    df = df[min_dists >= exclusion_km].copy()
    print(f"After HIFLD exclusion (>{exclusion_km} km): {len(df)} candidate cells.")

    if df.empty:
        return pd.DataFrame(columns=["h3_index", "lat", "lon", "elevation", "demand", "source"])

    refined = _refine_to_hilltop(df["h3_index"].tolist(), fine_res=fine_res)

    demand_map = df.set_index("h3_index")["demand"].to_dict()
    refined["demand"] = refined["h3_res5"].map(demand_map)
    refined["source"] = "demand"

    refined = refined.sort_values("elevation", ascending=False)
    print(f"Demand candidates after hilltop refinement: {len(refined)}")
    return refined[["h3_index", "lat", "lon", "elevation", "demand", "source"]]


# ---------------------------------------------------------------------------
# Combined candidate list
# ---------------------------------------------------------------------------

def build_rural_candidates(
    hifld_csv: str,
    df_rural: pd.DataFrame,
    df_anomaly: Optional[pd.DataFrame],
    bbox: Dict[str, float],
    resolution: int = 5,
    min_demand: float = 0.0,
    exclusion_km: float = 10.0,
    w_pop: float = 0.5,
    w_anomaly: float = 0.5,
    fine_res: int = 8,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Build the full rural candidate list combining HIFLD + demand hotspots.

    Args:
        hifld_csv: Path to Cellular_Towers_in_US.csv.
        df_rural: Rural cells DataFrame (output of classify_zones).
        df_anomaly: Optional anomaly counts (output of count_anomaly_per_cell).
        bbox: Bounding box with north/south/east/west keys.
        resolution: H3 resolution for candidate cells (default 5, ~8.5 km edge).
        min_demand: Minimum normalised demand threshold (0–1 scale).
        exclusion_km: Minimum distance (km) from any HIFLD tower for new
            proposals (default 15 km). Lower to include more candidates.
        w_pop: Weight for normalised population in demand score.
        w_anomaly: Weight for normalised anomaly count in demand score.
        fine_res: Child resolution for hilltop refinement (default 8).
        output_path: If provided, saves combined candidates as CSV here.

    Returns:
        DataFrame with ``h3_index``, ``lat``, ``lon``, ``elevation``,
        ``demand``, ``source`` columns.
    """
    print("=== Building Rural Candidate List ===")

    df_hifld = load_hifld_in_bbox(hifld_csv, bbox, resolution=resolution)

    df_demand = propose_demand_candidates(
        df_rural=df_rural,
        df_anomaly=df_anomaly,
        df_hifld=df_hifld,
        min_demand=min_demand,
        exclusion_km=exclusion_km,
        w_pop=w_pop,
        w_anomaly=w_anomaly,
        fine_res=fine_res,
    )

    df_all = pd.concat([df_hifld, df_demand], ignore_index=True)
    df_all = df_all.drop_duplicates("h3_index")

    print(f"\nTotal rural candidates: {len(df_all)} "
          f"({len(df_hifld)} HIFLD + {len(df_demand)} demand-driven)")

    if output_path:
        df_all.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

    return df_all
