"""Anomaly point aggregation to H3 grid cells."""

from typing import List

import h3
import pandas as pd


def count_anomaly_per_cell(
    parquet_path: str,
    h3_cells: List[str],
    resolution: int = 5,
) -> pd.DataFrame:
    """Count signal anomaly points falling within each H3 cell.

    Maps each (lat, lon) anomaly point to its H3 cell at the given resolution,
    then counts points per cell. Cells with no anomalies get a count of 0.

    Args:
        parquet_path: Path to the anomaly points parquet file. Must have
            ``lat`` and ``lon`` columns.
        h3_cells: Full list of H3 cell IDs (the base grid) so that every
            cell appears in the output, even those with zero anomalies.
        resolution: H3 resolution to aggregate to — must match the base grid.

    Returns:
        DataFrame with columns ``h3_index`` and ``anomaly_count``, one row
        per cell in ``h3_cells``.
    """
    print(f"--- Counting Anomaly Points (Res {resolution}) ---")
    df = pd.read_parquet(parquet_path)

    df["h3_index"] = [
        h3.latlng_to_cell(lat, lon, resolution)
        for lat, lon in zip(df["lat"], df["lon"])
    ]
    counts = df.groupby("h3_index").size().reset_index(name="anomaly_count")

    result = pd.DataFrame({"h3_index": h3_cells}).merge(counts, on="h3_index", how="left")
    result["anomaly_count"] = result["anomaly_count"].fillna(0).astype(int)

    nonzero = (result["anomaly_count"] > 0).sum()
    print(f"Anomaly points: {len(df)} across {nonzero} cells (out of {len(h3_cells)} total).")
    return result
