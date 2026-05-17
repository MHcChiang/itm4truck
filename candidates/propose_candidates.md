# Rural Candidate Proposal

This document explains how we build the list of candidate tower locations for the
rural Maximum Coverage Problem (MCP). All candidates are expressed as **H3 Res-5
cell IDs** (edge ~8.5 km), consistent with the rural demand grid.

> **No DEM download required.** Elevation is fetched via `py3dep.elevation_bycoords()`
> which queries the USGS 3DEP API at specific point coordinates only — no raster file
> is downloaded.

---

## Overview

```
Two sources → combine → deduplicate → save as rural_candidates.csv

Source 1: HIFLD existing towers  ──┐
                                    ├──► combined candidate list
Source 2: Demand hotspot cells   ──┘
```

---

## Source 1 — HIFLD Existing Cellular Towers

**File:** `data/Cellular_Towers_in_US.csv` (~289k towers nationwide)

**Steps:**
1. Filter to our bounding box using `latdec` / `londec` columns.
2. Map each tower's (lat, lon) to its Res-5 H3 cell with `h3.latlng_to_cell`.
3. If multiple towers fall in the same cell, keep one (deduplicate by cell).

**Why include existing towers?** An existing tower is already built — the GA can
select it at low marginal cost. Knowing where they are also lets us avoid placing
new towers right next to them.

**Output columns:** `h3_index`, `lat`, `lon`, `source="hifld"`

---

## Source 2 — Demand Hotspot Candidates

We propose new tower locations at rural cells with unmet demand — from
**population** or **signal anomalies recorded by the truck**.

### Step 1 — Score demand per rural cell (with normalization)

`pop` and `anomaly_count` are on completely different scales (thousands vs. single
digits), so a direct sum would let population dominate. Each factor is first
**min-max normalised** to [0, 1], then combined with equal weights:

```
norm_pop      = (pop - pop.min()) / (pop.max() - pop.min())
norm_anomaly  = (anomaly_count - min) / (max - min)
demand        = 0.5 × norm_pop + 0.5 × norm_anomaly
```

- `pop` from `processed_data/rural_cells.csv` (WorldPop raster)
- `anomaly_count` from `processed_data/anomaly_counts.csv` (truck signal anomaly points)

Cells with `demand == 0` (zero population AND zero anomalies) are dropped.

### Step 2 — Exclude cells too close to existing HIFLD towers

Any demand cell whose center is within **`exclusion_km`** (default 15 km) of any
HIFLD tower is removed. Distance is computed as a geodesic distance using
`pyproj.Geod` so it is accurate regardless of map projection.

Grid-ring exclusion (`grid_disk(k=1)`) was too coarse — it removed all cells
within one full Res-5 ring (~8.5 km), which is the same width as the cell itself
and excluded most viable candidates. Distance-based exclusion lets you set an
exact threshold and tune it freely.

```
demand cell center ──► geodesic distance to each HIFLD tower
                              ↓
                    keep if min_distance ≥ exclusion_km
```

Tune `exclusion_km` to trade off between:
- **Too large** (e.g. 30 km): very few new candidates, might miss coverage gaps
- **Too small** (e.g. 5 km): many redundant candidates near existing towers

### Step 3 — Refine to hilltop within the Res-5 cell

A Res-5 cell covers ~200 km² — using its center as the tower location is a rough
approximation. For better LOS coverage, we refine each candidate to the best spot
within the cell:

1. Enumerate all **Res-8 children** of the Res-5 cell (7³ = 343 children, ~0.5 km edge each).
2. Query elevation at every child's center in **one batched `py3dep.elevation_bycoords()` call** — the API returns elevations for a list of coordinates directly from USGS 3DEP without downloading any raster file.
3. Keep the child with the **highest elevation** as the actual tower location.

```
Res-5 cell (8.5 km edge)
  └─ 343 Res-8 children (0.5 km edge)
       │
       └─ py3dep.elevation_bycoords([center_coords...])  ← point API, no raster
            │
            └─ pick highest elevation child → tower location
```

### Step 4 — Record as candidates

Each surviving cell becomes one row:
- `h3_index` — the **Res-8** child cell ID (precise tower location)
- `lat`, `lon` — center of that Res-8 child
- `elevation` — metres above sea level at that point
- `demand` — normalised demand score inherited from the parent Res-5 cell
- `source="demand"`

---

## Combining Both Sources

```python
df_all = pd.concat([df_hifld, df_demand]).drop_duplicates("h3_index")
```

If an HIFLD tower already sits in a high-demand cell, that cell appears once with
`source="hifld"`. The demand list fills in cells with no existing tower.

---

## Output

**File:** `processed_data/rural_candidates.csv`

| Column | Description |
|---|---|
| `h3_index` | Res-8 H3 cell ID — the hilltop-refined tower location |
| `lat`, `lon` | Center of the cell |
| `elevation` | DEM elevation at cell center (metres) |
| `demand` | Combined pop + anomaly score (`NaN` for HIFLD rows) |
| `source` | `"hifld"` or `"demand"` |

This file is the **direct input to the GA precompute step** — each row is one
candidate site for which we compute an RSSI coverage vector.

---

## Key Parameters

| Parameter | Default | Effect |
|---|---|---|
| `resolution` | 5 | H3 resolution (~8.5 km edge) — matches rural demand grid |
| `min_demand` | 0.0 | Raise to filter out very low-demand cells |
| `exclusion_km` | 15.0 | Minimum km from any HIFLD tower for new proposals |

---

## Usage

```python
from candidates.candidates import build_rural_candidates
import pandas as pd

df_rural = pd.read_csv("processed_data/rural_cells.csv")
df_anomaly = pd.read_csv("processed_data/anomaly_counts.csv")

bbox = {"north": 38.457485, "south": 33.464998, "east": -77.000644, "west": -87.998634}

df_candidates = build_rural_candidates(
    hifld_csv="data/Cellular_Towers_in_US.csv",
    df_rural=df_rural,
    df_anomaly=df_anomaly,
    bbox=bbox,
    exclusion_km=15.0,           # lower to get more candidates
    output_path="processed_data/rural_candidates.csv",
)
```

No DEM path is needed — elevation is queried from the USGS 3DEP API on demand.

