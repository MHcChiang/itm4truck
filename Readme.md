# Cellular Coverage Optimization — IEEE Truck Project

This repository implements a **Maximum Coverage Problem (MCP)** pipeline for rural cellular base station placement. It combines population demand modeling, signal anomaly data from a measurement truck, and Longley-Rice ITM propagation to select the best subset of candidate tower locations via a Genetic Algorithm.

## Pipeline Overview

```
WorldPop raster + HIFLD towers + truck anomaly data
        │
        ▼
  process_pop.py          ← Zone classification, demand scoring, candidate generation
        │
        ▼
  precal_itm.py           ← ITM coverage precomputation (coverage matrix)
        │
        ▼
  [GA solver — in progress]
        │
        ▼
  viz_processed.py        ← Interactive map visualization
```

All geographic and RF parameters are centralized in **`config.yaml`**.

---

## Setup

This project requires geospatial C++ dependencies. Use Conda — do not use `pip install` directly.

### 1. Clone the repository

```bash
git clone https://github.com/MHcChiang/itm4truck.git
cd itm4truck
```

### 2. Install `itmlogic` (Longley-Rice model)

```bash
git clone https://github.com/edwardoughton/itmlogic.git
cd itmlogic && python setup.py develop && cd ..
```

### 3. Create the Python environment

```bash
conda env create -f environment.yml
conda activate Truck
```

---

## Configuration — `config.yaml`

All scripts load defaults from `config.yaml`. CLI arguments always override config values.

```yaml
region:          # bounding box (decimal degrees)
  north / south / east / west

h3:
  base_res: 6    # rural demand grid (~3.2 km edge)
  refined_res: 8 # town fine grid (~0.5 km edge)

dem:
  path: data/dem_data/target_area_dem.tif
  resolution: 90 # metres

rf:
  freq_mhz: 700.0
  tx_height_m: 30.0
  rx_height_m: 2.0
  tx_power_dbm: 43.0
  rssi_threshold_dbm: -90.0
  max_coverage_km: 30.0   # distance pruning cutoff for ITM precompute

paths:           # data file locations
  pop_file / hifld_csv / anomaly_file / scatter_file / out_dir / fig_dir
```

---

## Scripts

### `process_pop.py` — Demand pipeline

Processes population raster and truck anomaly data into an H3 grid, classifies urban/rural zones, and generates candidate tower locations.

```bash
python process_pop.py
python process_pop.py --min-core-pop 5.0 --base-res 6
```

**Steps:**
1. Sample WorldPop raster into Res-6 H3 base layer
2. Seed selection → k-ring expansion → BFS clustering → urban / rural classification
3. Generate fine Res-8 grid within town areas (sampled from raster)
4. Count truck signal anomaly points per Res-6 cell
5. Build rural candidate towers (HIFLD existing + demand hotspots)

**Outputs** (`processed_data/`):

| File | Description |
|---|---|
| `base_layer.csv` | All Res-6 H3 cells with population and `grid_type` (urban/rural) |
| `urban_cells.csv` | Town cells with `cluster_id` |
| `rural_cells.csv` | Rural demand cells — input to ITM precompute |
| `clusters.csv` | Cluster summary (id, total_pop, centroid) sorted by population |
| `town_grid.csv` | Res-8 fine grid within town areas |
| `anomaly_counts.csv` | Truck signal anomaly count per Res-6 cell |
| `rural_candidates.csv` | Candidate tower locations (HIFLD + demand-driven) |

---

### `precal_itm.py` — ITM coverage precomputation

For every candidate tower × rural demand cell pair, runs Longley-Rice ITM p2p to produce a binary coverage matrix. Uses distance pruning and windowed DEM loading for efficiency.

```bash
python precal_itm.py
python precal_itm.py --max-coverage-km 30 --workers 8
```

**Key design choices:**
- **Distance pruning**: pairs beyond `max_coverage_km` skip ITM entirely (~90% reduction in calls)
- **Windowed DEM**: each worker loads only the `~60 km` patch needed for one candidate (~3.5 MB) instead of the full raster
- **ProcessPoolExecutor**: one task = one candidate (all its in-range demand pairs), bypassing Python GIL for true CPU parallelism

**Output** (`processed_data/coverage_matrix.npz`):

| Array | Shape | Description |
|---|---|---|
| `coverage` | (n_cand, n_demand) uint8 | 1 if RSSI ≥ threshold, 0 otherwise |
| `rssi` | (n_cand, n_demand) float32 | Raw RSSI values in dBm |
| `candidates` | (n_cand,) str | H3 index of each candidate row |
| `demand` | (n_demand,) str | H3 index of each demand column |

---

### `viz_processed.py` — Interactive map visualization

Loads all processed CSVs and generates interactive Folium HTML maps. Re-run anytime to update maps without rerunning the full pipeline.

```bash
python viz_processed.py
python viz_processed.py --zones-file my_zones.html
```

**Outputs** (`fig/`):

| File | Layers |
|---|---|
| `h3_zones.html` | Town areas (by cluster), town fine grid, signal anomalies, scatter points, candidate towers |
| `coverage.html` | HIFLD coverage (green/red), demand-tower coverage, candidate markers |
| `pop_dist.html` | Base-layer population heatmap |

---

### `main.py` — Single-tower ITM estimation (original)

Computes RSSI from one fixed transmitter over a rectangular or H3 grid. Used for single-site validation and visualization.

```bash
python main.py          # rectangular grid
python main.py --h3     # H3 hexagonal grid
```

---

## Module Reference

### `DHS/` — Demand hotspot processing

| Module | Key functions |
|---|---|
| `pop.py` | `generate_base_layer(tif, bbox, base_res)` — sample WorldPop raster into H3 cells |
| `zone.py` | `classify_zones(df_base)` — urban/rural BFS clustering; `generate_town_grid(df_urban, tif)` — fine Res-8 population grid |
| `anomaly.py` | `count_anomaly_per_cell(parquet, h3_cells, resolution)` — aggregate truck anomaly points per H3 cell |

### `candidates/` — Candidate tower generation

| Module | Key functions |
|---|---|
| `candidates.py` | `build_rural_candidates(hifld_csv, df_rural, df_anomaly, bbox)` — combines HIFLD towers and demand-hotspot proposals into one candidate list |

**Candidate sources:**
- **HIFLD**: existing US cellular towers filtered to bbox, mapped to Res-6 H3 cells
- **Demand-driven**: rural Res-6 cells scored by `0.5 × norm_pop + 0.5 × norm_anomaly`, refined to the highest-elevation Res-8 child via USGS 3DEP point queries (`py3dep.elevation_bycoords`)

Cells within `exclusion_km` (default 15 km) of any HIFLD tower are excluded using geodesic distance.

### `itm/` — ITM signal estimation

| Module | Key functions |
|---|---|
| `dem_downloader.py` | `download_and_crop(west, south, east, north, resolution)` — fetch DEM from USGS 3DEP via py3dep |
| `signal_estimator.py` | `calculate_memory_p2p_rssi(tx_lon, tx_lat, rx_lon, rx_lat, dem_array, inv_transform, params, power)` — single-path RSSI via in-memory terrain profile; `estimate_hex_grid_signal(...)` — full H3 grid estimation with threading |
| `visualizer.py` | `plot_hex_signal_map(...)` — RSSI heatmap overlay on DEM |

### `viz/` — Visualization

`viz_pop.py` implements a **composable layer system** for Folium maps:

```python
map_obj = create_zone_map(df_urban)        # initialize map
add_town_areas_layer(map_obj, ...)         # add layers as needed
add_town_grid_layer(map_obj, ...)
add_anomaly_layer(map_obj, ...)
add_candidates_layer(map_obj, ...)
add_coverage_layer(map_obj, ...)           # from coverage_matrix.npz
add_scatter_layer(map_obj, ...)
finalize_map(map_obj, "output.html")       # add LayerControl and save
```

### `itmlogic/` — Longley-Rice model (external submodule)

Python implementation of the ITM propagation model. Key entry point: `itmlogic_p2p()` from `itmlogic/scripts/p2p.py`. Do not modify files inside this submodule.

---

## Data Directory

```
data/
  dem_data/
    target_area_dem.tif          ← 90 m DEM auto-downloaded by precal_itm.py
  Cellular_Towers_in_US.csv      ← HIFLD tower database (~289k towers)
  usa_pop_2026_CN_100m_R2025A_v1.tif  ← WorldPop 100 m population raster
  df_flags.parquet               ← Raw truck measurement points (lat, lon)
  h3_demand_point_level.parquet  ← Truck signal anomaly points (lat, lon)

processed_data/                  ← Generated by process_pop.py + precal_itm.py
fig/                             ← Generated by viz_processed.py
```

---

## Run Order

```bash
# 1. Generate demand grid and candidates
python process_pop.py

# 2. Pre-compute ITM coverage matrix
#    (auto-downloads 90 m DEM if missing or bbox mismatch)
python precal_itm.py

# 3. Visualize results
python viz_processed.py
```
