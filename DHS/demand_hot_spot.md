# Demand Hotspot Pipeline

This pipeline processes raw population raster data into structured H3 hex grids, classifies urban ("town") and rural zones, and exports the result as CSV files ready for the Genetic Algorithm (GA).

---

## What It Does (Big Picture)

```
Population Raster (.tif)
        │
        ▼
[1] Sample at low-res H3 grid (Res 5, ~110 km edge)
        │   → base_layer.csv
        ▼
[2] Identify "town seeds" (cells above population threshold)
    → Expand outward by 1 ring (to fill internal gaps like parks)
    → Group touching cells into clusters (Atlanta, Charlotte, etc.)
        │   → urban_cells.csv
        ▼
[3] For each town cluster, generate a fine-grained grid (Res 7, ~5 km edge)
        │   → town_grid.csv
        ▼
[4] Visualize results as interactive HTML maps
        │   → fig/h3_zones.html
        │   → fig/pop_dist.html
```

The output CSVs are the direct input to the GA — each row is one hex cell with a population weight.

---

## Key Concept: H3 Hexagonal Grid

We use [Uber H3](https://h3geo.org/) to divide the map into hexagons. Unlike square grids, hexagons have equal-distance neighbors in all 6 directions, which makes spatial operations (coverage, adjacency) more accurate.

Each cell is identified by an **H3 index** (a hex string like `"85283473fffffff"`). From the index you can get:
- Center coordinates: `h3.cell_to_latlng(cell_id)`
- Boundary polygon: `h3.cell_to_boundary(cell_id)`
- Neighbors: `h3.grid_disk(cell_id, k=1)`

**Resolution** controls cell size:

| Resolution | Edge Length | Use in This Pipeline |
|:---:|:---:|:---|
| 5 | ~108 km | Base layer — coarse population sampling |
| 7 | ~5.4 km | Town grid — fine-grained demand for GA |

---

## Step-by-Step Workflow

### Step 1 — Sample Population at Low Resolution

We place a Res-5 hex grid over the target area and sample the WorldPop raster at each hex center. This gives one population density value per large hex.

- `raw_pixel_pop` — raw value from the raster: **people per 100m × 100m pixel**
- `pop` — estimated total population of the hex (`raw_pixel_pop × ~25,000 pixels`)

> **Output:** `data/base_layer.csv`

---

### Step 2 — Classify Town vs Rural Zones

**"Core" cell:** a Res-5 cell where `raw_pixel_pop > MIN_CORE_POP`.  
Only cells above this density threshold become town seeds.

Then we **expand** each seed by 1 H3 ring outward. This:
- Connects seeds that are close together into one continuous town zone
- Fills internal "holes" (parks, stadiums) that show zero population in the raster

Finally, **connected components** (BFS on H3 adjacency) groups touching cells into named clusters. Cluster #1 = largest population, Cluster #2 = second largest, etc.

```
raw pop > threshold → "core" seed
              ↓
     expand 1 ring → town zone
              ↓
   BFS adjacency → cluster label
```

> **Output:** `data/urban_cells.csv` (town cells with cluster_id)

**Tuning `MIN_CORE_POP`:**

| Value | Meaning | Effect |
|---|---|---|
| 1 | 100 people/km² — sparse rural | Too many seeds, clusters merge |
| 10 | 1,000 people/km² — suburban | Good starting point |
| 50 | 5,000 people/km² — dense urban | Only city centers qualify |

If clusters are still too merged, increase `MIN_CORE_POP` or set `--expansion-rings 0`.

---

### Step 3 — Generate Fine-Grained Town Grid

For each town cell (Res 5), we enumerate all **Res-7 child cells** (~343 per parent) and sample population at each child center. Only children with `pop > 0` are kept.

This gives a fine-grained demand map inside town boundaries — the actual input weights for the GA.

> **Output:** `data/town_grid.csv`

---

### Step 4 — Visualize

Two interactive HTML maps are generated in `./fig/`:

| File | What It Shows |
|---|---|
| `h3_zones.html` | Town clusters (colored by cluster ID) + fine Res-7 grid inside towns |
| `pop_dist.html` | All Res-5 cells colored by population (blue = sparse → red = dense) |

Open either file in any web browser. Use the layer panel (top right) to toggle layers on/off.

---

## How to Run

```bash
# Default settings
python process_pop.py

# Custom bounding box and thresholds
python process_pop.py \
  --north 38.5 --south 33.5 --east -77.0 --west -88.0 \
  --min-core-pop 20 \
  --expansion-rings 1 \
  --base-res 5 \
  --refined-res 7
```

---

## Output Files for the GA

After running, `./data/` contains:

| File | Columns | Description |
|---|---|---|
| `base_layer.csv` | `h3_index, pop, res, raw_pixel_pop` | All Res-5 cells — full area map |
| `urban_cells.csv` | `h3_index, pop, res, raw_pixel_pop, cluster_id` | Town cells only, labeled by cluster |
| `town_grid.csv` | `h3_index, pop, res, parent, cluster_id` | Fine Res-7 demand grid — **primary GA input** |

### Loading in Python

```python
import pandas as pd
import h3

df = pd.read_csv("data/town_grid.csv")

# Get lat/lon center of each cell
df["lat"] = df["h3_index"].apply(lambda c: h3.cell_to_latlng(c)[0])
df["lon"] = df["h3_index"].apply(lambda c: h3.cell_to_latlng(c)[1])

# Pop column is the demand weight for the GA fitness function
demand_weights = df.set_index("h3_index")["pop"].to_dict()
```

---

## File Structure

```
Code/
├── process_pop.py              ← Entry point (run this)
├── DHS/
│   ├── pop.py               ← Step 1: raster sampling → H3 DataFrame
│   ├── zone.py              ← Step 2 & 3: zone classification + fine grid
│   └── viz_pop.py           ← Step 4: Folium map generation
├── data/
│   ├── dem_data/            ← DEM raster downloads
│   ├── *.tif                ← WorldPop raster (input)
│   └── *.parquet            ← Raw truck/anomaly measurement data
├── data/
│   ├── base_layer.csv       ← Output: low-res full grid
│   ├── urban_cells.csv      ← Output: town zone cells
│   └── town_grid.csv        ← Output: fine-res demand grid (GA input)
└── fig/
    ├── h3_zones.html        ← Output: zone visualization
    └── pop_dist.html        ← Output: population heatmap
```
