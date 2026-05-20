"""Demo: working with coverage_matrix.npz.

Shows how to load the coverage matrix, inspect statistics, filter by tower
source, and run a greedy site selection — the building block for the GA.

Run after precal_itm.py has produced processed_data/coverage_matrix.npz.

Usage:
    python demo_coverage.py
    python demo_coverage.py --k 20
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_coverage(npz_path: str, candidates_csv: str) -> dict:
    """Load coverage matrix and annotate with candidate metadata.

    Args:
        npz_path: Path to coverage_matrix.npz.
        candidates_csv: Path to rural_candidates.csv.

    Returns:
        Dict with keys: coverage, rssi, candidate_h3, demand_h3,
        rssi_threshold, df_candidates.
    """
    npz = np.load(npz_path, allow_pickle=True)
    coverage     = npz["coverage"].astype(bool)   # (n_cand, n_dem)
    rssi         = npz["rssi"]                    # (n_cand, n_dem) float32
    candidate_h3 = npz["candidates"].tolist()
    demand_h3    = npz["demand"].tolist()
    threshold    = float(npz["rssi_threshold"])

    df_cand = pd.read_csv(candidates_csv)
    # Align df_cand to the matrix row order
    h3_to_row   = {h: i for i, h in enumerate(candidate_h3)}
    df_cand["matrix_row"] = df_cand["h3_index"].map(h3_to_row)
    df_cand = df_cand.dropna(subset=["matrix_row"])
    df_cand["matrix_row"] = df_cand["matrix_row"].astype(int)

    return dict(
        coverage=coverage,
        rssi=rssi,
        candidate_h3=candidate_h3,
        demand_h3=demand_h3,
        rssi_threshold=threshold,
        df_candidates=df_cand,
    )


# ---------------------------------------------------------------------------
# Section 1 — Basic statistics
# ---------------------------------------------------------------------------

def print_basic_stats(data: dict) -> None:
    coverage = data["coverage"]
    n_cand, n_dem = coverage.shape

    cells_covered_per_cand = coverage.sum(axis=1)   # (n_cand,)
    cands_covering_per_dem = coverage.sum(axis=0)   # (n_dem,)

    print("=" * 60)
    print("1. BASIC STATISTICS")
    print("=" * 60)
    print(f"  Candidates   : {n_cand}")
    print(f"  Demand cells : {n_dem}")
    print(f"  RSSI threshold: {data['rssi_threshold']} dBm")
    print()
    print("  Cells covered per candidate:")
    print(f"    mean  = {cells_covered_per_cand.mean():.1f}")
    print(f"    median= {np.median(cells_covered_per_cand):.1f}")
    print(f"    max   = {cells_covered_per_cand.max()}")
    print(f"    min   = {cells_covered_per_cand.min()}")
    print()
    print("  Candidates that can cover each demand cell:")
    print(f"    mean  = {cands_covering_per_dem.mean():.1f}")
    print(f"    zero-coverage demand cells = "
          f"{(cands_covering_per_dem == 0).sum()} "
          f"({(cands_covering_per_dem == 0).mean():.1%})")
    print()


# ---------------------------------------------------------------------------
# Section 2 — Coverage by source (HIFLD vs demand)
# ---------------------------------------------------------------------------

def print_source_coverage(data: dict) -> None:
    coverage     = data["coverage"]
    df_cand      = data["df_candidates"]
    n_dem        = coverage.shape[1]

    print("=" * 60)
    print("2. COVERAGE BY TOWER SOURCE")
    print("=" * 60)

    for source in ["hifld", "demand"]:
        rows = df_cand[df_cand["source"] == source]["matrix_row"].values
        if len(rows) == 0:
            continue
        covered = coverage[rows].any(axis=0).sum()
        print(f"  {source.upper():8s} ({len(rows):4d} towers) → "
              f"{covered:5d} / {n_dem} demand cells covered  "
              f"({covered / n_dem:.1%})")

    # Combined (any tower)
    all_rows = df_cand["matrix_row"].values
    covered_any = coverage[all_rows].any(axis=0).sum()
    print(f"  {'ALL':8s} ({len(all_rows):4d} towers) → "
          f"{covered_any:5d} / {n_dem} demand cells covered  "
          f"({covered_any / n_dem:.1%})")
    print()


# ---------------------------------------------------------------------------
# Section 3 — Top candidates by coverage breadth
# ---------------------------------------------------------------------------

def print_top_candidates(data: dict, top_n: int = 10) -> None:
    coverage  = data["coverage"]
    df_cand   = data["df_candidates"]

    cells_covered = coverage.sum(axis=1)  # (n_cand,)

    # Map matrix row → candidate info
    row_to_info = df_cand.set_index("matrix_row")[["h3_index", "source", "elevation", "demand"]].to_dict("index")

    top_rows = np.argsort(cells_covered)[::-1][:top_n]

    print("=" * 60)
    print(f"3. TOP {top_n} CANDIDATES BY COVERAGE BREADTH")
    print("=" * 60)
    print(f"  {'Rank':>4}  {'Source':>8}  {'Cells':>6}  {'Elev(m)':>8}  {'Demand':>8}  H3 index")
    for rank, row in enumerate(top_rows, 1):
        info = row_to_info.get(row, {})
        elev   = f"{info.get('elevation', float('nan')):.0f}"
        demand = f"{info.get('demand', float('nan')):.3f}" if not pd.isna(info.get("demand", float("nan"))) else "  —  "
        print(f"  {rank:>4}  {info.get('source', '?'):>8}  {cells_covered[row]:>6}  "
              f"{elev:>8}  {demand:>8}  {info.get('h3_index', '?')}")
    print()


# ---------------------------------------------------------------------------
# Section 4 — Hardest demand cells to cover
# ---------------------------------------------------------------------------

def print_hard_demand_cells(data: dict, top_n: int = 10) -> None:
    coverage  = data["coverage"]
    demand_h3 = data["demand_h3"]

    cands_per_cell = coverage.sum(axis=0)  # (n_dem,)
    hard_idx = np.argsort(cands_per_cell)[:top_n]

    print("=" * 60)
    print(f"4. {top_n} HARDEST DEMAND CELLS TO COVER")
    print("=" * 60)
    print(f"  {'Rank':>4}  {'Towers that cover it':>22}  H3 index")
    for rank, j in enumerate(hard_idx, 1):
        print(f"  {rank:>4}  {cands_per_cell[j]:>22}  {demand_h3[j]}")
    print()


# ---------------------------------------------------------------------------
# Section 5 — Greedy K-tower selection
# ---------------------------------------------------------------------------

def greedy_select(coverage: np.ndarray, k: int) -> list:
    """Greedy maximum-coverage: iteratively pick the candidate that covers
    the most currently uncovered demand cells.

    Args:
        coverage: Boolean array (n_cand, n_dem).
        k: Number of towers to select.

    Returns:
        List of selected candidate row indices (length ≤ k).
    """
    uncovered = np.ones(coverage.shape[1], dtype=bool)
    selected  = []

    for _ in range(k):
        # Gain = number of newly covered demand cells per candidate
        gains = (coverage & uncovered).sum(axis=1)
        best  = int(gains.argmax())
        if gains[best] == 0:
            break  # no more demand cells can be covered
        selected.append(best)
        uncovered &= ~coverage[best]

    return selected


def print_greedy_selection(data: dict, k: int) -> None:
    coverage = data["coverage"]
    df_cand  = data["df_candidates"]
    n_dem    = coverage.shape[1]

    row_to_info = df_cand.set_index("matrix_row")[["h3_index", "source", "elevation"]].to_dict("index")

    print("=" * 60)
    print(f"5. GREEDY SELECTION — K = {k} TOWERS")
    print("=" * 60)

    selected = greedy_select(coverage, k)
    covered_mask = coverage[selected].any(axis=0)
    total_covered = covered_mask.sum()

    print(f"  Total demand cells covered : {total_covered} / {n_dem} ({total_covered / n_dem:.1%})")
    print()
    print(f"  {'#':>3}  {'Source':>8}  {'Elev(m)':>8}  H3 index")

    cumulative = np.zeros(n_dem, dtype=bool)
    for step, row in enumerate(selected, 1):
        newly = int((coverage[row] & ~cumulative).sum())
        cumulative |= coverage[row]
        info = row_to_info.get(row, {})
        elev = info.get("elevation", 0) or 0
        print(f"  {step:>3}  {info.get('source', '?'):>8}  "
              f"{elev:>8.0f}  "
              f"{info.get('h3_index', '?')}  (+{newly} cells)")
    print()


# ---------------------------------------------------------------------------
# Section 6 — GA fitness function preview
# ---------------------------------------------------------------------------

def print_ga_fitness_demo(data: dict, k: int) -> None:
    """Show how to evaluate a binary chromosome using the coverage matrix."""
    coverage  = data["coverage"]
    df_cand   = data["df_candidates"]
    n_cand, n_dem = coverage.shape

    print("=" * 60)
    print("6. GA FITNESS FUNCTION PREVIEW")
    print("=" * 60)

    # Load demand weights (population per demand cell)
    demand_h3 = data["demand_h3"]
    demand_weights = np.ones(n_dem, dtype=float)  # uniform if pop not available

    rural_csv = Path("processed_data/rural_cells.csv")
    if rural_csv.exists():
        df_rural = pd.read_csv(rural_csv)
        pop_map  = df_rural.set_index("h3_index")["pop"].to_dict()
        demand_weights = np.array([pop_map.get(h, 1.0) for h in demand_h3])

    # Build a random chromosome with exactly k towers selected
    rng = np.random.default_rng(42)
    chromosome = np.zeros(n_cand, dtype=bool)
    chromosome[rng.choice(n_cand, size=k, replace=False)] = True

    def fitness(chrom: np.ndarray, station_penalty: float = 0.0) -> float:
        """Weighted coverage minus station cost."""
        selected = np.where(chrom)[0]
        covered  = coverage[selected].any(axis=0)
        return float(demand_weights[covered].sum() - station_penalty * chrom.sum())

    # Compare random vs greedy chromosome
    greedy_rows = greedy_select(coverage, k)
    greedy_chrom = np.zeros(n_cand, dtype=bool)
    greedy_chrom[greedy_rows] = True

    random_fit = fitness(chromosome)
    greedy_fit = fitness(greedy_chrom)

    print(f"  Demand weights  : {'population' if rural_csv.exists() else 'uniform (pop not loaded)'}")
    print(f"  K towers        : {k}")
    print()
    print(f"  Random selection fitness : {random_fit:,.1f}")
    print(f"  Greedy selection fitness : {greedy_fit:,.1f}")
    print()
    print("  Fitness function (copy into GA):")
    print("""
    def coverage_fitness(chromosome, coverage, demand_weights, station_penalty=0.0):
        selected = np.where(chromosome)[0]
        if len(selected) == 0:
            return 0.0
        covered = coverage[selected].any(axis=0)
        return demand_weights[covered].sum() - station_penalty * len(selected)
    """)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo: coverage_matrix.npz usage")
    parser.add_argument("--npz",       default="processed_data/coverage_matrix.npz")
    parser.add_argument("--candidates", default="processed_data/rural_candidates.csv")
    parser.add_argument("--k", type=int, default=15, help="Towers to select in greedy demo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"\nLoading {args.npz} ...")
    data = load_coverage(args.npz, args.candidates)
    print(f"Matrix shape: {data['coverage'].shape[0]} candidates × "
          f"{data['coverage'].shape[1]} demand cells\n")

    print_basic_stats(data)
    print_source_coverage(data)
    print_top_candidates(data, top_n=10)
    print_hard_demand_cells(data, top_n=10)
    print_greedy_selection(data, k=args.k)
    print_ga_fitness_demo(data, k=args.k)


if __name__ == "__main__":
    main()
