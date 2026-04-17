# HP-TREE Project

## Build & Run

```bash
# Build benchmark (from benchmark_cpp/)
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)

# Run full benchmark (10M records, 4 distributions × 25 queries)
./build/hp_benchmark          # HP-Tree
./build/bplus_benchmark       # B+ Tree (tlx::btree_multimap v0.6.1)

# Generate datasets (if missing)
python3 python/generate_datasets.py --records 10000000
```

## Architecture

- **`cpp/include/hp_tree.hpp`** — Core HP-Tree: B+ tree with per-node DimStats (min/max/sum/count per dimension), 4-gate pruning cascade, sequential-append fast path, fused bulk_load with DimStats propagation.
- **`cpp/include/hp_tree_common.hpp`** — Config, enums (`WorkloadProfile`, `PredicateOp`, `AggregateOp`), schema definitions.
- **`benchmark_cpp/src/hp_runner.cpp`** — Benchmark runner: 25 query types using 3 primitives (iterator scan, predicate search, range aggregation).
- **`benchmark_cpp/src/bplus_runner.cpp`** — Equivalent B+ tree benchmark using tlx::btree_multimap.

## Key Features

- **WorkloadProfile enum** — Maps workload types to optimal leaf fill factors:
  - `ANALYTICAL` → 0.7, `SCAN_HEAVY` → 0.95, `WRITE_HEAVY` → 0.7, `BALANCED` → 0.84, `CUSTOM` → 0.84
- **Configurable `bulk_load_fill_factor`** — Controls leaf packing density (0.5–1.0). Set to -1.0 (default) to resolve from WorkloadProfile. This is a build-time parameter; all queries run against the same tree structure.
- **4-gate pruning cascade** — Key-range box test → DimStats exclusion → Full-containment fast-path → Adaptive fallback.
- **Lean leaf nodes** — No per-leaf DimStats; stats live only in inner nodes. Leaves store raw (key, dims[]) pairs.
- **Sequential-append fast path** — Bypasses O(log N) descent when inserting keys ≥ tail leaf max.

## Benchmark Results (10M records, WorkloadProfile::ANALYTICAL)

| Distribution | HP Wins | B+ Wins |
|---|---|---|
| Uniform | 14 | 11 |
| Clustered | 16 | 9 |
| Skewed | 16 | 9 |
| Sequential | 20 | 5 |
| **Total** | **66** | **34** |

## Fill Factor Empirical Results

| Fill | Wins | Notes |
|---|---|---|
| 0.50 | 51 | Catastrophic scan/delete penalty |
| 0.70 | 66 | Best overall (ANALYTICAL profile) |
| 0.84 | 65 | Original default (BALANCED) |
| 0.90 | 67 | Good for scan-heavy workloads |
