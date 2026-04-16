# HP-Tree: Homogeneity-Partitioned Multi-Dimensional Index Structure

A novel tree-based index structure that extends the classical B+ Tree with **adaptive beta-driven splitting**, **per-leaf dimensional metadata**, and a **delta buffer** for write optimisation. The HP-Tree provides sub-linear filtering on arbitrary secondary dimensions without auxiliary indexes, while preserving O(log N) point lookups and efficient range scans.

**Authors:** Sutirtha Chakraborty, Kingshuk Basak

---

## Key Innovations

1. **Beta Homogeneity Metric** — A dimensionless, scale-invariant measure of key-space spread:

   ```
   β(K_min, K_max) = (K_max - K_min)² / (4 · K_min · K_max)
   ```

   Splitting halts when β < 1/N^k (default k=2), producing larger, homogeneous leaves whose dimensional bounds are narrow by construction.

2. **Per-Leaf Dimensional Metadata** — Each leaf maintains `dim_min`, `dim_max`, and `dim_sum` vectors across all D dimensions, functioning as integrated zone maps coupled to the homogeneity-driven partitioning. This enables:
   - O(m)-per-leaf pruning for m-dimensional predicates
   - O(1) whole-leaf inclusion when bounds are fully contained
   - O(1) aggregate contribution via pre-computed sums

3. **Delta Buffer** — An LSM-Tree-inspired write buffer that absorbs single-record insertions and flushes them in sorted batches, amortising per-insert cost to O(log Δ + D).

## Repository Structure

```
HP-TREE/
├── README.md                       # This file
├── HP_TREE_Research_Article.md     # Full research article (pandoc-compatible)
├── cpp/                            # C++17 header-only implementation
│   ├── README.md                   # C++ build and usage instructions
│   ├── CMakeLists.txt              # CMake build configuration
│   ├── include/                    # Header-only library
│   │   ├── hp_tree.hpp             # Main HPTree class (public API)
│   │   ├── hp_tree_common.hpp      # Types, schemas, encoders, predicates
│   │   ├── hp_tree_node.hpp        # LeafNode, InternalNode, BetaComputer
│   │   ├── hp_tree_iterator.hpp    # Forward/reverse iterator with predicate filtering
│   │   ├── hp_tree_delta_buffer.hpp# Delta buffer (LSM-style write path)
│   │   ├── hp_tree_buffer_pool.hpp # Disk manager and buffer pool
│   │   ├── hp_tree_stats.hpp       # Statistics collector, histograms, cost model
│   │   └── hp_tree_wal.hpp         # Write-ahead log with ARIES-style recovery
│   ├── src/
│   │   └── main.cpp                # Comprehensive test suite (15 tests)
│   └── tests/                      # (reserved for future unit tests)
├── benchmark/                      # Python simulation and benchmarking
│   ├── README.md                   # Benchmark usage instructions
│   ├── hp_vs_bplus_benchmark.py    # Full HP-Tree vs B+ Tree benchmark (15 queries × 4 distributions)
│   └── hp_k_optimizer.py           # ML-based optimizer for split power k
└── .gitignore
```

## Quick Start

### C++ (Header-Only Library)

```bash
cd cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
./hp_tree_test
```

Requires C++17. Tested with AppleClang and GCC 10+.

### Python Benchmark

```bash
cd benchmark
python3 hp_vs_bplus_benchmark.py
```

Runs the full 1M-record benchmark across 4 distributions and 15 query types. Takes approximately 30–60 minutes depending on hardware.

### ML Optimizer

```bash
cd benchmark
python3 hp_k_optimizer.py
```

Requires `numpy`. Optionally uses `scikit-learn` for GradientBoostingRegressor; falls back to polynomial regression if unavailable.

### Render Research Article to PDF

```bash
pandoc HP_TREE_Research_Article.md -o HP_TREE_Research_Article.pdf --pdf-engine=xelatex
```

Requires [pandoc](https://pandoc.org/) and a LaTeX distribution (e.g., TeX Live or MacTeX).

## Benchmark Results (N = 1,000,000)

The HP-Tree was evaluated against a standard B+ Tree (with full rebalancing on delete and `dim_sum` per leaf) across 15 query types and 4 data distributions: Uniform, Clustered, Skewed, and Sequential.

| Query Type | Uniform | Clustered | Skewed | Sequential |
|:---|---:|---:|---:|---:|
| Bulk Load | B+ 1.2× | B+ 1.1× | B+ 1.2× | B+ 1.1× |
| Point Lookup | HP 1.4× | HP 1.4× | HP 1.5× | HP 1.4× |
| Dim Filter (year=2022) | HP 306× | HP 99× | HP 38× | HP 427× |
| Group-By Aggregation | HP 964× | HP 3,509× | HP 951× | HP 1,324× |
| Hypercube (3-dim) | HP 1.7× | HP 111× | HP 7.3× | HP 1.6× |
| Ad-Hoc Drill-Down | HP 6.3× | HP 116× | HP 20× | HP 6.4× |

- **53 of 60** query-distribution cells won by HP-Tree
- **Maximum speedup:** 3,509× (Group-By Aggregation, Clustered)
- **Only concession:** 1.1–1.2× slower bulk loading
- **Zero correctness failures** across all 60 cells

## Composite Key Schema

The benchmark uses a 7-dimension, 56-bit retail sales schema:

| Dimension | Bits | Type | Domain |
|:---|:---:|:---|:---|
| Year | 8 | Linear | 2000–2255 |
| Month | 4 | Linear | 1–12 |
| Day | 5 | Linear | 1–28 |
| State | 5 | Dictionary | 15 US states |
| Product | 5 | Dictionary | 8 categories |
| Price | 19 | Linear (×100) | $0.00–$5,242.87 |
| Version | 10 | Linear (×100) | 0.00–10.23 |

## Complexity Summary

| Operation | B+ Tree | HP-Tree (worst) | HP-Tree (best) |
|:---|:---|:---|:---|
| Point Lookup | O(log_B N) | O(Δ + log_B N) | — |
| Dim Filter | O(N) | O(N) | O(L·D + R) |
| Group-By Agg | O(N) | O(N) | O(L) |
| Single Insert | O(log_B N) | — | Amort. O(log Δ + D) |
| Delete | O(log_B N) amort. | O(log_B N) | — |

## C++ API Overview

```cpp
#include "hp_tree.hpp"
using namespace hptree;

// Configure
HPTreeConfig cfg;
cfg.max_leaf_size = 50;
cfg.branching_factor = 20;
cfg.beta_strategy = BetaStrategy::ARITHMETIC_MEAN;
cfg.enable_delta_buffer = true;

// Create tree with schema
auto schema = make_default_sales_schema();
HPTree tree(cfg, schema);

// Bulk load
std::vector<Record> records = /* ... */;
tree.bulk_load(records);

// Point lookup
auto results = tree.search(key);

// Predicate search (year = 2022)
PredicateSet ps;
ps.predicates.push_back(Predicate::eq(0, schema.dimensions[0].encode(2022)));
auto filtered = tree.predicate_search(ps);

// Aggregation
auto agg = tree.aggregate_dim(5);  // SUM(price)

// Iterator
for (auto it = tree.begin(); it.valid(); it.next()) {
    auto* rec = it.current();
}

// Statistics
auto stats = tree.statistics();
```

## License

This project is provided for academic and research purposes.

## Citation

If you use this work in your research, please cite:

```
Chakraborty, S. and Basak, K. (2026). "HP-Tree: A Homogeneity-Partitioned
Multi-Dimensional Index Structure with Adaptive Beta-Driven Splitting."
```
