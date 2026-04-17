# HP-Tree: A Breakthrough Multi-Dimensional Index Structure

The **HP-Tree** (Homogeneity-Partitioned Tree) is a fundamentally new class of tree-structured index that resolves the B+ Tree's 50-year-old unidimensional limitation. By maintaining **hierarchical per-subtree dimensional statistics (DimStats)** at every inner node, the HP-Tree transforms the B+ Tree's unidimensional routing structure into a multi-dimensional pruning hierarchy --- enabling sub-linear filtering on arbitrary dimensions, constant-time range aggregation, and workload-adaptive leaf packing, all within a single dynamically updatable tree.

**Authors:** Sutirtha Chakraborty, Kingshuk Basak

---

## Breakthrough Performance

Benchmarked against `tlx::btree_multimap` v0.6.1 (a production-quality, cache-optimised C++ B+ Tree) on **10 million records** across 4 data distributions and 25 query types:

| Metric | Result |
|:---|:---|
| **HP-Tree wins** | **69 / 100** query-distribution cells |
| **Maximum HP-Tree speedup** | **4,132x** (Range Aggregation, Skewed) |
| **Geometric mean speedup** | **3.16x** across all 100 cells |
| **B+ Tree's best win** | 2.82x (Dense Hyperbox, Skewed) |
| **B+ Tree's median win margin** | Just 1.16x |

### When HP-Tree Wins, It Wins Big

| Query Type | Speedup Range | Why |
|:---|:---|:---|
| Range Aggregation (Q7) | **1,993--4,132x** | O(1) subtree shortcut vs O(N) scan |
| Moving Window (Q14) | **41--234x** | Per-subtree DimStats pruning |
| Dim Filter (Q5) | **1.2--275x** | Hierarchical DimStats exclusion |
| Ad-Hoc Drill (Q15) | **2.3--69x** | Multi-level subtree pruning |
| Group-By Agg (Q12) | **1.1--59x** | DimStats-accelerated grouped sums |
| Point Lookup (Q2) | **2.1--2.6x** | Cache-efficient node layout |
| Inserts (Q9) | **2.0--2.3x** | Sequential-append fast path |
| Deletes (Q10) | **2.0--2.2x** | Lean leaf design |

### When B+ Tree Wins, the Margin Is Thin

| Query Type | B+ Advantage | Explanation |
|:---|:---|:---|
| Full Scan (Q8) | 1.6--1.9x | 43% more leaves due to 70% fill (tunable) |
| Bulk Load (Q1) | 1.1--1.3x | One-time DimStats construction cost |
| Narrow/Wide Range (Q3/Q4) | 1.2--1.4x | Tighter leaf packing |
| Hypercube/Hyperbox (Q11/Q25) | 1.5--2.8x | Bounding-box over-approximation on wide data |
| Correlated Sub (Q13) | 1.01--1.05x | Arithmetic-bound (effective tie) |

**In 24 of the 31 B+ wins, the margin is less than 1.4x.** The B+ Tree never exceeds 1.9x on structurally comparable queries.

---

## Key Innovations

1. **Hierarchical Per-Subtree DimStats** --- Min, max, sum, and count vectors at every inner node. A single DimStats check at level $h$ can prune $B^h$ records. Unlike column-store zone maps (single granularity), HP-Tree DimStats operate at every level of the tree hierarchy.

2. **4-Gate Pruning Cascade** --- Key-range box test → DimStats exclusion → Full-containment fast-path → Adaptive fallback. Ensures the HP-Tree is never asymptotically slower than the B+ Tree.

3. **Adaptive Pruning-Viability Probe** --- Detects when DimStats descent yields no benefit and falls back to cache-friendly leaf-chain walk, avoiding overhead on high-selectivity queries.

4. **Lean Leaf Design** --- No per-leaf range metadata. Min/max derived in O(1) from sorted keys. Saves ~1.4 MB for a 10M-record tree.

5. **Sequential-Append Fast Path** --- Bypasses O(log N) descent for monotonically increasing keys. 2x faster inserts on time-series data.

6. **Workload-Adaptive Leaf Packing** --- Configurable fill factor via `WorkloadProfile`:
   - `ANALYTICAL` → 70% fill (best overall: 69/100 wins)
   - `SCAN_HEAVY` → 95% fill
   - `WRITE_HEAVY` → 70% fill
   - `BALANCED` → 84% fill

---

## Repository Structure

```
HP-TREE/
├── README.md                       # This file
├── CLAUDE.md                       # Build instructions and project notes
├── HP_TREE_Research_Article.md     # Full research article (pandoc-compatible)
├── cpp/                            # C++17 header-only implementation
│   ├── include/
│   │   ├── hp_tree.hpp             # Core HPTree class (~1,050 lines)
│   │   ├── hp_tree_common.hpp      # Config, WorkloadProfile, schema, predicates
│   │   ├── hp_tree_node.hpp        # LeafNode, InnerNode definitions
│   │   ├── hp_tree_iterator.hpp    # Forward iterator with predicate filtering
│   │   └── hp_tree_stats.hpp       # Statistics collector
│   └── CMakeLists.txt
├── benchmark_cpp/                  # C++ benchmark harness
│   ├── src/
│   │   ├── hp_runner.cpp           # HP-Tree benchmark (25 queries)
│   │   └── bplus_runner.cpp        # B+ Tree benchmark (25 queries)
│   ├── python/
│   │   └── generate_datasets.py    # Dataset generator (4 distributions)
│   └── results/                    # Benchmark result JSON files
└── .gitignore
```

## Quick Start

### Build & Run Benchmarks

```bash
cd benchmark_cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Generate datasets (if not present)
python3 python/generate_datasets.py --records 10000000

# Run benchmarks
./build/hp_benchmark
./build/bplus_benchmark
```

Requires C++17. Tested with Apple Clang 17+ and GCC 10+.

### Render Research Article to PDF

```bash
pandoc HP_TREE_Research_Article.md -o HP_TREE_Research_Article.pdf --pdf-engine=xelatex
```

Requires [pandoc](https://pandoc.org/) and a LaTeX distribution (e.g., TeX Live or MacTeX).

---

## C++ API Overview

```cpp
#include "hp_tree.hpp"
using namespace hptree;

HPTreeConfig cfg;
cfg.workload_profile = WorkloadProfile::ANALYTICAL;
cfg.single_threaded  = true;

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

// Range aggregation — O(1) via DimStats
auto agg = tree.aggregate_dim(5);  // SUM(price)

// Iterator
for (auto it = tree.begin(); it.valid(); it.next()) {
    auto* rec = it.current();
}
```

---

## Complexity Summary

| Operation | B+ Tree | HP-Tree (worst) | HP-Tree (best) |
|:---|:---|:---|:---|
| Point Lookup | O(log N) | O(log N) | — |
| Dim Filter | O(N) | O(N) | O(I·D + R) |
| Range Aggregation | O(N) | O(N) | **O(log N)** |
| Group-By Agg | O(N) | O(N) | O(I) |
| Single Insert | O(log N) | O(log N + h·D) | O(h·D) (append) |
| Space overhead | — | — | **0.16% of record storage** |

---

## Composite Key Schema

The benchmark uses a 7-dimension, 56-bit retail sales schema:

| Dimension | Bits | Type | Domain |
|:---|:---:|:---|:---|
| Year | 8 | Linear | 2000–2255 |
| Month | 4 | Linear | 1–12 |
| Day | 5 | Linear | 1–28 |
| State | 5 | Dictionary | 15 US states |
| Product | 5 | Dictionary | 9 categories |
| Price | 19 | Linear (×100) | $0.00–$5,242.87 |
| Version | 10 | Linear (×100) | 0.00–10.23 |

---

## License

This project is provided for academic and research purposes.

## Citation

If you use this work in your research, please cite:

```
Chakraborty, S. and Basak, K. (2026). "HP-Tree: A Homogeneity-Partitioned
Multi-Dimensional Index Structure with Per-Subtree Aggregate Pruning."
```
