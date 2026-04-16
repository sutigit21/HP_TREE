# HP-Tree C++ Implementation

Header-only C++17 implementation of the HP-Tree index structure.

## Requirements

- C++17 compiler (GCC 10+, Clang 12+, AppleClang 13+)
- CMake 3.16+
- POSIX threads

## Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

For debug mode with AddressSanitizer:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
make
```

## Run Tests

```bash
./build/hp_tree_test
```

Executes 15 test suites covering:

| Test | Description |
|:---|:---|
| 1 | Basic CRUD (insert, search, delete, update) |
| 2 | Bulk load + range queries |
| 3 | Predicate search (EQ, BETWEEN, IN) |
| 4 | Forward and reverse iterators |
| 5 | Delta buffer (LSM-style batched writes) |
| 6 | MVCC visibility (snapshot isolation) |
| 7 | Beta threshold strategy comparison (6 strategies) |
| 8 | Statistics collector and cost model |
| 9 | Aggregates (COUNT, SUM, GROUP BY) |
| 10 | Stress test (100K operations) |
| 11 | Write-ahead log (WAL) with recovery |
| 12 | Composite key encode/decode roundtrip |
| 13 | Split and merge under sequential ops |
| 14 | Online index rebuild |
| 15 | Adaptive local beta threshold |

## Architecture

The library is structured as a set of composable headers:

```
include/
├── hp_tree.hpp             # Main HPTree class — public API entry point
├── hp_tree_common.hpp      # Foundation types and utilities
│   ├── CompositeKeySchema    # Multi-dimensional key schema definition
│   ├── DimensionDesc         # Per-dimension encoding (linear/dictionary)
│   ├── CompositeKeyEncoder   # Encode/decode/extract dimensions
│   ├── PredicateSet          # Multi-predicate query specification
│   ├── BetaComputer          # Beta metric computation and thresholds
│   └── HPTreeConfig          # Tree configuration parameters
├── hp_tree_node.hpp        # Node structures
│   ├── LeafNode              # Records + dim_min/dim_max/dim_sum metadata
│   ├── InternalNode          # Routing nodes with separator keys
│   └── Record                # Key + payload + version + tombstone
├── hp_tree_iterator.hpp    # Cursor-based iteration
│   └── HPTreeIterator        # Forward/reverse scan with predicate filtering
├── hp_tree_delta_buffer.hpp# Write-optimised insertion buffer
│   └── DeltaBuffer           # Sorted batch flush to leaf chain
├── hp_tree_buffer_pool.hpp # Storage layer
│   ├── DiskManager           # Page-level I/O
│   └── BufferPool            # LRU page cache with dirty tracking
├── hp_tree_stats.hpp       # Analytics
│   ├── StatisticsCollector   # Tree-wide statistics gathering
│   ├── DimensionHistogram    # Per-dimension equi-depth histograms
│   └── QueryCost             # Cost-based query plan estimation
└── hp_tree_wal.hpp         # Durability
    ├── WalManager            # Append-only log with checkpointing
    └── WalRecord             # Serialisable log record
```

## Usage

```cpp
#include "hp_tree.hpp"
using namespace hptree;

// Define schema
CompositeKeySchema schema = make_default_sales_schema();

// Configure tree
HPTreeConfig cfg;
cfg.max_leaf_size = 50;
cfg.branching_factor = 20;
cfg.beta_strategy = BetaStrategy::ARITHMETIC_MEAN;
cfg.enable_delta_buffer = true;
cfg.delta_buffer_cap = 256;
cfg.enable_wal = false;

// Create tree
HPTree tree(cfg, schema);

// Bulk load records
std::vector<Record> records;
// ... populate records ...
tree.bulk_load(records);

// Point lookup
auto results = tree.search(key);

// Range search
auto range_results = tree.range_search(lo_key, hi_key);

// Predicate search
PredicateSet ps;
ps.predicates.push_back(Predicate::eq(0, encoded_value));
auto filtered = tree.predicate_search(ps);

// Aggregation
auto agg = tree.aggregate_dim(5);  // SUM over dimension 5

// Group by
auto groups = tree.group_by_count(0);  // COUNT(*) GROUP BY dimension 0

// Iterators
for (auto it = tree.begin(); it.valid(); it.next()) {
    Record* rec = it.current();
}

// Statistics and cost estimation
auto stats = tree.statistics();
auto cost = tree.estimate_query_cost(ps);
```

## Beta Strategies

Six threshold strategies are available:

| Strategy | Description |
|:---|:---|
| `FIXED_STRICT` | Fixed threshold β < 0.01 |
| `ARITHMETIC_MEAN` | β < 1/N^k with k from config (default k=2) |
| `MEDIAN` | Median-based adaptive threshold |
| `STDDEV_2X` | Mean + 2σ threshold |
| `STDDEV_6X` | Mean + 6σ threshold |
| `ADAPTIVE_LOCAL` | Per-partition local beta adaptation |

## Thread Safety

- Tree-level shared mutex (`std::shared_mutex`) for read/write locking
- Per-node reader-writer latches
- Atomic counters for node IDs, transaction IDs, and statistics
- MVCC with snapshot-based visibility (`xmin`/`xmax` versioning)
