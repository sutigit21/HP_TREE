# HP-Tree Benchmark Suite

Python-based simulation and benchmarking tools for comparing the HP-Tree against a standard B+ Tree.

## Requirements

- Python 3.10+
- No external dependencies for the main benchmark
- `numpy` for the ML optimizer
- `scikit-learn` (optional) for GradientBoostingRegressor in the optimizer

## Benchmark: HP-Tree vs B+ Tree

```bash
python3 hp_vs_bplus_benchmark.py
```

### What It Does

Runs a comprehensive performance comparison of the HP-Tree and a standard B+ Tree on **1,000,000 records** across **4 data distributions** and **15 query types**.

### Data Distributions

| Distribution | Description |
|:---|:---|
| **Uniform** | All 7 dimensions drawn independently and uniformly at random |
| **Clustered** | Records concentrate around 3 synthetic cluster centres (CA/Laptops, NY/Mice, TX/Keyboards) |
| **Skewed** | 80% of records in a narrow band (CA, Laptop, June 2022); 20% uniform |
| **Sequential** | Monotonically increasing across all dimensions (time-series pattern) |

### Query Types

| # | Query | Description |
|:---:|:---|:---|
| Q1 | Bulk Load | Sort N pairs; construct tree |
| Q2 | Point Lookup | 2,000 random exact-key lookups |
| Q3 | Narrow Range | 1-month window (June 2022) |
| Q4 | Wide Range | 4-year window (2020–2023) |
| Q5 | Dim Filter | year = 2022 (non-prefix dimension) |
| Q6 | Multi-Dim Filter | year = 2022 AND state = CA |
| Q7 | Range Aggregation | SUM(price) over 3-year range |
| Q8 | Full Scan | Retrieve all N records |
| Q9 | Single Inserts | 1,000 inserts into 5K-record tree |
| Q10 | Deletes | 500 random deletes from 1M-record tree |
| Q11 | Hypercube | 3-dim bounding box query |
| Q12 | Group-By Agg | SUM(price) grouped by 15 states, filtered to year=2022 |
| Q13 | Correlated Sub | Per-product: avg(price), then count above avg |
| Q14 | Moving Window | 3-month sliding window SUM(price) × 12 months |
| Q15 | Ad-Hoc Drill | 30 random multi-dim filter queries |

### Composite Key Schema

7 dimensions packed into a 56-bit integer:

| Dimension | Bits | Domain |
|:---|:---:|:---|
| Year | 8 | 2000–2255 |
| Month | 4 | 1–12 |
| Day | 5 | 1–28 |
| State | 5 | 15 US states |
| Product | 5 | 8 product categories |
| Price | 19 | $0.00–$5,242.87 (×100 scale) |
| Version | 10 | 0.00–10.23 (×100 scale) |

### Fairness Measures

The B+ Tree baseline includes:
- **Full delete rebalancing** (borrow from sibling, merge, cascade)
- **`dim_sum` per leaf** for O(1) aggregate shortcuts on range queries
- **Identical key encoding** functions shared by both trees
- **Bitwise correctness verification** on all query results

### Output

The benchmark prints:
1. Per-distribution tree structure statistics (leaves, depth, homogeneity %)
2. Per-query timing comparison with speedup ratios
3. Correctness verification (result match YES/NO)
4. Cross-distribution summary table

### Configuration

Key parameters at the top of `run_benchmark()`:

```python
N = 1000000         # Dataset size
ORDER = 50          # B+ Tree order / HP-Tree max leaf size
BRANCHING = 20      # HP-Tree branching factor
```

HP-Tree also uses `split_power=2.0` and `delta_buffer_cap=256`.

### Runtime

Approximately 30–60 minutes for the full 1M-record benchmark on a modern machine.

---

## ML Optimizer: Optimal Split Power k

```bash
python3 hp_k_optimizer.py
```

### What It Does

Finds the optimal value of the split power exponent `k` in the HP-Tree's stopping criterion `β < 1/N^k` using machine learning.

### Methodology

1. **Training data collection:** Runs benchmarks across a grid of:
   - N values: 50K, 100K, 250K
   - k values: 1.0, 1.25, 1.5, ..., 3.0
   - 4 distributions (Uniform, Clustered, Skewed, Sequential)

2. **Scoring function:**
   ```
   Score = Σ wᵢ · ln(bp_msᵢ / hp_msᵢ)
   ```
   Each HP-Tree win contributes a positive term; each loss contributes negative. Weights prioritise point lookups (1.5×) and inserts (1.3×).

3. **Model:** GradientBoostingRegressor (scikit-learn) or polynomial regression fallback.

4. **Features:** `[log(N), k, log(β_total), coefficient_of_variation, skewness]`

### Output

- Training data summary with per-configuration scores
- Feature importance rankings
- Optimal k predictions for 12 dataset configurations
- Score-vs-k curves per distribution
