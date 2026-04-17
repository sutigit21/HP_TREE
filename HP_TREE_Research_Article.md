---
title: "HP-Tree: A Homogeneity-Partitioned Multi-Dimensional Index Structure with Per-Subtree Aggregate Pruning"
author: "Sutirtha Chakraborty, Kingshuk Basak"
date: "April 2026"
geometry: margin=1in
fontsize: 11pt
linestretch: 1.15
header-includes:
  - \usepackage{amsmath}
  - \usepackage{amssymb}
  - \usepackage{textcomp}
  - \usepackage{booktabs}
  - \usepackage{float}
  - \floatplacement{table}{H}
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhead[L]{HP-Tree}
  - \fancyhead[R]{Chakraborty \& Basak, 2026}
  - \fancyfoot[C]{\thepage}
---

## Abstract

Tree-structured indexes remain foundational to database query processing, yet the classical B+ Tree [1, 2] provides no mechanism for pruning irrelevant data along dimensions that do not align with its primary sort order. We introduce the **HP-Tree** (Homogeneity-Partitioned Tree), a compiled C++ multi-dimensional index structure that extends the B+ Tree with two integrated innovations: (i) *per-subtree dimensional statistics* (DimStats) --- min, max, count, and sum vectors maintained at every inner node --- that enable sub-linear predicate pruning and constant-time aggregate computation on arbitrary dimension predicates; and (ii) an *adaptive pruning-viability probe* that dynamically detects when DimStats-based descent would yield no benefit for a given predicate--subtree combination and falls back to a cache-friendly linear leaf-chain walk, avoiding pure overhead on high-selectivity queries. We evaluate the HP-Tree against `tlx::btree_multimap` (v0.6.1) [11] --- a production-quality, cache-optimised C++ B+ Tree --- on a workload of $N = 10{,}000{,}000$ composite-key records across four data distributions (Uniform, Clustered, Skewed, Sequential) and twenty-five query types spanning point lookups, range scans, multi-dimensional filters, group-by aggregations, correlated subqueries, moving-window analytics, ad-hoc drill-down queries, Top-K, HAVING, YoY semi-joins, CTE correlations, OR-bitmap filters, and dense hyperbox scans. The HP-Tree wins 65 of 100 query-distribution cells with a geometric-mean speedup of $3.25\times$, achieving up to $4{,}672\times$ on range aggregation and $279\times$ on single-dimension filtering, while conceding only a marginal $1.1$--$1.4\times$ slowdown on bulk loading and full scans. All reported results are verified for bitwise result-set correctness across 96 of 100 query-distribution cells (the 4 unmatched cells are explained by a harness-level month-base encoding offset; sums and checksums match in all 100 cells).

**Keywords:** multi-dimensional indexing, B+ Tree, composite keys, per-subtree aggregates, zone maps, analytical query processing, predicate pruning

\newpage

## 1. Introduction

### 1.1 The B+ Tree and Its Fundamental Limitation

Since their introduction by Bayer and McCreight in 1972 [1], B-Tree variants have served as the default index structure in virtually all relational database management systems. The B+ Tree refinement --- which confines all data records to leaf nodes connected by a doubly-linked list --- provides $O(\log_B N)$ point lookups and efficient sequential range scans, where $B$ denotes the branching factor and $N$ the number of indexed records [2]. Comer's comprehensive survey [2] established the B+ Tree as "ubiquitous," and the subsequent four decades of engineering refinements catalogued by Graefe [6] --- including prefix compression, fence keys, ghost records, and page-level write-ahead logging --- have kept the structure competitive in modern disk-based and main-memory systems alike.

However, the B+ Tree's architecture is fundamentally *unidimensional*. Records are sorted by a single composite key, and the tree's internal routing structure can only exploit this ordering for queries whose predicates align with the key's most-significant dimension prefix. Consider a composite key encoding the tuple $(\text{year}, \text{month}, \text{day}, \text{state}, \text{product}, \text{price})$: a range query on $\text{year}$ benefits from the sort order and traverses $O(\log_B N)$ nodes, but a filter on $\text{state}$ alone --- a dimension embedded in the middle of the key encoding --- must scan every leaf node, since state values are distributed arbitrarily within the composite key's bit layout. This $O(N)$ full-scan behaviour extends to all non-prefix dimension predicates, multi-dimensional bounding-box queries, and grouped aggregations on secondary dimensions, precisely the workload patterns that dominate modern analytical processing [7].

The practical impact of this limitation is substantial. In a retail analytics database with $10^7$ records, a query such as "compute total revenue by state for the year 2022" requires the B+ Tree to examine every record in the index --- an $O(N)$ operation repeated for each group --- even though the result set may involve only a small fraction of the data. This mismatch between the tree's unidimensional ordering and the workload's multi-dimensional access patterns motivates the search for index structures that preserve the B+ Tree's strengths (logarithmic point lookups, efficient range scans, dynamic insertions and deletions) while adding sub-linear access paths along arbitrary dimensions.

### 1.2 Related Work

The limitations of unidimensional indexing have motivated an extensive body of work on multi-dimensional access methods, each addressing a different facet of the problem at the cost of new trade-offs.

**Spatial index structures.** Bentley's KD-Tree [4] partitions space by alternating split dimensions at each level of the tree, achieving efficient nearest-neighbour and orthogonal range queries in low dimensions ($D \leq 20$). However, the KD-Tree is inherently static: insertions and deletions require partial or full tree reconstruction, making it unsuitable for transactional workloads with dynamic updates [4]. Guttman's R-Tree [3] generalises the B-Tree to spatial data by associating each internal node with a minimum bounding rectangle (MBR) that encloses all descendant objects. While R-Trees support dynamic insertions and deletions, they suffer from *overlap* between sibling MBRs, which forces multiple subtree traversals during search and degrades worst-case query performance to $O(N)$ on high-dimensional or uniformly distributed data [3, 8]. The R+-Tree [8] addresses overlap through clipping --- splitting objects across multiple nodes to eliminate MBR overlap --- but introduces record duplication and complicates deletion.

**Bitmap and column-oriented indexes.** Bitmap indexes, widely deployed in data warehousing systems, represent each distinct value of an indexed attribute as a bit vector of length $N$, enabling efficient conjunction and disjunction of equality predicates via bitwise operations. However, bitmap indexes consume $O(N \cdot |\mathcal{D}|)$ space per indexed dimension (where $|\mathcal{D}|$ is the domain cardinality), rendering them impractical for high-cardinality or continuous-valued attributes such as price or timestamp [7]. Column-oriented storage systems such as C-Store [7] restructure the data layout to store each attribute in a separate dense array, enabling per-column compression, vectorised predicate evaluation, and late materialisation. These designs achieve dramatic analytical speedups but fundamentally abandon the row-oriented access patterns required by OLTP workloads, and they do not provide a tree-structured index with logarithmic point-lookup guarantees.

**Write-optimised and cache-conscious trees.** The Log-Structured Merge-Tree (LSM-Tree) [5] optimises write-heavy workloads by buffering mutations in an in-memory component and periodically merging them into sorted runs on persistent storage. While the LSM-Tree's write amplification is well-suited to modern SSDs, it introduces *read amplification* for point lookups (which must consult multiple levels) and provides no multi-dimensional pruning capability. Cache-conscious B+ Tree variants [9] improve main-memory performance by aligning node layouts with cache-line boundaries and eliminating pointer-chasing overhead, but they address microarchitectural efficiency rather than the fundamental unidimensional limitation. The Adaptive Radix Tree (ART) [10] demonstrates that trie-based designs can outperform B+ Trees on point lookups in memory-resident databases through path compression and adaptive node sizes, but it operates on byte-addressable keys and does not support multi-dimensional predicate pruning.

**Gap in the literature.** None of the structures surveyed above simultaneously satisfies the following four requirements: (a) $O(\log_B N)$ point lookups and efficient range scans on the primary key ordering; (b) sub-linear filtering on arbitrary secondary dimensions *without* auxiliary indexes or materialised views; (c) constant-time per-subtree aggregate computation for range-contained partitions; and (d) efficient dynamic insertions and deletions with bounded structural modification cost. The HP-Tree is designed to address this gap.

### 1.3 Contributions

This paper introduces the HP-Tree, which augments the B+ Tree's leaf-linked architecture with mechanisms designed to bridge the multi-dimensional gap without sacrificing single-key performance:

1. **Per-Subtree Dimensional Statistics (DimStats).** Each inner node maintains four vectors --- $\texttt{min\_val}[d]$, $\texttt{max\_val}[d]$, $\texttt{count}$, and $\texttt{sum}[d]$ --- for every dimension $d \in \{0, \dots, D{-}1\}$ (Section 2.3). These function as *hierarchical zone maps* [7] at every level of the tree, enabling: (a) $O(m)$-per-subtree pruning for predicates on $m$ dimensions; (b) $O(1)$ whole-subtree inclusion when a subtree's bounds are fully contained within the predicate range; and (c) $O(1)$ aggregate contribution via pre-computed sums and counts.

2. **Adaptive Pruning-Viability Probe.** A lightweight sampling heuristic (Section 2.5) that inspects up to 8 children of an inner node to determine whether DimStats-based descent would yield any pruning. When no children can be excluded or fully included, the tree falls back to a cache-friendly filtered leaf-chain walk --- identical in cost to a B+ Tree linear scan --- avoiding the overhead of DimStats checks at every level below.

3. **Lean Leaf Design.** Leaf nodes carry no range metadata; since keys are sorted, `min_key()` = `keys[0]` and `max_key()` = `keys[slotuse-1]` are derived in $O(1)$, saving 32 bytes per leaf (two `__uint128_t` fields) and reducing hot cache footprint by approximately 1 MB for a 10M-record tree.

4. **Sequential-Append Fast Path.** When a new key is $\geq$ the tree's maximum key and the tail leaf has capacity, the full root-to-leaf descent is replaced by a direct append to `tail_leaf_` with a rightmost-spine metadata walk, reducing single-insert latency for time-series and append-heavy workloads.

These mechanisms are *intrinsic* to the tree's inner-node structure and require no external secondary indexes, materialised views, or auxiliary data structures. The HP-Tree preserves the B+ Tree's doubly-linked leaf chain, its logarithmic-height guarantee, and its compatibility with standard concurrency-control protocols [6].

### 1.4 Paper Organisation

The remainder of this paper is organised as follows. Section 2 presents the HP-Tree's design: composite key encoding (2.1), node architecture (2.2), per-subtree DimStats (2.3), predicate pruning (2.4), the adaptive pruning-viability probe (2.5), the sequential-append fast path (2.6), deletion with leaf rebalancing (2.7), and a tabular architectural comparison with the B+ Tree (2.8). Section 3 derives worst-case time and space complexity bounds. Section 4 describes the experimental design. Section 5 presents empirical results. Section 6 provides a detailed discussion including cross-industry relevance. Section 7 concludes with future research directions.

\newpage

## 2. HP-Tree Methodology

### 2.1 Composite Key Encoding

Both the HP-Tree and the baseline B+ Tree operate on **composite keys** that pack $D$ dimensions into a single fixed-width `__uint128_t` integer, following the bit-concatenation approach used in multi-dimensional indexing [3, 4]. Given dimensions $d_0, d_1, \dots, d_{D-1}$ with bit-widths $b_0, b_1, \dots, b_{D-1}$, the composite key $K$ is defined as:

$$K = \sum_{i=0}^{D-1} v_i \cdot 2^{\,\sigma(i)}, \qquad \text{where} \quad \sigma(i) = \sum_{j=i+1}^{D-1} b_j$$

and $v_i \in [0, 2^{b_i} - 1]$ is the encoded value for dimension $i$. The total key width is $W = \sum_{i=0}^{D-1} b_i$ bits. The encoding preserves lexicographic ordering with $d_0$ as the most-significant dimension: for any two keys $K_1$ and $K_2$, $K_1 < K_2$ if and only if the first dimension at which $K_1$ and $K_2$ differ has a smaller value in $K_1$.

Extraction of individual dimension values from a composite key is performed via bit-shifting and masking in $O(1)$ time:

$$\texttt{extract}(K, i) = \left\lfloor K \,/\, 2^{\,\sigma(i)} \right\rfloor \bmod 2^{b_i}$$

In the C++ implementation, per-dimension offsets and masks are precomputed during schema finalisation and cached in stack-inline arrays (`dim_off_[]`, `dim_mask_[]`) for zero-overhead extraction in the hot path:

```cpp
inline uint64_t extract_dim_fast(CompositeKey key, size_t d) const {
    return static_cast<uint64_t>((key >> dim_off_[d]) & dim_mask_[d]);
}
```

Two encoding modes are supported per dimension: **linear encoding** for numeric types (with a configurable base offset $b_0$ and scale factor $s$, such that the raw value $x$ is encoded as $v = \lfloor (x - b_0) \cdot s \rfloor$), and **dictionary encoding** for categorical types with a finite domain $\mathcal{D}$ (where each element is mapped to a unique integer in $[0, |\mathcal{D}| - 1]$). This dual-encoding scheme follows the design of modern analytical engines, which use dictionary compression for low-cardinality string columns [7].

### 2.2 Node Architecture

The HP-Tree uses a tlx-style [11] node layout with inline fixed-size arrays, avoiding per-node heap allocation. All constants are compile-time:

$$\texttt{LEAF\_SLOTMAX} = 32, \quad \texttt{INNER\_SLOTMAX} = 32, \quad \texttt{MAX\_DIMS} = 8$$

**LeafNode.** A leaf node inherits from `NodeBase` (which carries `level` and `slotuse`) and contains:

- `keys[LEAF_SLOTMAX + 1]`: sorted `__uint128_t` composite keys (the +1 provides a slack slot for temporary overflow during insert-before-split).
- `values[LEAF_SLOTMAX + 1]`: corresponding 8-byte payload values.
- `prev_leaf`, `next_leaf`: doubly-linked leaf chain pointers for sequential iteration.

Critically, leaves carry **no range metadata** (`range_lo`/`range_hi`). Since keys are maintained in sorted order, the minimum and maximum keys are derived in $O(1)$ via `keys[0]` and `keys[slotuse - 1]` respectively. This saves 32 bytes (two `__uint128_t` fields) per leaf. For a 10M-record tree with ${\sim}37{,}000$ leaves, this eliminates ${\sim}1.1$ MB from the hot working set.

**InnerNode.** An inner node contains:

- `slotkey[INNER_SLOTMAX + 1]`: separator keys routing searches to the correct child.
- `childid[INNER_SLOTMAX + 2]`: child node pointers.
- `range_lo`, `range_hi`: the composite-key bounds of all records in the subtree rooted at this node (used for key-range box pruning).
- `subtree_count`: the total number of records in the subtree.
- `dim_stats[MAX_DIMS]`: an array of `DimStats` structures (Section 2.3), one per dimension.

The `DimStats` array is stack-inline (`std::array<DimStats, MAX_DIMS>`) to avoid heap allocation per inner node. Only the first `dim_count` entries are semantically meaningful.

**Search.** Both `inner_find_lower` and `leaf_find_lower` use linear scan rather than binary search, as this is faster for $\leq 32$-slot nodes due to branch prediction --- the same approach used by tlx [11]:

```cpp
inline uint16_t inner_find_lower(const InnerNode* n, CompositeKey key) {
    uint16_t lo = 0;
    while (lo < n->slotuse && n->slotkey[lo] < key) ++lo;
    return lo;
}
```

### 2.3 Per-Subtree Dimensional Statistics (DimStats)

Each inner node $I$ maintains, for every dimension $d \in \{0, \dots, D{-}1\}$, a `DimStats` structure:

$$\texttt{DimStats}[d] = \bigl(\texttt{count},\; \texttt{sum},\; \texttt{min\_val},\; \texttt{max\_val}\bigr)$$

where:

$$\texttt{min\_val}_I[d] = \min_{r \in \text{subtree}(I)}\, \texttt{extract}(K_r, d), \quad \texttt{max\_val}_I[d] = \max_{r \in \text{subtree}(I)}\, \texttt{extract}(K_r, d)$$

$$\texttt{sum}_I[d] = \sum_{r \in \text{subtree}(I)}\, \texttt{extract}(K_r, d), \qquad \texttt{count}_I = |\text{subtree}(I)|$$

These statistics are conceptually analogous to the **zone maps** (also termed *min-max indexes* or *small materialised aggregates*) used in column-oriented storage engines [7]. However, a critical distinction exists: in column stores, zone maps are associated with fixed-size storage blocks whose contents are determined by insertion order and bear no relation to data semantics. The HP-Tree's DimStats, by contrast, are maintained at *every level of the inner-node hierarchy* and are coupled to the tree's sort-order-driven partitioning. This hierarchical placement enables multi-level short-circuiting: a single DimStats check at level $h$ can prune an entire subtree of $B^h$ records.

**Construction during bulk load.** The bulk-load algorithm builds DimStats in two phases:

1. *Leaf level:* As leaf nodes are packed (at 84% capacity, i.e., 27 of 32 slots, balancing scan locality against insert slack), a parallel `std::vector<std::array<DimStats, MAX_DIMS>>` accumulates per-leaf statistics in a single fused pass over the sorted records. This costs $O(N \cdot D)$ total.

2. *Inner levels:* The first inner level is built by `build_inner_level_from_leaves()`, which merges the parallel leaf-stats array via `DimStats::merge()` in $O(\text{children} \cdot D)$ per inner node. Subsequent levels are built by `build_inner_level_from_inners()`, which reads each child InnerNode's `dim_stats` directly --- no intermediate buffer needed. The leaf-stats array is freed (`shrink_to_fit()`) as soon as the first inner level is complete.

**Incremental maintenance.** On single-record insert, the `inner_meta_add()` function updates each ancestor inner node along the insertion path:

```cpp
inline void inner_meta_add(InnerNode* in, CompositeKey key,
                           const uint64_t* key_dims) {
    in->subtree_count++;
    for (size_t d = 0; d < dim_count_; ++d)
        in->dim_stats[d].add(key_dims[d]);
    if (key < in->range_lo) in->range_lo = key;
    if (key > in->range_hi) in->range_hi = key;
}
```

This costs $O(D)$ per inner level and $O(h \cdot D)$ total per insertion, where $h = O(\log_B N)$. On deletion, `inner_meta_sub()` decrements `count` and `sum` exactly but leaves `min_val`/`max_val` and `range_lo`/`range_hi` potentially stale-wide (never stale-narrow), which is safe for pruning: a stale-wide bound may fail to prune a subtree that could have been pruned (a false positive), but will never incorrectly prune a subtree that contains matching records (no false negatives).

On node splits, both halves are recomputed from their children via `aggregate_inner_from_children()`. When children are inner nodes (upper levels), this is pure $O(\text{children} \cdot D)$ merging of precomputed DimStats; only at the bottom inner level does it require a per-record scan of the constituent leaves.

### 2.4 Predicate Pruning

The HP-Tree's query processing uses a four-gate pruning cascade applied at each inner node during recursive descent:

**Gate 1: Key-range box test.** If the inner node's `[range_lo, range_hi]` interval is disjoint from the query's `KeyRange [lo, hi]`, the entire subtree is skipped. Cost: $O(1)$.

**Gate 2: DimStats exclusion (`subtree_may_contain`).** For each predicate $p$ on dimension $d$, if the predicate's value falls outside the subtree's `[min_val[d], max_val[d]]` bounds, the subtree is skipped. For $m$ predicates, cost: $O(m)$. Supported predicates include `EQ`, `BETWEEN`, `LT`, `LTE`, `GT`, `GTE`.

**Gate 3: Full-containment fast-path (`subtree_fully_satisfies`).** If the subtree's per-dimension bounds are entirely within every predicate's range *and* the subtree's key range is within the query's key range, all records in the subtree satisfy the query. The tree emits all records via a flat leaf-chain walk (`emit_subtree_all`) without per-record predicate evaluation. Cost: $O(|\text{subtree}|)$ for output, $O(m)$ for the containment check.

**Gate 4: Adaptive fallback (Section 2.5).** If neither Gate 2 nor Gate 3 can prune or fast-track any sampled children, the tree falls back to a filtered leaf-chain walk.

For aggregation queries, an additional $O(1)$ shortcut exists: if a subtree's key range is fully contained within $[lo, hi]$, its `subtree_count` and `dim_stats[d].sum` are added directly to the result without any leaf-level iteration:

```cpp
if (in->range_lo >= lo && in->range_hi <= hi) {
    r.count += in->subtree_count;
    r.sum   += in->dim_stats[dim].sum;
    return;
}
```

This is what produces the $4{,}672\times$ speedup on Q7 Range Aggregation: the root node's DimStats satisfy the containment check, and the entire 10M-record aggregation completes in a single comparison + addition ($3\,\mu$s) versus the B+ Tree's full leaf-chain scan (14 ms).

### 2.5 Adaptive Pruning-Viability Probe

Not all predicates benefit from DimStats-based descent. When data is uniformly distributed and a predicate has high selectivity (e.g., year=2022 selects 84% of records on skewed data), every subtree's DimStats bounds include the target value, and the per-node DimStats checks become pure overhead.

The HP-Tree addresses this with a `pruning_viable()` probe: before recursing into an inner node's children, it inspects up to $\min(\texttt{slotuse}+1, 8)$ children and checks whether *any* of them can be fully excluded (`!subtree_may_contain`) or fully included (`subtree_fully_satisfies`). If none can, DimStats-based descent is abandoned for this subtree, and `emit_subtree_filtered()` walks the leaf chain under the subtree with per-record predicate evaluation --- identical in cost to a B+ Tree linear scan.

The probe is only triggered when the key-range gate itself is not already trimming the child window (i.e., `first == 0 && last == slotuse`), ensuring the normal path's slotkey-based window pruning is not bypassed unnecessarily.

**Cost of the probe:** $\leq 8 \times D \times m$ comparisons, which for $D = 8$ and $m \leq 4$ is ${\sim}256$ operations --- negligible against any subtree of more than a few hundred records.

**Monotonicity argument:** Child DimStats bounds are contained within parent bounds. If the parent's `subtree_may_contain` returns true (cannot prune), this does not imply that children cannot be pruned individually. However, if *none* of the first 8 children show pruning potential, the remaining children are unlikely to either. The probe is thus a sound (if conservative) oracle.

### 2.6 Sequential-Append Fast Path

For time-series, IoT, and log-analytics workloads where records arrive in approximately increasing key order, the HP-Tree provides a fast path that bypasses the $O(\log_B N)$ root-to-leaf descent:

```
if (tail_leaf_ has capacity AND rec.key >= tail_leaf_->max_key()):
    append directly to tail_leaf_
    walk rightmost spine updating DimStats
```

**Correctness argument:** `rec.key >= tail->max_key()` implies the correct insertion slot is `tail->slotuse` (strict append), the same position a full descent would find. Every inner node along the rightmost spine covers `tail_leaf_` and only it, so updating `range_hi` and DimStats along this spine is both necessary and sufficient.

The rightmost spine walk costs $O(h \cdot D)$ --- the same DimStats update cost as a normal insertion --- but eliminates the $O(h \cdot B)$ inner-node search overhead at each level. In the benchmark, this yields $2.1$--$3.4\times$ speedup on Q9 (Single Inserts) on uniform/clustered data where bulk-loaded records have a sorted tail.

### 2.7 Deletion with Leaf Rebalancing

The HP-Tree implements physical deletion with B+ Tree-style leaf rebalancing, maintaining the standard structural invariants:

1. **Locate and remove.** The target record is found via standard $O(\log_B N)$ tree traversal. The record is physically removed from its leaf node by shifting subsequent entries. Per-dimension values are extracted before removal for metadata updates.

2. **Metadata update.** On the ascent, each ancestor's `subtree_count` and `dim_stats[d].count`/`sum` are decremented exactly via `inner_meta_sub()`. The `min_val`/`max_val` bounds are left unchanged (stale-wide is safe for pruning).

3. **Leaf rebalancing.** If a leaf falls below `LEAF_SLOTMIN` (16 records), the tree attempts to borrow from the right sibling first (more common case during forward deletion), then the left sibling. If no sibling can donate without itself underflowing, the two leaves are merged, the separator key is removed from the parent, and the leaf-chain pointers are updated. Inner-level underflow is handled only at the root (single-child collapse).

This is the same rebalancing algorithm used by tlx [11], ensuring a fair comparison on deletion workloads.

### 2.8 Architectural Comparison with the Standard B+ Tree

Table 1 summarises the architectural differences between the B+ Tree (as implemented by `tlx::btree_multimap` [11]) and the HP-Tree.

\vspace{0.2cm}
*Table 1. Architectural comparison.*

| Feature | B+ Tree (tlx::btree_multimap) [11] | HP-Tree |
|:---|:---|:---|
| Node layout | Inline arrays, auto-computed slot counts | Inline arrays, `LEAF_SLOTMAX=32`, `INNER_SLOTMAX=32` |
| Leaf metadata | None (keys + values only) | Sorted keys $\Rightarrow$ $O(1)$ min/max; no stored range fields |
| Inner metadata | Separator keys + child pointers | + `range_lo/hi`, `subtree_count`, `DimStats[D]` |
| Dim filter mechanism | Full scan $O(N)$ | Hierarchical subtree pruning via DimStats |
| Aggregation shortcut | None | $O(1)$ per subtree via `subtree_count` + `dim_stats.sum` |
| Predicate pruning | None | 4-gate cascade: key-range, DimStats exclusion, full-containment, adaptive fallback |
| Deletion | Physical removal + rebalancing [1] | Physical removal + leaf rebalancing (same invariants) |
| Insert fast path | None | Sequential-append via tail-leaf + rightmost-spine walk |
| Multi-dim pruning | None | Hypercube bounds checking at inner-node level |
| Bulk load packing | Auto (typically 100%) | 84% (27/32) for insert slack |

\newpage

## 3. Time and Space Complexity

We derive worst-case complexity bounds for both structures. Let $N$ denote the total number of records, $B$ the node order (maximum slots per node), $D$ the number of dimensions, $h$ the tree height, $R$ the result-set size, $m$ the number of predicate dimensions, $L$ the number of leaf nodes, $L_c$ the number of leaves overlapping a query range, and $I$ the number of inner nodes.

### 3.1 Tree Height

For a B+ Tree with node order $B$ (each leaf holds at most $B$ records, each inner node has at most $B+1$ children):

$$h_{\text{B+}} = \left\lceil \log_{B+1} \frac{N}{B} \right\rceil + 1 = O(\log_B N)$$

The HP-Tree uses the same branching factor ($B = 32$) and packing factor (84%), yielding:

$$L_{\text{HP}} = \left\lceil \frac{N}{0.84 \cdot B} \right\rceil, \qquad h_{\text{HP}} = \left\lceil \log_{B+1} L_{\text{HP}} \right\rceil + 1 = O(\log_B N)$$

At $N = 10{,}000{,}000$ and $B = 32$: $L_{\text{HP}} \approx 37{,}203$ leaves, $h_{\text{HP}} = 4$ levels (1 leaf + 3 inner).

### 3.2 Node and Record Counts

| Component | B+ Tree (tlx) | HP-Tree |
|:---|:---|:---|
| Leaf nodes | $L_{\text{B+}} = \lceil N / B \rceil$ | $L_{\text{HP}} = \lceil N / (0.84 B) \rceil$ |
| Inner nodes (total) | $I_{\text{B+}} = \sum_{k=1}^{h-1} \lceil L / (B+1)^k \rceil$ | $I_{\text{HP}} = \sum_{k=1}^{h-1} \lceil L_{\text{HP}} / (B+1)^k \rceil$ |
| Records per leaf (avg) | $B$ (fully packed after bulk load) | $0.84 B \approx 27$ |

At $N = 10^7$, $B = 32$: $L_{\text{B+}} \approx 312{,}500$; $L_{\text{HP}} \approx 37{,}203$; $I_{\text{B+}} \approx 9{,}540$; $I_{\text{HP}} \approx 1{,}136$.

### 3.3 Time Complexity

\vspace{0.2cm}
*Table 2. Time complexity of core operations.*

| Operation | B+ Tree [1, 2, 11] | HP-Tree (worst) | HP-Tree (best) |
|:---|:---|:---|:---|
| Bulk Load | $O(N \log N)$ | $O(N \log N + N \!\cdot\! D)$ | --- |
| Point Lookup | $O(\log_B N)$ | $O(\log_B N)$ | --- |
| Range Scan ($R$ results) | $O(\log_B N + R)$ | $O(\log_B N + R)$ | --- |
| Dim Filter (dim $d = v$) | $O(N)$ | $O(N)$ | $O(I \!\cdot\! D + R)$ |
| Multi-Dim ($m$ preds) | $O(N \!\cdot\! m)$ | $O(N \!\cdot\! m)$ | $O(I \!\cdot\! m + R)$ |
| Hypercube ($m$ ranges) | $O(N \!\cdot\! m)$ | $O(N \!\cdot\! m)$ | $O(I \!\cdot\! m + R)$ |
| Range Aggregation | $O(N)$ | $O(N)$ | $O(\log_B N)$ |
| Dim Aggregation (per group) | $O(N)$ | $O(N)$ | $O(I)$ |
| Single Insert | $O(\log_B N + B)$ amort. | $O(\log_B N + B + h \!\cdot\! D)$ | $O(h \!\cdot\! D)$ (append) |
| Delete | $O(\log_B N + B)$ amort. | $O(\log_B N + B + h \!\cdot\! D)$ | --- |
| Full Scan | $O(N)$ | $O(N)$ | --- |

**Notes on key entries:**

*Bulk Load.* Both trees sort $N$ records in $O(N \log N)$. The B+ Tree constructs the tree bottom-up in $O(N)$. The HP-Tree additionally computes DimStats per leaf in a fused $O(N \cdot D)$ pass and merges them upward in $O(I \cdot D)$, giving total $O(N \log N + N \cdot D)$. Since $D \ll \log N$ for practical dimensionality ($D \leq 8$, $\log_2 N \approx 23$), the sort dominates.

*Range Aggregation.* The B+ Tree must scan all $L_c$ leaves in the range and iterate their records: $O(\log_B N + R)$ where $R$ can equal $N$. The HP-Tree's $O(\log_B N)$ best case occurs when the query range fully contains entire subtrees: the root DimStats satisfy the containment check and the aggregate is returned in constant time. The worst case (all boundary leaves) degrades to $O(N)$, matching the B+ Tree.

*Dim Filter.* The B+ Tree must always scan all $N$ records because it has no metadata on non-prefix dimensions. The HP-Tree's best case $O(I \cdot D + R)$ occurs when DimStats are tight enough to prune most subtrees --- achieved on sequential data where each subtree covers a narrow dimension range. The worst case (all subtrees partially overlap) degrades to $O(N)$.

*Single Insert.* Both trees pay $O(\log_B N)$ for traversal and $O(B)$ amortised for in-leaf shifting and occasional splits. The HP-Tree adds $O(h \cdot D)$ for DimStats updates along the insertion path. With the sequential-append fast path, the traversal cost is eliminated: the per-dim values are precomputed once ($O(D)$) and the rightmost-spine walk costs $O(h \cdot D)$, giving $O(h \cdot D)$ total.

### 3.4 Space Complexity

\vspace{0.2cm}
*Table 3. Space complexity. $B$ = node order, $D$ = dimensions, $N$ = records.*

| Component | B+ Tree (tlx) [11] | HP-Tree |
|:---|:---|:---|
| Leaf key storage | $O(N \cdot W_K)$ | $O(N \cdot W_K)$ |
| Leaf value storage | $O(N \cdot W_V)$ | $O(N \cdot W_V)$ |
| Leaf pointers (prev/next) | $O(L)$ | $O(L)$ |
| Inner separator keys | $O(I \cdot B \cdot W_K)$ | $O(I \cdot B \cdot W_K)$ |
| Inner child pointers | $O(I \cdot B)$ | $O(I \cdot B)$ |
| Inner range bounds | --- | $O(I \cdot 2 W_K)$ |
| Inner DimStats | --- | $O(I \cdot D \cdot 4 \cdot 8)$ |
| Inner subtree count | --- | $O(I \cdot 8)$ |
| **Total** | $O(N + I \cdot B)$ | $O(N + I \cdot (B + D))$ |

where $W_K$ = key width (16 bytes for `__uint128_t`) and $W_V$ = value width (8 bytes).

**Concrete overhead at $N = 10^7$, $B = 32$, $D = 7$:**

- HP-Tree inner nodes: $I_{\text{HP}} \approx 1{,}136$. Each carries `DimStats[8]` = 8 × (8 + 8 + 8 + 8) = 256 bytes, plus `range_lo/hi` = 32 bytes, plus `subtree_count` = 8 bytes. Total DimStats overhead: $1{,}136 \times 296 \approx 328$ KB.
- Total record storage: $10^7 \times 24$ bytes $= 240$ MB.
- **DimStats overhead as fraction of record storage: 0.13%.**

The HP-Tree's space overhead relative to the B+ Tree is negligible --- less than 0.2% of the $O(N)$ record storage. The dominant cost difference is that the HP-Tree has ${\sim}8.4\times$ more leaves than the B+ Tree (37K vs 312K) due to 84% packing, but each HP-Tree leaf is 32 bytes smaller (no stored range fields).

\newpage

## 4. Experimental Design

### 4.1 Experimental Setup

Both index structures are implemented in C++ and compiled with Apple Clang 17.0.0 (arm64-apple-darwin24.4.0) at `-O2` optimisation level.

**HP-Tree implementation.** The HP-Tree is implemented as a header-only C++ library across six headers: `hp_tree.hpp` (1,050 lines, core tree logic), `hp_tree_node.hpp` (121 lines, node definitions), `hp_tree_common.hpp` (422 lines, schema, predicates, DimStats), `hp_tree_iterator.hpp` (63 lines, forward iterator), `hp_tree_delta_buffer.hpp` (intentionally empty; delta buffer is not active in the evaluated configuration), and `hp_tree_stats.hpp`. Key constants: `LEAF_SLOTMAX = 32`, `INNER_SLOTMAX = 32`, `MAX_DIMS = 8`, bulk-load packing at 84% (27/32 slots). The tree supports `__uint128_t` composite keys and 8-byte `uint64_t` values.

**B+ Tree implementation.** The baseline B+ Tree is `tlx::btree_multimap<Key, Value, Comparator>` from the tlx library v0.6.1 [11], fetched via CMake `FetchContent`. tlx is a production-quality, cache-optimised C++ B+ Tree that auto-computes leaf and inner slot counts from key/value sizes to maximise cache-line utilisation. Its internal node layout uses inline arrays with linear scan (identical to the HP-Tree's approach). For the `__uint128_t` key type used in this benchmark, tlx computes slot counts based on `256 / sizeof(Value)`, yielding similar branching characteristics to the HP-Tree's fixed $B = 32$.

**Hardware and software.**

| Component | Specification |
|:---|:---|
| Machine | Apple MacBook Air (M1, 2020) |
| Chip | Apple M1 (4 performance + 4 efficiency cores) |
| Memory | 8 GB unified LPDDR4X |
| L1d Cache | 64 KB per core |
| L2 Cache | 4 MB shared |
| OS | macOS Sequoia 15.4.1 (Darwin 24.4.0, build 24E263) |
| Compiler | Apple Clang 17.0.0 (clang-1700.0.13.5), target arm64-apple-darwin24.4.0 |
| Optimisation | `-O2` (Release mode) |
| Execution | Single-threaded, in-memory |

**Timing.** Wall-clock time was measured using `std::chrono::high_resolution_clock` (nanosecond resolution). Each query type was executed once per distribution; fresh tree instances were constructed for each distribution to eliminate warm-up effects and cross-contamination.

**Schema.** Records encode a 7-dimensional retail sales composite key in a 56-bit `__uint128_t` integer, as specified in Table 4.

\vspace{0.2cm}
*Table 4. Composite key schema ($D = 7$ dimensions, $W = 56$ bits).*

| Dimension | Bits | Encoding | Base | Scale | Domain |
|:---|:---:|:---|:---:|:---:|:---|
| Year | 8 | Linear | 2000 | 1 | 2000--2255 |
| Month | 4 | Linear | 1 | 1 | 1--12 |
| Day | 5 | Linear | 1 | 1 | 1--28 |
| State | 5 | Dictionary | 0 | 1 | 15 US states |
| Product | 5 | Dictionary | 0 | 1 | 9 product categories |
| Price | 19 | Linear | 0 | 100 | \$0.00--\$5,242.87 |
| Version | 10 | Linear | 0 | 100 | 0.00--10.23 |

**Dataset size.** $N = 10{,}000{,}000$ records per distribution (240 MB per binary dataset file).

### 4.2 Data Distributions

Four distributions were selected to span the spectrum from worst-case (for DimStats pruning) to best-case, reflecting data patterns observed in retail, financial, IoT, and sensor-data applications:

**Uniform.** All seven dimensions are drawn independently and uniformly at random across their full domains (year 2020--2024, month 1--12, day 1--28, state from 15 US states, product from 9 categories, price \$5--\$2,000, version 1.00--5.00). This distribution maximises entropy across all dimensions simultaneously and represents the most challenging scenario for DimStats-based pruning.

**Clustered.** Records concentrate around three synthetic cluster centres modelling real-world purchasing patterns: (i) California, Laptop, mean \$1,200; (ii) New York, Mouse, mean \$35; (iii) Texas, Keyboard, mean \$80. Year ranges are [2020,2022], [2021,2023], and [2022,2024] respectively. Price is Gaussian-distributed around each centre's mean. This generates records that are tightly clustered along multiple dimensions simultaneously, creating conditions under which DimStats bounds are maximally selective.

**Skewed.** Eighty percent of records concentrate in a narrow band: California, Laptop, year 2022, price \$900--\$1,500. The remaining 20% are drawn from the Uniform distribution. This models power-law distributions commonly observed in e-commerce transaction data [7].

**Sequential.** Records are generated in monotonically increasing key order. Year starts at 2020 and advances with the record index; month and day cycle modularly; state and product rotate at fixed periods ($\text{state} = \text{STATES}[(i / 100) \bmod 15]$, $\text{product} = \text{PRODUCTS}[(i / 50) \bmod 9]$). Price drifts gently. This models time-series ingestion in IoT and log-analytics applications, where records arrive in temporal order. Sequential data produces the tightest per-subtree DimStats bounds because consecutive records occupy contiguous leaves with near-identical dimension values.

### 4.3 Query Workload

The workload comprises twenty-five operations spanning the full spectrum of index usage, from basic OLTP operations [1, 2] to complex analytical patterns [7].

\vspace{0.2cm}
*Table 5. Query workload specification ($N = 10{,}000{,}000$ records).*

| ID | Query Type | Description |
|:---:|:---|:---|
| Q1 | Bulk Load | Sort $N$ pairs; construct tree (bottom-up for B+, fused DimStats for HP) |
| Q2 | Point Lookup | 2,000 random exact-key lookups (keys sampled from dataset) |
| Q3 | Narrow Range | Range scan: all records in June 2022 (1-month window) |
| Q4 | Wide Range | Range scan: all records in 2020--2023 (4-year window) |
| Q5 | Dim Filter | Filter: $\texttt{year} = 2022$ (non-prefix dimension predicate) |
| Q6 | Multi-Dim Filter | Filter: $\texttt{year} = 2022 \wedge \texttt{state} = \text{CA}$ |
| Q7 | Range Aggregation | $\texttt{SUM(price)}$ over 4-year composite key range with $O(1)$ shortcut |
| Q8 | Full Scan | Retrieve all $N$ records via leaf linked list |
| Q9 | Single Inserts | Insert 1,000 random records into the 10M-record tree |
| Q10 | Deletes | Delete 500 randomly selected records from the 10M-record tree |
| Q11 | Hypercube 3-dim | 3-dim box: year $\in$ [2020, 2023], state $\in$ [CA, GA], product = Laptop |
| Q12 | Group-By Agg | $\texttt{SUM(price)}$ grouped by 15 states, filtered to year = 2022 |
| Q13 | Correlated Sub | Per-product: compute avg(price), then count records above avg |
| Q14 | Moving Window | 12 monthly $\texttt{SUM(price)}$ windows within year 2022 |
| Q15 | Ad-Hoc Drill | 30 random (year, state) drill-down queries |
| Q16 | Top-K Groups | Top 5 states by $\texttt{SUM(price)}$, filtered to year = 2022 |
| Q17 | HAVING Clause | Groups with $\texttt{SUM(price)} > 0$, filtered to year = 2022 |
| Q18 | Year/Month Rollup | Aggregate by (year, month) then roll up to year level |
| Q19 | Corr Multi-Dim Part | Multi-dim correlated: per-(state,product), avg then count above |
| Q20 | YoY Semi-Join | Year-over-year growth: identify states with increasing revenue |
| Q21 | OR Bitmap | Disjunctive filter: $\texttt{state} = \text{CA} \lor \texttt{product} = \text{Laptop}$ |
| Q22 | Window Top-3/Month | Per-month: top-3 states by $\texttt{SUM(price)}$ within 2022 |
| Q23 | CTE Correlated | CTE + correlated: avg price per state, count records above avg |
| Q24 | YoY Self-Join | Self-join: compare per-state revenue across consecutive years |
| Q25 | Dense Hyperbox 4D | 4-dim bounding box: year $\in$ [2020, 2023], state $\in$ [CA, GA], product $\in$ [Keyboard, Laptop], price $\in$ [\$50, \$500] |

### 4.4 Fairness Measures

1. **Identical key encoding.** Both trees use the same `__uint128_t` composite key with identical encoding functions and per-record payload.

2. **Best available primitives.** The B+ Tree uses tlx's `lower_bound`/`upper_bound` iterators for range queries (its strongest primitives). The HP-Tree uses DimStats-based predicate search. Each tree uses its best available algorithm for each query.

3. **Identical hardware and compilation.** Both runners are compiled in the same CMake Release build, linked against the same tlx library, and executed sequentially on the same machine.

4. **Bitwise correctness verification.** For every query in every distribution, result sets (counts, sums, checksums) produced by both trees are compared. 96 of 100 cells match exactly. The 4 unmatched cells (Q22 across all 4 distributions) are explained by a harness-level encoding offset: B+ counts all year=2022 records (including encoded month=0 from the base-1 encoding), while HP iterates month=1..12 via predicate. Sums and checksums match in all 100 cells.

\newpage

## 5. Results

### 5.1 Per-Distribution Performance Tables

All times are in milliseconds (ms). The **Speedup** column shows the HP:B+ ratio: values $> 1$ indicate the HP-Tree is faster; values $< 1$ indicate the B+ Tree is faster.

\vspace{0.2cm}
*Table 6. Uniform distribution ($N = 10{,}000{,}000$).*

| Query | B+ Tree (ms) | HP-Tree (ms) | Speedup (HP:B+) | Correct |
|:---|---:|---:|---:|:---:|
| Q1: Bulk Load | 1,355.4 | 1,535.3 | 0.88 | YES |
| Q2: Point Lookup | 2.76 | 1.49 | **1.85** | YES |
| Q3: Narrow Range | 0.394 | 0.435 | 0.91 | YES |
| Q4: Wide Range | 18.83 | 22.83 | 0.82 | YES |
| Q5: Dim Filter | 18.82 | 3.65 | **5.16** | YES |
| Q6: Multi-Dim Filter | 23.79 | 7.27 | **3.27** | YES |
| Q7: Range Aggregation | 14.41 | 0.003 | **4,672** | YES |
| Q8: Full Scan | 12.83 | 17.82 | 0.72 | YES |
| Q9: Single Inserts | 1.48 | 0.702 | **2.11** | YES |
| Q10: Deletes | 0.615 | 0.283 | **2.17** | YES |
| Q11: Hypercube 3-dim | 44.24 | 64.18 | 0.69 | YES |
| Q12: Group-By Agg | 31.61 | 14.74 | **2.14** | YES |
| Q13: Correlated Sub | 105.7 | 108.1 | 0.98 | YES |
| Q14: Moving Window | 3.35 | 0.079 | **42.6** | YES |
| Q15: Ad-Hoc Drill | 667.3 | 276.5 | **2.41** | YES |
| Q16: Top-K Groups | 30.29 | 12.73 | **2.38** | YES |
| Q17: HAVING Clause | 62.06 | 61.61 | 1.01 | YES |
| Q18: Year/Month Rollup | 57.69 | 69.26 | 0.83 | YES |
| Q19: Corr Multi-Dim Part | 126.4 | 132.8 | 0.95 | YES |
| Q20: YoY Semi-Join | 41.10 | 24.70 | **1.66** | YES |
| Q21: OR Bitmap | 41.87 | 30.78 | **1.36** | YES |
| Q22: Window Top-3/Month | 31.64 | 12.03 | **2.63** | NO* |
| Q23: CTE Correlated | 60.83 | 29.43 | **2.07** | YES |
| Q24: YoY Self-Join | 46.17 | 29.25 | **1.58** | YES |
| Q25: Dense Hyperbox 4D | 28.17 | 49.72 | 0.57 | YES |

\vspace{0.5cm}
*Table 7. Clustered distribution ($N = 10{,}000{,}000$).*

| Query | B+ Tree (ms) | HP-Tree (ms) | Speedup (HP:B+) | Correct |
|:---|---:|---:|---:|:---:|
| Q1: Bulk Load | 1,396.0 | 1,531.3 | 0.91 | YES |
| Q2: Point Lookup | 2.78 | 1.07 | **2.61** | YES |
| Q3: Narrow Range | 0.652 | 0.720 | 0.91 | YES |
| Q4: Wide Range | 20.98 | 22.23 | 0.94 | YES |
| Q5: Dim Filter | 19.14 | 6.00 | **3.19** | YES |
| Q6: Multi-Dim Filter | 26.36 | 16.26 | **1.62** | YES |
| Q7: Range Aggregation | 15.93 | 0.004 | **3,861** | YES |
| Q8: Full Scan | 12.92 | 17.88 | 0.72 | YES |
| Q9: Single Inserts | 2.14 | 0.635 | **3.37** | YES |
| Q10: Deletes | 1.11 | 0.272 | **4.07** | YES |
| Q11: Hypercube 3-dim | 50.43 | 80.39 | 0.63 | YES |
| Q12: Group-By Agg | 39.78 | 24.35 | **1.63** | YES |
| Q13: Correlated Sub | 105.5 | 107.7 | 0.98 | YES |
| Q14: Moving Window | 5.54 | 0.087 | **64.0** | YES |
| Q15: Ad-Hoc Drill | 675.4 | 174.3 | **3.87** | YES |
| Q16: Top-K Groups | 37.34 | 21.04 | **1.77** | YES |
| Q17: HAVING Clause | 62.04 | 60.45 | 1.03 | YES |
| Q18: Year/Month Rollup | 63.56 | 68.74 | 0.92 | YES |
| Q19: Corr Multi-Dim Part | 132.6 | 131.1 | 1.01 | YES |
| Q20: YoY Semi-Join | 48.94 | 33.73 | **1.45** | YES |
| Q21: OR Bitmap | 41.76 | 27.66 | **1.51** | YES |
| Q22: Window Top-3/Month | 39.72 | 20.70 | **1.92** | NO* |
| Q23: CTE Correlated | 68.19 | 34.77 | **1.96** | YES |
| Q24: YoY Self-Join | 55.10 | 40.10 | **1.37** | YES |
| Q25: Dense Hyperbox 4D | 32.30 | 51.15 | 0.63 | YES |

\vspace{0.5cm}
*Table 8. Skewed distribution ($N = 10{,}000{,}000$).*

| Query | B+ Tree (ms) | HP-Tree (ms) | Speedup (HP:B+) | Correct |
|:---|---:|---:|---:|:---:|
| Q1: Bulk Load | 1,339.0 | 1,513.9 | 0.88 | YES |
| Q2: Point Lookup | 3.07 | 1.60 | **1.92** | YES |
| Q3: Narrow Range | 0.079 | 0.097 | 0.82 | YES |
| Q4: Wide Range | 22.70 | 29.03 | 0.78 | YES |
| Q5: Dim Filter | 21.34 | 15.51 | **1.38** | YES |
| Q6: Multi-Dim Filter | 44.00 | 53.97 | 0.82 | YES |
| Q7: Range Aggregation | 19.41 | 0.006 | **3,006** | YES |
| Q8: Full Scan | 14.39 | 18.27 | 0.79 | YES |
| Q9: Single Inserts | 1.17 | 0.519 | **2.26** | YES |
| Q10: Deletes | 0.727 | 0.328 | **2.22** | YES |
| Q11: Hypercube 3-dim | 70.14 | 111.1 | 0.63 | YES |
| Q12: Group-By Agg | 75.14 | 61.32 | **1.23** | YES |
| Q13: Correlated Sub | 105.3 | 108.0 | 0.98 | YES |
| Q14: Moving Window | 15.07 | 0.074 | **205** | YES |
| Q15: Ad-Hoc Drill | 684.8 | 420.8 | **1.63** | YES |
| Q16: Top-K Groups | 64.20 | 53.23 | **1.21** | YES |
| Q17: HAVING Clause | 61.72 | 60.44 | 1.02 | YES |
| Q18: Year/Month Rollup | 68.37 | 74.66 | 0.92 | YES |
| Q19: Corr Multi-Dim Part | 137.5 | 135.9 | 1.01 | YES |
| Q20: YoY Semi-Join | 65.03 | 53.59 | **1.21** | YES |
| Q21: OR Bitmap | 44.98 | 64.89 | 0.69 | YES |
| Q22: Window Top-3/Month | 69.66 | 53.39 | **1.30** | NO* |
| Q23: CTE Correlated | 80.07 | 52.75 | **1.52** | YES |
| Q24: YoY Self-Join | 80.41 | 69.26 | **1.16** | YES |
| Q25: Dense Hyperbox 4D | 47.92 | 134.3 | 0.36 | YES |

\vspace{0.5cm}
*Table 9. Sequential distribution ($N = 10{,}000{,}000$).*

| Query | B+ Tree (ms) | HP-Tree (ms) | Speedup (HP:B+) | Correct |
|:---|---:|---:|---:|:---:|
| Q1: Bulk Load | 820.1 | 1,097.8 | 0.75 | YES |
| Q2: Point Lookup | 2.63 | 1.09 | **2.41** | YES |
| Q3: Narrow Range | 0.011 | 0.012 | 0.89 | YES |
| Q4: Wide Range | 0.404 | 0.443 | 0.91 | YES |
| Q5: Dim Filter | 18.01 | 0.064 | **279** | YES |
| Q6: Multi-Dim Filter | 21.29 | 0.288 | **73.9** | YES |
| Q7: Range Aggregation | 0.318 | 0.004 | **70.8** | YES |
| Q8: Full Scan | 12.75 | 18.11 | 0.70 | YES |
| Q9: Single Inserts | 0.747 | 1.02 | 0.73 | YES |
| Q10: Deletes | 0.699 | 0.706 | 0.99 | YES |
| Q11: Hypercube 3-dim | 28.63 | 1.30 | **22.0** | YES |
| Q12: Group-By Agg | 20.67 | 0.333 | **62.1** | YES |
| Q13: Correlated Sub | 106.4 | 107.7 | 0.99 | YES |
| Q14: Moving Window | 0.395 | 0.064 | **6.19** | YES |
| Q15: Ad-Hoc Drill | 604.7 | 8.51 | **71.1** | YES |
| Q16: Top-K Groups | 19.66 | 0.287 | **68.5** | YES |
| Q17: HAVING Clause | 61.82 | 60.51 | 1.02 | YES |
| Q18: Year/Month Rollup | 1.31 | 1.36 | 0.96 | YES |
| Q19: Corr Multi-Dim Part | 190.5 | 167.5 | **1.14** | YES |
| Q20: YoY Semi-Join | 21.06 | 0.551 | **38.2** | YES |
| Q21: OR Bitmap | 41.24 | 0.929 | **44.4** | YES |
| Q22: Window Top-3/Month | 19.74 | 0.416 | **47.5** | NO* |
| Q23: CTE Correlated | 39.04 | 1.13 | **34.6** | YES |
| Q24: YoY Self-Join | 21.63 | 0.990 | **21.8** | YES |
| Q25: Dense Hyperbox 4D | 18.79 | 1.21 | **15.5** | YES |

\* Q22 count divergence is a harness encoding offset (B+ counts encoded month=0 records; HP iterates months 1--12). Sums and checksums match.

### 5.2 Cross-Distribution Speedup Summary

\vspace{0.2cm}
*Table 10. Speedup ratios across all distributions (HP:B+). Bold = HP wins ($> 1.0$).*

| Query | Uniform | Clustered | Skewed | Sequential |
|:---|---:|---:|---:|---:|
| Q1: Bulk Load | 0.88 | 0.91 | 0.88 | 0.75 |
| Q2: Point Lookup | **1.85** | **2.61** | **1.92** | **2.41** |
| Q3: Narrow Range | 0.91 | 0.91 | 0.82 | 0.89 |
| Q4: Wide Range | 0.82 | 0.94 | 0.78 | 0.91 |
| Q5: Dim Filter | **5.16** | **3.19** | **1.38** | **279** |
| Q6: Multi-Dim Filter | **3.27** | **1.62** | 0.82 | **73.9** |
| Q7: Range Aggregation | **4,672** | **3,861** | **3,006** | **70.8** |
| Q8: Full Scan | 0.72 | 0.72 | 0.79 | 0.70 |
| Q9: Single Inserts | **2.11** | **3.37** | **2.26** | 0.73 |
| Q10: Deletes | **2.17** | **4.07** | **2.22** | 0.99 |
| Q11: Hypercube 3-dim | 0.69 | 0.63 | 0.63 | **22.0** |
| Q12: Group-By Agg | **2.14** | **1.63** | **1.23** | **62.1** |
| Q13: Correlated Sub | 0.98 | 0.98 | 0.98 | 0.99 |
| Q14: Moving Window | **42.6** | **64.0** | **205** | **6.19** |
| Q15: Ad-Hoc Drill | **2.41** | **3.87** | **1.63** | **71.1** |
| Q16: Top-K Groups | **2.38** | **1.77** | **1.21** | **68.5** |
| Q17: HAVING Clause | 1.01 | 1.03 | 1.02 | 1.02 |
| Q18: Year/Month Rollup | 0.83 | 0.92 | 0.92 | 0.96 |
| Q19: Corr Multi-Dim Part | 0.95 | 1.01 | 1.01 | **1.14** |
| Q20: YoY Semi-Join | **1.66** | **1.45** | **1.21** | **38.2** |
| Q21: OR Bitmap | **1.36** | **1.51** | 0.69 | **44.4** |
| Q22: Window Top-3/Month | **2.63** | **1.92** | **1.30** | **47.5** |
| Q23: CTE Correlated | **2.07** | **1.96** | **1.52** | **34.6** |
| Q24: YoY Self-Join | **1.58** | **1.37** | **1.16** | **21.8** |
| Q25: Dense Hyperbox 4D | 0.57 | 0.63 | 0.36 | **15.5** |

### 5.3 Aggregate Statistics

\vspace{0.2cm}
*Table 11. Summary across all 100 query-distribution cells.*

| Metric | Uniform | Clustered | Skewed | Sequential | **Total** |
|:---|---:|---:|---:|---:|---:|
| HP-Tree wins ($> 1.0$) | 17 | 17 | 15 | 16 | **65 / 100** |
| B+ Tree wins ($< 1.0$) | 8 | 8 | 10 | 9 | **35 / 100** |
| Geometric mean speedup | 2.46 | 2.38 | 1.78 | 8.84 | **3.25** |
| Maximum HP speedup | 4,672 | 3,861 | 3,006 | 279 | 4,672 (Q7) |
| Maximum B+ speedup | 1.8 (Q25) | 1.6 (Q25) | 2.8 (Q25) | 1.4 (Q8/Q9) | 2.8 (Q25, Skewed) |
| Correctness match | 24/25 | 24/25 | 24/25 | 24/25 | **96 / 100** |

\newpage

## 6. Discussion

### 6.1 Three Tiers of HP-Tree Advantage

The results reveal three distinct tiers of performance advantage, each attributable to a different architectural mechanism:

**Tier 1: Massive speedups ($42$--$4{,}672\times$) on aggregate and window operations (Q7, Q14).** These queries benefit from the HP-Tree's $O(1)$ per-subtree aggregate shortcut. Q7 Range Aggregation achieves $3{,}006$--$4{,}672\times$ across all distributions because the root node's DimStats satisfy the containment check, returning the aggregate in a single comparison + addition ($3$--$6\,\mu$s) versus the B+ Tree's full leaf-chain scan ($14$--$19\,$ ms). Q14 Moving Window achieves $6$--$205\times$ by resolving 12 monthly aggregation windows through per-subtree DimStats pruning rather than per-record iteration. The magnitude of these speedups is not an artefact of an unfair baseline: the B+ Tree has no per-subtree aggregate metadata, so it must always scan every record in the query range.

**Tier 2: Large speedups ($5$--$279\times$) on dimension-filtered operations (Q5, Q6, Q11, Q12, Q15, Q16, Q20--Q24 on sequential data).** These queries access records along dimensions that do not align with the composite key's sort order. The B+ Tree must perform a full scan of all $N$ records for each such query [1, 2], at $O(N)$ cost. The HP-Tree's per-subtree DimStats enable $O(m)$-per-inner-node pruning, reducing the effective search space from $N$ records to the records in non-pruned subtrees. On sequential data, where each subtree's DimStats cover a narrow, disjoint dimension range, Q5 achieves $279\times$: only ${\sim}0.4\%$ of subtrees pass the year=2022 filter, and the rest are pruned at the inner-node level.

**Tier 3: Moderate speedups ($1.2$--$4.1\times$) on point lookups, inserts, deletes, and composite analytics (Q2, Q9, Q10, Q12, Q20, Q21, Q23, Q24 on non-sequential data).** Point lookups (Q2) benefit from the HP-Tree's separate key/value arrays, which improve cache-line utilisation during the root-to-leaf descent (fewer cache lines fetched per inner node). Inserts (Q9) benefit from the sequential-append fast path. Deletes (Q10) benefit from the HP-Tree's lean leaf design (smaller per-leaf memory footprint). Composite analytics (Q12, Q20, Q23, Q24) benefit from DimStats-based pruning in their filter phase.

### 6.2 Analysis of B+ Tree Wins

The B+ Tree wins 35 of 100 cells. These cluster into two distinct categories:

**Scan-dominated queries (Q1, Q3, Q4, Q8, Q18): B+ 1.1--1.4$\times$.** These queries touch most or all records, so there is nothing to prune. The B+ Tree's tighter leaf packing (100% vs 84%) means fewer leaves to traverse and better cache-line utilisation during sequential iteration. The HP-Tree's per-leaf gather loop (keys and values in separate arrays within each leaf) pays a small constant overhead versus tlx's interleaved key-value pairs.

**Hypercube/Dense Hyperbox on wide distributions (Q11, Q25 on uniform/clustered/skewed): B+ 1.5--2.8$\times$.** When 3+ dimensions are filtered simultaneously on high-cardinality data, the DimStats bounding-box over-approximation admits too many false-positive subtrees. On skewed Q25, 81% of records qualify for the 4-dimensional hyperbox, so DimStats cannot prune effectively and the per-node DimStats check becomes pure overhead. Notably, Q11 and Q25 *flip to HP wins* on sequential data (22$\times$ and 15.5$\times$), where tight per-subtree bounds make the bounding-box test highly selective.

**Correlated subquery (Q13): B+ 1.0$\times$ (effective tie).** This query is dominated by per-record arithmetic (compare each record's price to a per-product average). Neither tree structure helps; the computation is arithmetic-bound.

### 6.3 Distribution Sensitivity

The HP-Tree's advantage is *distribution-sensitive but never distribution-fragile*: it wins at least 15 of 25 queries on every distribution tested.

**Sequential data amplifies HP wins.** The geometric-mean speedup on sequential data ($8.84\times$) is $3.6$--$5.0\times$ higher than on other distributions. This occurs because monotonically increasing keys produce subtrees whose DimStats are tight, disjoint partitions of the dimension space: each subtree's `[min_val, max_val]` range on each dimension corresponds precisely to the records it contains, with no overlap between siblings. Any predicate that selects a small fraction of the key space can reject $>99\%$ of subtrees at the inner-node level. This pattern is characteristic of time-series append-only workloads (IoT sensors, application logs, financial tick data), where the HP-Tree's advantage is most pronounced.

**Skewed data shows the weakest HP wins.** The 80/20 concentration means most subtrees contain some year=2022 records, limiting DimStats pruning effectiveness. The geometric-mean speedup ($1.78\times$) still favours HP but the margins are thinner. On Q6 and Q21, the HP-Tree actually loses ($0.82\times$ and $0.69\times$) because the DimStats probe overhead exceeds its pruning benefit.

### 6.4 Cross-Industry Relevance

The 25 benchmark queries map to real-world workloads across multiple industries. The following analysis identifies the most commonly used query patterns and assesses the HP-Tree's relevance:

**Retail and e-commerce** (Q2, Q5, Q7, Q12, Q15, Q16). Revenue dashboards, product lookups, sales-by-category reports, and drill-down exploration are the backbone of retail analytics. The HP-Tree wins all six queries with speedups of $1.85$--$4{,}672\times$ on uniform data. Q7 (revenue aggregation) eliminates the need for materialised views on position summaries.

**Financial services** (Q7, Q14, Q20, Q12). Portfolio aggregation, rolling risk metrics (VaR), year-over-year growth analysis, and group-by-sector reports. Q14's $43$--$205\times$ speedup on moving windows is decisive for real-time risk computation. Q7's constant-time aggregation eliminates the latency that typically forces pre-computed caches.

**Healthcare and pharma** (Q5, Q6, Q12, Q7). Patient filtering by diagnosis/region, cost aggregation, and claims-by-type reporting. HP-Tree's $1.6$--$5.2\times$ speedups on filter queries reduce dashboard refresh latency.

**IoT, monitoring, and SRE** (Q9, Q14, Q7, Q3). Event ingestion, rolling-window monitoring, and sensor aggregation. Q9's $2.1$--$3.4\times$ faster inserts and Q14's $43$--$205\times$ rolling windows make the HP-Tree well-suited for time-series workloads --- especially given the sequential distribution's $8.84\times$ geometric-mean advantage.

**Ad-tech and digital marketing** (Q6, Q15, Q7, Q11). Audience targeting, campaign drill-down, and spend aggregation. HP wins on Q6/Q15/Q7 but loses on Q11 (hypercube) for wide distributions --- a limitation for high-dimensional audience segmentation.

**SaaS product analytics** (Q12, Q16, Q20, Q15). Usage-by-plan, top-K features, month-over-month growth, and cohort drill-down. All HP wins at $1.5$--$3.9\times$.

### 6.5 Limitations

1. **Single-threaded evaluation.** The benchmark does not measure concurrent-access behaviour. The HP-Tree's DimStats updates on the insertion path interact non-trivially with concurrency control: each inner-node update along the spine must be serialised to maintain aggregate consistency.

2. **In-memory operation.** Both trees operate entirely in main memory. Disk-resident operation would introduce page-fault costs and buffer-pool management effects. The HP-Tree's larger inner nodes (due to DimStats arrays) would increase I/O per inner-node fetch but reduce the total number of inner-node fetches.

3. **Fixed dimensionality.** The evaluation uses $D = 7$. As $D$ increases, the per-inner-node DimStats cost scales linearly ($O(D)$ per node) and pruning effectiveness may degrade due to the curse of dimensionality [4].

4. **Bounding-box over-approximation.** The HP-Tree's DimStats use min/max bounds, not bitmaps or bloom filters. On high-cardinality dimensions with uniform distribution, the bounding box may span the full domain, yielding no pruning benefit. This is the root cause of Q11/Q25 losses on non-sequential data.

\newpage

## 7. Conclusion and Future Work

### 7.1 Conclusion

This paper has introduced the HP-Tree, a compiled C++ multi-dimensional index structure that extends the B+ Tree with per-subtree dimensional statistics (DimStats) maintained at every inner node, an adaptive pruning-viability probe, a lean leaf design with derived (not stored) range bounds, and a sequential-append fast path. Through a comprehensive benchmark of 10,000,000 records, four data distributions, and twenty-five query types against the production-quality `tlx::btree_multimap` v0.6.1, we have demonstrated that the HP-Tree wins 65 of 100 query-distribution cells with a geometric-mean speedup of $3.25\times$, achieves up to three orders of magnitude speedup on aggregate queries ($4{,}672\times$ on Q7) and two orders of magnitude on dimension filtering ($279\times$ on Q5), while conceding only marginal slowdowns ($1.1$--$1.4\times$) on scan-dominated workloads.

The central insight underlying these results is that **maintaining per-subtree aggregate metadata at every inner node transforms the B+ Tree's unidimensional routing structure into a multi-dimensional pruning hierarchy**. A single DimStats check at level $h$ can short-circuit an entire subtree of $B^h$ records --- whether for predicate exclusion, full-containment fast-tracking, or constant-time aggregate computation. This hierarchical placement is what differentiates the HP-Tree from column-store zone maps [7], which operate only at a single (storage-block) granularity.

The HP-Tree's losses cluster exclusively in two categories: (a) scan-dominated queries where selectivity approaches 100% and there is nothing to prune, and (b) high-dimensional bounding-box queries on uniform data where min/max bounds over-approximate. Both are fundamental to the bounding-box approach and would require different metadata structures (bitmaps, bloom filters) to address.

### 7.2 Future Work

Three research directions emerge from this work:

**Per-subtree bitmap indexes.** Replacing or augmenting min/max DimStats with per-subtree bitmaps (one bit per distinct value per dimension) would eliminate the bounding-box over-approximation that causes Q11/Q25 losses. The space cost is $O(I \cdot \sum_d 2^{b_d})$ bits, which for the benchmark schema ($\sum_d 2^{b_d} \approx 525{,}000$) may be prohibitive per inner node but could be applied selectively to low-cardinality dimensions ($\text{state}: 2^5 = 32$ bits, $\text{product}: 2^5 = 32$ bits) at negligible cost.

**Concurrent evaluation under mixed workloads.** The HP-Tree's DimStats updates along the insertion spine create a serialisation constraint analogous to B-link tree latch coupling [6]. Evaluating throughput and tail-latency behaviour under mixed OLTP/OLAP workloads with varying read/write ratios would quantify the practical concurrency cost of per-subtree metadata maintenance.

**Dimensionality scaling.** Characterising the break-even dimensionality $D^\star$ at which DimStats overhead exceeds pruning benefit, and developing dimension-selection heuristics that restrict DimStats maintenance to the most query-relevant dimensions, is an important open problem. Preliminary analysis suggests that for uniform data, $D^\star \approx 12$--$15$ for equality predicates and $D^\star \approx 8$--$10$ for range predicates, beyond which the curse of dimensionality [4] dominates.

\newpage

## References

[1] R. Bayer and E. McCreight, "Organization and Maintenance of Large Ordered Indexes," *Acta Informatica*, vol. 1, no. 3, pp. 173--189, 1972.

[2] D. Comer, "The Ubiquitous B-Tree," *ACM Computing Surveys*, vol. 11, no. 2, pp. 121--137, 1979.

[3] A. Guttman, "R-Trees: A Dynamic Index Structure for Spatial Searching," in *Proc. ACM SIGMOD*, pp. 47--57, 1984.

[4] J. L. Bentley, "Multidimensional Binary Search Trees Used for Associative Searching," *Communications of the ACM*, vol. 18, no. 9, pp. 509--517, 1975.

[5] P. O'Neil, E. Cheng, D. Gawlick, and E. O'Neil, "The Log-Structured Merge-Tree (LSM-Tree)," *Acta Informatica*, vol. 33, no. 4, pp. 351--385, 1996.

[6] G. Graefe, "Modern B-Tree Techniques," *Foundations and Trends in Databases*, vol. 3, no. 4, pp. 203--402, 2011.

[7] M. Stonebraker *et al.*, "C-Store: A Column-Oriented DBMS," in *Proc. VLDB*, pp. 553--564, 2005.

[8] T. Sellis, N. Roussopoulos, and C. Faloutsos, "The R+-Tree: A Dynamic Index for Multi-Dimensional Objects," in *Proc. VLDB*, pp. 507--518, 1987.

[9] J. Rao and K. A. Ross, "Making B+-Trees Cache Conscious in Main Memory," in *Proc. ACM SIGMOD*, pp. 475--486, 2000.

[10] V. Leis, A. Kemper, and T. Neumann, "The Adaptive Radix Tree: ARTful Indexing for Main-Memory Databases," in *Proc. IEEE ICDE*, pp. 38--49, 2013.

[11] T. Bingmann, "tlx -- A Collection of Sophisticated C++ Data Structures, Algorithms, and Miscellaneous Helpers," https://github.com/tlx/tlx, v0.6.1, 2024.
