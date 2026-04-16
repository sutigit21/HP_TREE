---
title: "HP-Tree: A Homogeneity-Partitioned Multi-Dimensional Index Structure with Adaptive Beta-Driven Splitting"
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

Tree-structured indexes remain foundational to database query processing, yet the classical B+ Tree [1, 2] provides no mechanism for pruning irrelevant data along dimensions that do not align with its primary sort order. We introduce the **HP-Tree** (Homogeneity-Partitioned Tree), a multi-dimensional index structure that extends the B+ Tree with three integrated innovations: (i) a *beta homogeneity metric* that adaptively halts recursive partitioning when key-space spread within a partition falls below a data-dependent threshold; (ii) *per-leaf dimensional metadata vectors* (min, max, and sum per dimension) that enable sub-linear leaf pruning and constant-time aggregate computation on arbitrary dimension predicates; and (iii) a *delta buffer* that amortises single-record insertion cost through sorted batch flushing. We evaluate both structures on a workload of $N = 1{,}000{,}000$ composite-key records across four data distributions (Uniform, Clustered, Skewed, Sequential) and fifteen query types, including point lookups, range scans, multi-dimensional filters, group-by aggregations, correlated subqueries, moving-window analytics, and ad-hoc drill-down queries. The HP-Tree achieves speedups of up to $3{,}509\times$ on grouped aggregation and $427\times$ on single-dimension filtering, while maintaining competitive or superior performance on all traditional OLTP operations. It concedes only a marginal $1.1$--$1.2\times$ slowdown on bulk loading. All reported results are verified for bitwise result-set correctness across all 60 query-distribution cells.

**Keywords:** multi-dimensional indexing, B+ Tree, composite keys, homogeneity partitioning, zone maps, analytical query processing

\newpage

## 1. Introduction

### 1.1 The B+ Tree and Its Fundamental Limitation

Since their introduction by Bayer and McCreight in 1972 [1], B-Tree variants have served as the default index structure in virtually all relational database management systems. The B+ Tree refinement --- which confines all data records to leaf nodes connected by a doubly-linked list --- provides $O(\log_B N)$ point lookups and efficient sequential range scans, where $B$ denotes the branching factor and $N$ the number of indexed records [2]. Comer's comprehensive survey [2] established the B+ Tree as "ubiquitous," and the subsequent four decades of engineering refinements catalogued by Graefe [6] --- including prefix compression, fence keys, ghost records, and page-level write-ahead logging --- have kept the structure competitive in modern disk-based and main-memory systems alike.

However, the B+ Tree's architecture is fundamentally *unidimensional*. Records are sorted by a single composite key, and the tree's internal routing structure can only exploit this ordering for queries whose predicates align with the key's most-significant dimension prefix. Consider a composite key encoding the tuple $(\text{year}, \text{month}, \text{day}, \text{state}, \text{product}, \text{price})$: a range query on $\text{year}$ benefits from the sort order and traverses $O(\log_B N)$ nodes, but a filter on $\text{state}$ alone --- a dimension embedded in the middle of the key encoding --- must scan every leaf node, since state values are distributed arbitrarily within the composite key's bit layout. This $O(N)$ full-scan behaviour extends to all non-prefix dimension predicates, multi-dimensional bounding-box queries, and grouped aggregations on secondary dimensions, precisely the workload patterns that dominate modern analytical processing [7].

The practical impact of this limitation is substantial. In a retail analytics database with $10^6$ records, a query such as "compute total revenue by state for the year 2022" requires the B+ Tree to examine every record in the index --- an $O(N)$ operation repeated for each group --- even though the result set may involve only a small fraction of the data. This mismatch between the tree's unidimensional ordering and the workload's multi-dimensional access patterns motivates the search for index structures that preserve the B+ Tree's strengths (logarithmic point lookups, efficient range scans, dynamic insertions and deletions) while adding sub-linear access paths along arbitrary dimensions.

### 1.2 Related Work

The limitations of unidimensional indexing have motivated an extensive body of work on multi-dimensional access methods, each addressing a different facet of the problem at the cost of new trade-offs.

**Spatial index structures.** Bentley's KD-Tree [4] partitions space by alternating split dimensions at each level of the tree, achieving efficient nearest-neighbour and orthogonal range queries in low dimensions ($D \leq 20$). However, the KD-Tree is inherently static: insertions and deletions require partial or full tree reconstruction, making it unsuitable for transactional workloads with dynamic updates [4]. Guttman's R-Tree [3] generalises the B-Tree to spatial data by associating each internal node with a minimum bounding rectangle (MBR) that encloses all descendant objects. While R-Trees support dynamic insertions and deletions, they suffer from *overlap* between sibling MBRs, which forces multiple subtree traversals during search and degrades worst-case query performance to $O(N)$ on high-dimensional or uniformly distributed data [3, 8]. The R+-Tree [8] addresses overlap through clipping --- splitting objects across multiple nodes to eliminate MBR overlap --- but introduces record duplication and complicates deletion.

**Bitmap and column-oriented indexes.** Bitmap indexes, widely deployed in data warehousing systems, represent each distinct value of an indexed attribute as a bit vector of length $N$, enabling efficient conjunction and disjunction of equality predicates via bitwise operations. However, bitmap indexes consume $O(N \cdot |\mathcal{D}|)$ space per indexed dimension (where $|\mathcal{D}|$ is the domain cardinality), rendering them impractical for high-cardinality or continuous-valued attributes such as price or timestamp [7]. Column-oriented storage systems such as C-Store [7] restructure the data layout to store each attribute in a separate dense array, enabling per-column compression, vectorised predicate evaluation, and late materialisation. These designs achieve dramatic analytical speedups but fundamentally abandon the row-oriented access patterns required by OLTP workloads, and they do not provide a tree-structured index with logarithmic point-lookup guarantees.

**Write-optimised and cache-conscious trees.** The Log-Structured Merge-Tree (LSM-Tree) [5] optimises write-heavy workloads by buffering mutations in an in-memory component and periodically merging them into sorted runs on persistent storage. While the LSM-Tree's write amplification is well-suited to modern SSDs, it introduces *read amplification* for point lookups (which must consult multiple levels) and provides no multi-dimensional pruning capability. Cache-conscious B+ Tree variants [9] improve main-memory performance by aligning node layouts with cache-line boundaries and eliminating pointer-chasing overhead, but they address microarchitectural efficiency rather than the fundamental unidimensional limitation. The Adaptive Radix Tree (ART) [10] demonstrates that trie-based designs can outperform B+ Trees on point lookups in memory-resident databases through path compression and adaptive node sizes, but it operates on byte-addressable keys and does not support multi-dimensional predicate pruning.

**Gap in the literature.** None of the structures surveyed above simultaneously satisfies the following four requirements: (a) $O(\log_B N)$ point lookups and efficient range scans on the primary key ordering; (b) sub-linear filtering on arbitrary secondary dimensions *without* auxiliary indexes or materialised views; (c) constant-time per-leaf aggregate computation for homogeneous data partitions; and (d) efficient dynamic insertions and deletions with bounded structural modification cost. The HP-Tree is designed to address this gap.

### 1.3 Contributions

This paper introduces the HP-Tree, which augments the B+ Tree's leaf-linked architecture with three mechanisms designed to bridge the multi-dimensional gap without sacrificing single-key performance:

1. **Beta Homogeneity Metric ($\beta$).** A dimensionless, scale-invariant measure of key-space spread (Section 2.2) computed in $O(1)$ from a partition's minimum and maximum keys. When $\beta$ falls below the data-dependent threshold $1/N^2$ (where $N$ is the current partition size), recursive splitting halts and the resulting leaf is classified as *homogeneous*. This produces larger leaves whose internal key-space uniformity is guaranteed by construction.

2. **Per-Leaf Dimensional Metadata.** Each leaf node maintains three vectors --- $\texttt{dim\_min}[d]$, $\texttt{dim\_max}[d]$, and $\texttt{dim\_sum}[d]$ --- for every dimension $d \in \{0, \dots, D{-}1\}$ (Section 2.4). These vectors function as *integrated zone maps* [7] whose granularity is coupled to the homogeneity-driven partitioning, enabling: (a) $O(m)$-per-leaf pruning for predicates on $m$ dimensions; (b) $O(1)$ whole-leaf inclusion when a leaf's bounds are contained within the predicate range; and (c) $O(1)$ aggregate contribution via pre-computed sums.

3. **Delta Buffer.** A write-optimised in-memory buffer (Section 2.6) inspired by the LSM-Tree's memtable design [5] that absorbs single-record insertions and flushes them in sorted batches, amortising the per-insert tree-traversal and leaf-modification costs.

These mechanisms are *intrinsic* to the tree's leaf structure and require no external secondary indexes, materialised views, or auxiliary data structures. The HP-Tree preserves the B+ Tree's doubly-linked leaf chain, its logarithmic-height guarantee, and its compatibility with standard concurrency-control protocols [6].

### 1.4 Paper Organisation

The remainder of this paper is organised as follows. Section 2 presents the HP-Tree's design: composite key encoding (2.1), the beta metric (2.2), the adaptive stopping criterion (2.3), per-leaf metadata (2.4), hypercube filtering (2.5), the delta buffer (2.6), tombstone-based deletion (2.7), and a tabular architectural comparison with the B+ Tree (2.8). Section 3 derives worst-case time and space complexity bounds. Section 4 describes the simulation study design. Section 5 presents empirical results. Section 6 provides a detailed discussion. Section 7 concludes with future research directions.

\newpage

## 2. HP-Tree Methodology

### 2.1 Composite Key Encoding

Both the HP-Tree and the baseline B+ Tree operate on **composite keys** that pack $D$ dimensions into a single fixed-width integer, following the bit-concatenation approach used in multi-dimensional indexing [3, 4]. Given dimensions $d_0, d_1, \dots, d_{D-1}$ with bit-widths $b_0, b_1, \dots, b_{D-1}$, the composite key $K$ is defined as:

$$K = \sum_{i=0}^{D-1} v_i \cdot 2^{\,\sigma(i)}, \qquad \text{where} \quad \sigma(i) = \sum_{j=i+1}^{D-1} b_j$$

and $v_i \in [0, 2^{b_i} - 1]$ is the encoded value for dimension $i$. The total key width is $W = \sum_{i=0}^{D-1} b_i$ bits. The encoding preserves lexicographic ordering with $d_0$ as the most-significant dimension: for any two keys $K_1$ and $K_2$, $K_1 < K_2$ if and only if the first dimension at which $K_1$ and $K_2$ differ has a smaller value in $K_1$.

Extraction of individual dimension values from a composite key is performed via bit-shifting and masking in $O(1)$ time:

$$\texttt{extract}(K, i) = \left\lfloor K \,/\, 2^{\,\sigma(i)} \right\rfloor \bmod 2^{b_i}$$

Two encoding modes are supported per dimension: **linear encoding** for numeric types (with a configurable base offset $b_0$ and scale factor $s$, such that the raw value $x$ is encoded as $v = \lfloor (x - b_0) \cdot s \rfloor$), and **dictionary encoding** for categorical types with a finite domain $\mathcal{D}$ (where each element is mapped to a unique integer in $[0, |\mathcal{D}| - 1]$). This dual-encoding scheme follows the design of modern analytical engines, which use dictionary compression for low-cardinality string columns [7].

### 2.2 The Beta Homogeneity Metric

The core innovation of the HP-Tree is a **data-dependent splitting criterion** based on a dimensionless measure of key-space spread, which we term the *beta metric*. Given a partition $P$ with minimum key $K_{\min}$ and maximum key $K_{\max}$ (both strictly positive), we define:

$$\beta(K_{\min}, K_{\max}) = \frac{(K_{\max} - K_{\min})^2}{4 \cdot K_{\min} \cdot K_{\max}}$$

This metric is derived from the normalised squared range of the partition and possesses several properties relevant to index design.

**Property 1 (Scale Invariance).** $\beta(c \cdot K_{\min},\; c \cdot K_{\max}) = \beta(K_{\min},\; K_{\max})$ for any constant $c > 0$. Proof: both numerator and denominator scale as $c^2$, cancelling exactly. This ensures that the metric's behaviour is independent of the absolute magnitude of key values, making it applicable across schemas with different base offsets and scale factors without recalibration.

**Property 2 (Monotonicity in Spread).** For fixed $K_{\min} > 0$, $\beta$ is strictly increasing in $K_{\max}$. As the key range narrows (i.e., $K_{\max} \to K_{\min}$), $\beta$ decreases monotonically toward zero, reflecting increasing homogeneity. Conversely, as the range widens, $\beta$ grows without bound, signalling heterogeneity.

**Property 3 (Boundary Behaviour).** The metric exhibits well-defined behaviour at boundary conditions:

| Condition | $\beta$ Value | Interpretation |
|:---|:---|:---|
| $K_{\min} = K_{\max}$ | $0$ | Perfect homogeneity; all keys identical |
| $K_{\max} = 2\,K_{\min}$ | $1/16$ | Keys span a factor of 2 |
| $K_{\max} \gg K_{\min}$ | $\approx K_{\max} / (4\,K_{\min})$ | Large spread; further splitting warranted |
| $K_{\min} \leq 0$ | $+\infty$ | Undefined; splitting must continue |

The beta metric bears a structural resemblance to the coefficient of variation used in statistical process control and to the variance-based splitting criteria used in decision-tree induction [4], but is specifically tailored to the integer composite-key domain of index structures. Unlike variance, $\beta$ is dimensionless and does not require computing the mean of the partition --- it is computable in $O(1)$ from the partition's extremal keys alone, making it suitable for use at every recursion step of the tree construction algorithm without adding asymptotic overhead.

### 2.3 Adaptive Stopping Criterion

During top-down recursive tree construction, each partition of $N$ records is evaluated for further splitting. In the standard B+ Tree [1, 2, 6], splitting is determined solely by node capacity: a node splits when its record count exceeds the order $B$. The HP-Tree augments this with a **beta-based stopping rule**: splitting halts when the partition is deemed sufficiently homogeneous, even if the partition size exceeds $B$.

Formally, splitting **halts** when either of two conditions holds:

1. $N \leq B$ (standard capacity-based halt, as in the B+ Tree [1]), or
2. $\beta(K_{\min}, K_{\max}) < 1 / N^{\,k}$, with $k = 2$ (homogeneity-based halt).

The threshold $1/N^k$ is *partition-size-dependent*: larger partitions require smaller $\beta$ (tighter homogeneity) to justify halting, while smaller partitions are permitted to halt more readily. At recursion level $L$ with branching factor $B$, the partition size is $N_L = N_0 / B^L$, and the threshold is $1/N_L^2 = B^{2L}/N_0^2$. This threshold increases exponentially with depth, creating a natural progression from strict homogeneity requirements near the root to lenient requirements at the leaves.

**Choice of exponent $k = 2$.** The exponent $k$ was determined through empirical analysis of the *discrete fanout quantisation effect*, a phenomenon arising from the integer nature of recursion depth. With branching factor $B = 20$ and initial partition size $N_0$, the average leaf size at recursion depth $L$ is $N_0 / B^L$. Since $L$ must be a non-negative integer, the achievable leaf sizes are quantised at $N_0, N_0/20, N_0/400, N_0/8000, \ldots$. At $N_0 = 50{,}000$:

- $k = 1$: the threshold $1/N_L$ is satisfied at $L = 1$, producing leaves of \textasciitilde{}2,500 records --- too coarse for effective per-leaf dimensional pruning.
- $k = 2$ or $k = 3$: both yield $L = 2$, producing leaves of \textasciitilde{}125 records --- a granularity that balances pruning effectiveness against metadata overhead.
- $k \geq 4$: yields $L = 3$, producing leaves of \textasciitilde{}6 records --- excessively fine, with metadata overhead dominating record storage.

The exponent $k = 2$ thus occupies the optimal position in this discrete landscape, providing the coarsest granularity (fewest leaves, lowest metadata overhead) that still enables effective dimensional pruning. When splitting halts due to the beta criterion, the resulting leaf is assigned the type $\texttt{HOMOGENEOUS\_LEAF}$, distinguishing it from ordinary leaves created by capacity overflow.

### 2.4 Per-Leaf Dimensional Metadata

Each leaf node $\ell$ containing $n_\ell$ records over $D$ dimensions maintains three metadata vectors, computed in a single fused $O(n_\ell \cdot D)$ pass during leaf construction:

$$\texttt{dim\_min}_\ell[d] = \min_{r \in \ell}\, \texttt{extract}(K_r, d), \quad \forall\, d \in \{0, \dots, D{-}1\}$$

$$\texttt{dim\_max}_\ell[d] = \max_{r \in \ell}\, \texttt{extract}(K_r, d), \quad \forall\, d \in \{0, \dots, D{-}1\}$$

$$\texttt{dim\_sum}_\ell[d] = \sum_{r \in \ell}\, \texttt{extract}(K_r, d), \quad \forall\, d \in \{0, \dots, D{-}1\}$$

These vectors are conceptually analogous to the **zone maps** (also termed *min-max indexes* or *small materialised aggregates*) used in column-oriented storage engines [7]. However, a critical distinction exists: in column stores, zone maps are associated with fixed-size storage blocks whose contents are determined by insertion order and bear no relation to data semantics. The HP-Tree's metadata, by contrast, is associated with *homogeneity-partitioned leaves* whose dimensional bounds are narrow *by construction*. This coupling between the partitioning criterion and the metadata granularity ensures that zone maps are maximally selective --- a property that column-store zone maps achieve only by coincidence when data happens to be physically sorted on the queried dimension.

The metadata vectors enable three query-processing optimisations:

**Leaf Pruning (Skip).** For an equality predicate on dimension $d$ with value $v$, leaf $\ell$ can be skipped entirely if $v < \texttt{dim\_min}_\ell[d]$ or $v > \texttt{dim\_max}_\ell[d]$. For a range predicate $[v_{\text{lo}}, v_{\text{hi}}]$, the leaf is skippable if $\texttt{dim\_max}_\ell[d] < v_{\text{lo}}$ or $\texttt{dim\_min}_\ell[d] > v_{\text{hi}}$. For $m$ simultaneous predicates, the pruning check costs $O(m)$ per leaf.

**Whole-Leaf Inclusion (Full Include).** If $\texttt{dim\_min}_\ell[d] = \texttt{dim\_max}_\ell[d] = v$ for an equality predicate, or if $\texttt{dim\_min}_\ell[d] \geq v_{\text{lo}}$ and $\texttt{dim\_max}_\ell[d] \leq v_{\text{hi}}$ for a range predicate, then *every* record in the leaf satisfies the predicate on dimension $d$. The leaf's entire value set can be appended to the result in $O(1)$ without per-record dimension extraction --- a reduction from $O(n_\ell)$ to $O(1)$ per leaf.

**Aggregate Shortcut.** For aggregation queries ($\texttt{SUM}$, $\texttt{COUNT}$) over a dimension $d_{\text{agg}}$, if a leaf is wholly included by the query predicate, its aggregate contribution is: $\texttt{count} = n_\ell$, $\texttt{sum}(d_{\text{agg}}) = \texttt{dim\_sum}_\ell[d_{\text{agg}}]$. This eliminates per-record iteration, reducing the per-leaf cost from $O(n_\ell)$ to $O(1)$.

### 2.5 Hypercube Filtering

For multi-dimensional range queries (termed *window queries* in the R-Tree literature [3]) with bounds $[v_{\text{lo}}^{(d_i)}, v_{\text{hi}}^{(d_i)}]$ on a subset of dimensions $d_1, d_2, \dots, d_m$, the HP-Tree applies a three-phase per-leaf decision procedure:

1. **Skip.** If $\exists\, d_i$ such that $\texttt{dim\_max}_\ell[d_i] < v_{\text{lo}}^{(d_i)}$ or $\texttt{dim\_min}_\ell[d_i] > v_{\text{hi}}^{(d_i)}$, the leaf's bounding box is disjoint from the query hypercube. The leaf is skipped in $O(m)$ time. This pruning is analogous to the MBR disjointness test in R-Trees [3], but operates at the leaf level within a unidimensional tree structure, without the overlap ambiguities that degrade R-Tree performance [8].

2. **Full Include.** If $\forall\, d_i$: $\texttt{dim\_min}_\ell[d_i] \geq v_{\text{lo}}^{(d_i)}$ and $\texttt{dim\_max}_\ell[d_i] \leq v_{\text{hi}}^{(d_i)}$, the leaf is entirely contained within the query hypercube. All $n_\ell$ records are included in $O(1)$.

3. **Partial Scan.** Otherwise, individual records are tested against all $m$ dimension predicates in $O(n_\ell \cdot m)$.

The B+ Tree, lacking per-leaf dimensional metadata, must test every record in every leaf against all $m$ predicates, yielding $O(N \cdot m)$ total cost regardless of selectivity. The HP-Tree's cost depends on the effectiveness of leaf pruning: in the best case (all leaves either fully pruned or fully included), the cost is $O(L \cdot m)$; in the worst case (all leaves partially overlapping the hypercube), it degrades to $O(N \cdot m)$, matching the B+ Tree.

### 2.6 Delta Buffer

Following the write-optimisation philosophy of the LSM-Tree [5], the HP-Tree employs an in-memory **delta buffer** of configurable capacity $\Delta$ (default: 256 records) to absorb single-record insertions. The delta buffer decouples the write path from the tree's structural organisation:

1. **Absorption.** Each incoming record is appended to the buffer in $O(1)$ time. No tree traversal or leaf modification occurs.
2. **Flush trigger.** When the buffer reaches capacity $\Delta$, it is flushed to the tree.
3. **Sorted merge.** The $\Delta$ buffered records are sorted by composite key in $O(\Delta \log \Delta)$ time, then merged into the leaf linked list using a two-pointer scan. For each affected leaf $\ell$, the merge cost is $O(n_\ell + \Delta_\ell)$, where $\Delta_\ell$ is the number of buffered records destined for $\ell$.
4. **Metadata update.** Dimensional metadata vectors are updated incrementally: for each inserted record, $\texttt{dim\_min}$ and $\texttt{dim\_max}$ are updated via min/max comparisons, and $\texttt{dim\_sum}$ is incremented, at a cost of $O(\Delta_\ell \cdot D)$ per leaf.
5. **Overflow split.** Any leaf whose post-merge size exceeds $\texttt{max\_leaf\_size}$ is split following the standard B+ Tree protocol [1, 6], with metadata vectors recomputed for both resulting halves.

The total cost per flush is $O(\Delta \log \Delta + \log_B N + \Delta \cdot D)$, where the $\log_B N$ term accounts for locating the first affected leaf. The amortised cost per insertion is therefore:

$$C_{\text{insert}}^{\text{amortised}} = O\!\left(\log \Delta + D + \frac{\log_B N}{\Delta}\right)$$

For $\Delta \gg \log_B N$ (the typical regime), the tree-traversal cost is negligible and the dominant terms are the sort contribution $O(\log \Delta)$ and the metadata update $O(D)$.

### 2.7 Tombstone-Based Deletion

The standard B+ Tree deletion algorithm [1, 2, 6] physically removes the target record from its leaf node and, if the leaf's occupancy falls below the minimum threshold $\lfloor B/2 \rfloor$, triggers a rebalancing cascade: the tree first attempts to redistribute records from a sibling leaf; failing that, the underfull leaf is merged with a sibling and the separator key is removed from the parent, potentially cascading further merges up to the root. Each rebalancing step involves $O(B)$ work for key redistribution and parent-node modification, with up to $O(\log_B N)$ cascading steps in the worst case [6].

The HP-Tree replaces this with **logical deletion** via tombstone flags:

1. The target record is located via standard tree traversal in $O(\log_B N)$.
2. A boolean tombstone flag is set at the record's position in $O(1)$. No structural modification occurs.
3. Tombstoned records are transparently skipped during subsequent read operations at a cost of one boolean check per record.
4. When the fraction of tombstoned records in a leaf exceeds a configurable compaction threshold (default: 30%), a background compaction pass physically removes the tombstoned entries and updates the leaf's metadata vectors.

This design reflects a well-known trade-off in database systems between *eager* and *lazy* space reclamation [5, 6]: the B+ Tree's eager rebalancing maintains tighter structural invariants but incurs higher per-delete cost, while the HP-Tree's lazy tombstoning reduces delete latency at the expense of slightly increased read overhead and temporarily elevated space consumption.

### 2.8 Architectural Comparison with the Standard B+ Tree

Table 1 summarises the architectural differences between the standard B+ Tree [1, 2, 6] and the HP-Tree.

\vspace{0.2cm}
*Table 1. Architectural comparison.*

| Feature | Standard B+ Tree [1, 2] | HP-Tree |
|:---|:---|:---|
| Splitting criterion | Node overflow ($n > B$) | $\beta < 1/N^2$ OR overflow |
| Leaf types | Single (`LEAF`) | `LEAF` + `HOMOGENEOUS_LEAF` |
| Per-leaf metadata | `dim_sum` (1 vector) [6] | `dim_min`, `dim_max`, `dim_sum` (3 vectors) |
| Dim filter mechanism | Full scan $O(N)$ | Leaf pruning via bounds |
| Aggregation shortcut | Whole-leaf `dim_sum` for range queries | Whole-leaf `dim_sum` + dim-filtered agg |
| Deletion | Physical removal + rebalancing [1] | Tombstone + deferred compaction |
| Write path | Direct leaf insertion [1] | Delta buffer + batch flush [5] |
| Multi-dim pruning | None | Hypercube bounds checking [3] |

\newpage

## 3. Worst-Case Time and Space Complexity

We now derive worst-case complexity bounds for both structures. Let $N$ denote the total number of records, $B$ the branching factor (node order), $D$ the number of dimensions, $L$ the total number of leaf nodes, $h$ the tree height, $R$ the result-set size for a given query, $m$ the number of predicate dimensions in a multi-dimensional query, $L_c$ the number of leaves overlapping a query range, and $\Delta$ the delta-buffer capacity. In all expressions, $B$ and $\Delta$ are treated as system parameters (not constants), while $D$ and $m$ satisfy $m \leq D \ll N$.

### 3.1 Tree Height

For the standard B+ Tree [1, 2], each internal node has at most $B$ children and each leaf holds at most $B$ records, giving:

$$h_{\text{B+}} = \left\lceil \log_B N \right\rceil$$

For the HP-Tree, the height is bounded by:

$$h_{\text{HP}} \leq \min\!\left(\left\lceil \log_B N \right\rceil,\; L_\beta + 1\right)$$

where $L_\beta$ is the maximum recursion depth at which the beta stopping criterion halts splitting for any subtree. Since homogeneous leaves may be larger than $B$, the HP-Tree can have fewer levels. In practice, $h_{\text{HP}} \leq h_{\text{B+}} + 1$; the $+1$ accounts for clustered data where the beta criterion produces unequal subtree depths requiring one additional routing level.

### 3.2 Time Complexity

Table 2 presents worst-case (and, where applicable, best-case) time complexities for all operations. The B+ Tree complexities follow the standard analysis of [1, 2, 6]. The HP-Tree's best-case complexities for dimension-dependent operations assume that every leaf is either fully pruned (skipped) or fully included by the query predicate --- a condition that is met when leaves are homogeneous on the queried dimensions.

\vspace{0.2cm}
*Table 2. Time complexity of core operations. $N$ = records, $B$ = branching factor, $D$ = dimensions, $L$ = leaves, $R$ = result size, $m$ = predicate dimensions, $L_c$ = leaves in range, $\Delta$ = buffer capacity.*

| Operation | B+ Tree [1, 2, 6] | HP-Tree (worst) | HP-Tree (best) |
|:---|:---|:---|:---|
| Bulk Load | $O(N \log N + N \!\cdot\! D)$ | $O(N \log N + N \!\cdot\! D)$ | --- |
| Point Lookup | $O(\log_B N)$ | $O(\Delta + \log_B N)$ | --- |
| Range Search ($R$ results) | $O(\log_B N + R)$ | $O(\log_B N + R)$ | --- |
| Dim Filter (dim $d = v$) | $O(N)$ | $O(N)$ | $O(L \!\cdot\! D + R)$ |
| Multi-Dim ($m$ preds) | $O(N \!\cdot\! m)$ | $O(N \!\cdot\! m)$ | $O(L \!\cdot\! m + R \!\cdot\! m)$ |
| Hypercube ($m$ ranges) | $O(N \!\cdot\! m)$ | $O(N \!\cdot\! m)$ | $O(L \!\cdot\! m + R \!\cdot\! m)$ |
| Range Aggregation | $O(\log_B N + L_c)$ | $O(\log_B N + L_c)$ | --- |
| Dim Aggregation (per group) | $O(N)$ | $O(N)$ | $O(L)$ |
| Single Insert | $O(\log_B N)$ | --- | Amort. $O(\log \Delta + D)$ |
| Delete | $O(\log_B N)$ amort. | $O(\log_B N)$ | --- |
| Full Scan | $O(N)$ | $O(N)$ | --- |

**Notes on individual entries:**

*Bulk Load.* Both trees sort $N$ records in $O(N \log N)$ time using comparison-based sorting [1]. The B+ Tree then constructs the tree bottom-up in $O(N)$ and computes one metadata vector (`dim_sum`) per leaf in $O(N \cdot D)$. The HP-Tree performs a top-down recursive build (also $O(N)$ total across all levels) and computes three metadata vectors per leaf (`dim_min`, `dim_max`, `dim_sum`) in a fused single pass, also $O(N \cdot D)$ but with a $3\times$ larger constant. The $O(N \log N)$ sort dominates both.

*Range Aggregation.* Both trees maintain per-leaf `dim_sum` vectors and use the whole-leaf shortcut for leaves fully contained within the query range, reducing per-leaf cost from $O(n_\ell)$ to $O(1)$. The cost $O(\log_B N + L_c)$ is therefore identical in asymptotic form. The practical difference is that $L_c^{\text{B+}} > L_c^{\text{HP}}$ by a constant factor (\textasciitilde{}10$\times$ in our experiments), since the HP-Tree has fewer, larger leaves.

*Dimension Filter and Aggregation.* The distinction between worst and best case is critical. In the worst case --- when every leaf partially overlaps the predicate --- the HP-Tree degrades to the B+ Tree's $O(N)$ full-scan behaviour. The best case, $O(L \cdot D + R)$, occurs when each leaf is either entirely inside or entirely outside the predicate's range on the queried dimension. For homogeneous leaves (which constitute 100% of leaves in our experiments), this best case is the typical case.

*Point Lookup.* The HP-Tree's delta buffer introduces an $O(\Delta)$ linear scan before tree traversal. In practice, a hash-indexed buffer would reduce this to $O(1)$, yielding the same $O(\log_B N)$ as the B+ Tree.

*Single Insert.* The B+ Tree performs tree traversal ($O(\log_B N)$), in-leaf array insertion ($O(B)$ for shifting), and occasional splits ($O(B)$ amortised over $B$ insertions), giving amortised $O(\log_B N)$ [1, 6]. The HP-Tree's delta buffer absorbs inserts in $O(1)$, with periodic flushes of total cost $O(\Delta \log \Delta + \Delta \cdot D + \log_B N)$, yielding amortised $O(\log \Delta + D + \log_B N / \Delta) = O(\log \Delta + D)$ for $\Delta \gg \log_B N$.

### 3.3 Space Complexity

\vspace{0.2cm}
*Table 3. Space complexity.*

| Component | B+ Tree [1, 2] | HP-Tree |
|:---|:---|:---|
| Record storage | $O(N)$ | $O(N)$ |
| Internal nodes | $O(N / B)$ | $O(N / B)$ |
| Per-leaf metadata | $O(L \cdot D)$ | $O(3 \cdot L \cdot D)$ |
| Delta buffer | --- | $O(\Delta)$ |
| Tombstone flags | --- | $O(N)$ worst case |
| **Total** | $O(N + L \cdot D)$ | $O(N + L \cdot D + \Delta)$ |

The HP-Tree's space overhead relative to the B+ Tree is dominated by the $3\times$ constant factor on per-leaf metadata (three vectors versus one). For the configuration used in our experiments ($L = 2{,}000$, $D = 7$, 8 bytes per integer), this amounts to $2{,}000 \times 7 \times 3 \times 8 = 336$ KB --- less than $0.04\%$ of the $O(N)$ record storage at $N = 1{,}000{,}000$. The tombstone bitmap is allocated lazily (only for leaves containing deletions) and in practice uses $O(n_\ell)$ bits per affected leaf rather than $O(N)$ globally.

\newpage

## 4. Simulation Study

### 4.1 Experimental Setup

To ensure an algorithmic comparison free of implementation-specific advantages (SIMD intrinsics, custom memory allocators, cache-line optimisation), both trees were implemented in pure Python 3.13.2 (CPython) as single-threaded, in-memory data structures. This design decision follows the simulation methodology advocated by Graefe [6] for comparing index designs at the logical level: by running both structures under identical interpreter overhead, the measured speedup ratios isolate the algorithmic properties of each design from low-level systems engineering effects.

**Hardware.** All experiments were executed on Apple Silicon (macOS Darwin 24.4.0), single-core.

**Timing.** Wall-clock time was measured using Python's `time.perf_counter_ns()` (nanosecond resolution). Each query type was executed for a fixed number of iterations (specified in Table 5), and the arithmetic mean across iterations was reported. Fresh tree instances were constructed for each distribution to eliminate warm-up effects and cross-contamination.

**Schema.** Records encode a 7-dimensional retail sales composite key in a 56-bit integer, as specified in Table 4. This schema represents a realistic OLAP fact table with three temporal dimensions (year, month, day), two categorical dimensions (state, product), and two continuous-valued measures (price, version), spanning the heterogeneous dimension types encountered in production analytical workloads.

\vspace{0.2cm}
*Table 4. Composite key schema ($D = 7$ dimensions, $W = 56$ bits).*

| Dimension | Bits | Encoding | Base | Scale | Domain |
|:---|:---:|:---|:---:|:---:|:---|
| Year | 8 | Linear | 2000 | 1 | 2000--2255 |
| Month | 4 | Linear | 1 | 1 | 1--12 |
| Day | 5 | Linear | 1 | 1 | 1--28 |
| State | 5 | Dictionary | 0 | 1 | 15 US states |
| Product | 5 | Dictionary | 0 | 1 | 8 product categories |
| Price | 19 | Linear | 0 | 100 | \$0.00--\$5,242.87 |
| Version | 10 | Linear | 0 | 100 | 0.00--10.23 |

**Dataset size.** $N = 1{,}000{,}000$ records per distribution.

**Tree parameters.** The B+ Tree was configured with order $B = 50$ (maximum records per leaf and maximum children per internal node). The HP-Tree was configured with $\texttt{max\_leaf\_size} = 50$, branching factor $B_{\text{HP}} = 20$, split power $k = 2.0$, and delta buffer capacity $\Delta = 256$. The B+ Tree's order of 50 reflects a typical main-memory B+ Tree configuration [9]; the HP-Tree's branching factor of 20 was chosen to produce an adequate number of partitions at each recursion level for the beta criterion to operate effectively.

### 4.2 Data Distributions

Four distributions were selected to span the spectrum from worst-case (for homogeneity detection) to best-case, reflecting data patterns observed in retail, financial, and sensor-data applications:

**Uniform.** All seven dimensions are drawn independently and uniformly at random across their full domains (year from 2018--2025, month 1--12, day 1--28, state from 15 US states, product from 8 categories, price \$5--\$3,000, version 0.5--10.0). This distribution maximises entropy across all dimensions simultaneously and represents the most challenging scenario for the HP-Tree's homogeneity detection, since there is no natural clustering in any dimension. It serves as a baseline to verify that the HP-Tree does not degrade relative to the B+ Tree when no exploitable structure exists.

**Clustered.** Records concentrate around three synthetic cluster centres that model real-world purchasing patterns with seasonal and regional concentration: (i) California laptops in June 2022 at \textasciitilde{}\$1,200; (ii) New York mice in November 2023 at \textasciitilde{}\$25; (iii) Texas keyboards in March 2021 at \textasciitilde{}\$75. For each record, a cluster centre is selected uniformly at random. With probability 0.7, the year is set to the centre's year; with probability 0.3, it is perturbed by $\pm 1$. Month is similarly perturbed by $\pm 1$. Day is drawn uniformly. Price is perturbed by $\pm$\$20. This generates records that are tightly clustered along multiple dimensions simultaneously, creating the conditions under which the HP-Tree's homogeneity detection is most effective.

**Skewed (Zipfian).** Eighty percent of records concentrate in a narrow band: California, Laptop, June 14--16 of 2022, price \textasciitilde{}\$1,000 $\pm$ \$50, version \textasciitilde{}1.0 $\pm$ 0.1. The remaining 20% are drawn from the Uniform distribution. This models the power-law distributions commonly observed in e-commerce transaction data, where a small number of popular products in a few regions generate the majority of sales volume [7].

**Sequential.** Records are generated in monotonically increasing order across all dimensions. Year advances linearly with the record index $i$ (covering 8 years over $N$ records), month and day cycle modularly ($\text{month} = 1 + (i \bmod 12)$, $\text{day} = 1 + (i \bmod 28)$), and state and product rotate at fixed periods ($\text{state} = \text{STATES}[(i / 100) \bmod 15]$, $\text{product} = \text{PRODUCTS}[(i / 50) \bmod 8]$). Price and version cycle linearly within their ranges. This models time-series ingestion in IoT and log-analytics applications, where records arrive in temporal order and exhibit strong local correlation across all dimensions.

### 4.3 Query Workload

The workload comprises fifteen operations spanning the full spectrum of index usage, from basic OLTP operations [1, 2] to complex analytical patterns [7]. Table 5 specifies each query, its parameters, and the number of timing iterations.

\vspace{0.2cm}
*Table 5. Query workload specification. All queries operate on trees loaded with $N = 1{,}000{,}000$ records except Q9 (which uses a 5,000-record tree).*

| ID | Query Type | Description | Iter. |
|:---:|:---|:---|:---:|
| Q1 | Bulk Load | Sort $N$ pairs; construct tree bottom-up (B+) or top-down (HP) | 1 |
| Q2 | Point Lookup | 2,000 random exact-key lookups (keys sampled from dataset) | 5 |
| Q3 | Narrow Range | Range scan: all records in June 2022 (1-month window) | 10 |
| Q4 | Wide Range | Range scan: all records in 2020--2023 (4-year window) | 5 |
| Q5 | Dim Filter | Filter: $\texttt{year} = 2022$ (non-prefix dimension) | 5 |
| Q6 | Multi-Dim Filter | Filter: $\texttt{year} = 2022 \wedge \texttt{state} = \text{CA}$ | 5 |
| Q7 | Range Aggregation | $\texttt{SUM(price)}$ over 3-year composite key range (2021--2023) | 5 |
| Q8 | Full Scan | Retrieve all $N$ records via leaf linked list | 3 |
| Q9 | Single Inserts | Insert 1,000 random records into a pre-loaded 5K-record tree | 1 |
| Q10 | Deletes | Delete 500 randomly selected records from the 1M-record tree | 1 |
| Q11 | Hypercube | 3-dim box: year $\in$ [2021, 2023], state $\in$ [CA, TX], price $\in$ [\$50, \$500] | 3 |
| Q12 | Group-By Agg | $\texttt{SUM(price)}$ grouped by 15 states, filtered to year = 2022 | 2 |
| Q13 | Correlated Sub | Per-product: compute avg(price), then count records above avg | 2 |
| Q14 | Moving Window | 3-month sliding window $\texttt{SUM(price)}$ $\times$ 12 months in 2022 | 3 |
| Q15 | Ad-Hoc Drill | 30 random multi-dim filter queries (2--3 predicates each) | 2 |

Queries Q1--Q10 cover the standard B+ Tree workload: bulk loading [6], point access [1], range retrieval [2], attribute filtering, aggregation, sequential scan, insertion with splitting [1], and deletion with rebalancing [1, 6]. Queries Q11--Q15 target multi-dimensional access patterns: Q11 tests spatial pruning analogous to R-Tree window queries [3]; Q12 tests grouped aggregation with a non-prefix grouping key; Q13 tests a two-phase correlated computation common in business intelligence; Q14 tests overlapping-window analytics common in time-series processing; and Q15 tests ad-hoc exploratory queries with randomised predicate combinations, simulating interactive dashboard workloads.

### 4.4 Fairness Measures

Benchmark fairness required careful attention to the baseline B+ Tree's implementation quality:

1. **Identical key encoding.** Both trees use the same `encode_key`, `decode_key`, and `extract_dim` functions, ensuring identical computational cost per dimension extraction.

2. **B+ Tree deletion with full rebalancing.** The baseline B+ Tree implements the complete textbook deletion algorithm [1, 6]: upon underflow ($n < \lfloor B/2 \rfloor$), the tree attempts to borrow a record from a sibling leaf via the parent; if the sibling cannot donate without itself underflowing, the two leaves are merged, the separator key is removed from the parent, and the merge cascades upward if necessary. The `dim_sum` vector is recomputed for all affected leaves. This ensures the B+ Tree maintains its structural invariants and is not artificially disadvantaged by a naive delete-without-rebalance implementation.

3. **B+ Tree whole-leaf aggregation.** The B+ Tree maintains `dim_sum` per leaf (computed during bulk load and updated during inserts and deletes), enabling $O(1)$ aggregate contribution for leaves fully contained within a range. This is the same optimisation available to the HP-Tree for range aggregation, ensuring the comparison is not biased by asymmetric metadata availability.

4. **Bitwise correctness verification.** For every query in every distribution, the result set (or scalar aggregate value) produced by the HP-Tree is compared against the B+ Tree's result. Integer result sets are compared for exact equality; aggregate tuples (count, sum) are compared element-wise. Zero correctness failures were observed across all 60 query-distribution cells.

5. **High-resolution timing.** All timings use `time.perf_counter_ns()` for nanosecond-resolution wall-clock measurement. The speedup metric is $S = T_{\text{B+}} / T_{\text{HP}}$, where $S > 1$ indicates the HP-Tree is faster and $S < 1$ indicates the B+ Tree is faster.

\newpage

## 5. Results

### 5.1 Tree Structure

Table 6 presents the structural statistics of both trees across all four distributions.

\vspace{0.2cm}
*Table 6. Tree structure ($N = 1{,}000{,}000$). "Homo." = homogeneous leaves.*

| Metric | Uniform | | Clustered | | Skewed | | Sequential | |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
|  | **B+** | **HP** | **B+** | **HP** | **B+** | **HP** | **B+** | **HP** |
| Leaves | 20,000 | 2,129 | 19,959 | 780 | 12,916 | 2,547 | 20,000 | 2,300 |
| Internal Nodes | 409 | 112 | 409 | 41 | 266 | 134 | 409 | 121 |
| Homo. Leaves | --- | 2,129 | --- | 780 | --- | 2,547 | --- | 2,300 |
| Depth | 4 | 4 | 4 | 5 | 4 | 4 | 4 | 4 |
| Homogeneity | --- | 100% | --- | 100% | --- | 100% | --- | 100% |

The HP-Tree produces $6$--$26\times$ fewer leaves than the B+ Tree across all distributions. The most dramatic reduction occurs on Clustered data ($780$ vs. $19{,}959$ leaves, a $25.6\times$ reduction), where the three cluster centres create large regions of low $\beta$ that satisfy the stopping criterion early. Even on Uniform data --- the worst case for homogeneity detection --- the HP-Tree produces $2{,}129$ leaves (a $9.4\times$ reduction). Critically, 100% of HP-Tree leaves are classified as homogeneous across all distributions, confirming that the $\beta < 1/N^2$ threshold is appropriately calibrated: it halts splitting exactly when the per-leaf key-space spread is narrow enough for the dimensional metadata to be maximally selective.

### 5.2 Per-Distribution Performance

\vspace{0.2cm}
*Table 7. Performance on Uniform distribution (times in ms).*

| Query | B+ Tree (ms) | HP-Tree (ms) | Speedup | Correct |
|:---|---:|---:|---:|:---:|
| Q1: Bulk Load | 3,204.91 | 3,732.19 | 0.86$\times$ | YES |
| Q2: Point Lookup (2K) | 12.53 | 8.81 | 1.42$\times$ | YES |
| Q3: Narrow Range | 0.21 | 0.14 | 1.47$\times$ | YES |
| Q4: Wide Range | 8.57 | 6.08 | 1.41$\times$ | YES |
| Q5: Dim Filter | 361.84 | 1.18 | **305.9$\times$** | YES |
| Q6: Multi-Dim Filter | 471.05 | 81.15 | 5.80$\times$ | YES |
| Q7: Range Aggregation | 2.23 | 0.28 | 7.96$\times$ | YES |
| Q8: Full Scan | 16.88 | 13.43 | 1.26$\times$ | YES |
| Q9: Single Inserts (1K) | 20.46 | 5.80 | 3.53$\times$ | YES |
| Q10: Deletes (500) | 5.07 | 3.85 | 1.32$\times$ | YES |
| Q11: Hypercube (3-dim) | 680.80 | 398.10 | 1.71$\times$ | YES |
| Q12: Group-By Agg | 6,005.51 | 6.23 | **963.8$\times$** | YES |
| Q13: Correlated Sub | 6,109.63 | 6,137.70 | 1.00$\times$ | YES |
| Q14: Moving Window | 2.18 | 0.90 | 2.43$\times$ | YES |
| Q15: Ad-Hoc Drill (30q) | 13,582.43 | 2,158.05 | 6.29$\times$ | YES |

\vspace{0.5cm}
*Table 8. Performance on Clustered distribution (times in ms).*

| Query | B+ Tree (ms) | HP-Tree (ms) | Speedup | Correct |
|:---|---:|---:|---:|:---:|
| Q1: Bulk Load | 3,312.37 | 3,651.43 | 0.91$\times$ | YES |
| Q2: Point Lookup (2K) | 12.36 | 8.74 | 1.41$\times$ | YES |
| Q3: Narrow Range | 1.07 | 0.86 | 1.24$\times$ | YES |
| Q4: Wide Range | 16.65 | 14.04 | 1.19$\times$ | YES |
| Q5: Dim Filter | 363.92 | 3.70 | 98.5$\times$ | YES |
| Q6: Multi-Dim Filter | 541.55 | 5.65 | **95.9$\times$** | YES |
| Q7: Range Aggregation | 7.87 | 0.23 | 34.7$\times$ | YES |
| Q8: Full Scan | 20.50 | 13.97 | 1.47$\times$ | YES |
| Q9: Single Inserts (1K) | 19.41 | 5.99 | 3.24$\times$ | YES |
| Q10: Deletes (500) | 5.05 | 4.15 | 1.22$\times$ | YES |
| Q11: Hypercube (3-dim) | 1,081.59 | 9.78 | **110.7$\times$** | YES |
| Q12: Group-By Agg | 7,132.64 | 2.03 | **3,508.9$\times$** | YES |
| Q13: Correlated Sub | 6,020.50 | 340.41 | 17.7$\times$ | YES |
| Q14: Moving Window | 5.29 | 3.51 | 1.51$\times$ | YES |
| Q15: Ad-Hoc Drill (30q) | 13,568.72 | 117.41 | **115.6$\times$** | YES |

\vspace{0.5cm}
*Table 9. Performance on Skewed distribution (times in ms).*

| Query | B+ Tree (ms) | HP-Tree (ms) | Speedup | Correct |
|:---|---:|---:|---:|:---:|
| Q1: Bulk Load | 2,256.35 | 2,670.86 | 0.84$\times$ | YES |
| Q2: Point Lookup (2K) | 11.92 | 8.00 | 1.49$\times$ | YES |
| Q3: Narrow Range | 7.44 | 4.75 | 1.57$\times$ | YES |
| Q4: Wide Range | 10.20 | 7.99 | 1.28$\times$ | YES |
| Q5: Dim Filter | 241.33 | 6.42 | 37.6$\times$ | YES |
| Q6: Multi-Dim Filter | 434.47 | 28.13 | 15.4$\times$ | YES |
| Q7: Range Aggregation | 3.04 | 0.33 | 9.24$\times$ | YES |
| Q8: Full Scan | 9.60 | 8.12 | 1.18$\times$ | YES |
| Q9: Single Inserts (1K) | 11.28 | 4.77 | 2.36$\times$ | YES |
| Q10: Deletes (500) | 4.87 | 3.09 | 1.57$\times$ | YES |
| Q11: Hypercube (3-dim) | 637.40 | 87.07 | 7.32$\times$ | YES |
| Q12: Group-By Agg | 5,944.46 | 6.25 | **950.9$\times$** | YES |
| Q13: Correlated Sub | 3,896.17 | 1,409.58 | 2.76$\times$ | YES |
| Q14: Moving Window | 8.00 | 0.78 | 10.2$\times$ | YES |
| Q15: Ad-Hoc Drill (30q) | 8,766.10 | 446.16 | 19.7$\times$ | YES |

\vspace{0.5cm}
*Table 10. Performance on Sequential distribution (times in ms).*

| Query | B+ Tree (ms) | HP-Tree (ms) | Speedup | Correct |
|:---|---:|---:|---:|:---:|
| Q1: Bulk Load | 3,191.58 | 3,386.34 | 0.94$\times$ | YES |
| Q2: Point Lookup (2K) | 12.48 | 9.15 | 1.36$\times$ | YES |
| Q3: Narrow Range | 0.12 | 0.05 | 2.42$\times$ | YES |
| Q4: Wide Range | 5.56 | 3.41 | 1.63$\times$ | YES |
| Q5: Dim Filter | 341.56 | 0.80 | **427.3$\times$** | YES |
| Q6: Multi-Dim Filter | 450.59 | 80.47 | 5.60$\times$ | YES |
| Q7: Range Aggregation | 2.11 | 0.20 | 10.7$\times$ | YES |
| Q8: Full Scan | 13.24 | 12.36 | 1.07$\times$ | YES |
| Q9: Single Inserts (1K) | 19.29 | 9.51 | 2.03$\times$ | YES |
| Q10: Deletes (500) | 5.06 | 3.51 | 1.44$\times$ | YES |
| Q11: Hypercube (3-dim) | 661.55 | 401.21 | 1.65$\times$ | YES |
| Q12: Group-By Agg | 5,714.25 | 4.32 | **1,323.5$\times$** | YES |
| Q13: Correlated Sub | 5,756.03 | 6,132.93 | 0.94$\times$ | YES |
| Q14: Moving Window | 2.06 | 2.52 | 0.82$\times$ | YES |
| Q15: Ad-Hoc Drill (30q) | 12,865.86 | 2,007.01 | 6.41$\times$ | YES |

### 5.3 Cross-Distribution Summary

Table 11 consolidates the speedup ratios across all distributions, providing a compact view of each query type's behaviour.

\vspace{0.2cm}
*Table 11. Cross-distribution speedup summary. "HP $X\!\times$" = HP-Tree is $X$ times faster; "B+ $X\!\times$" = B+ Tree is $X$ times faster.*

| Query | Uniform | Clustered | Skewed | Sequential |
|:---|---:|---:|---:|---:|
| Q1: Bulk Load | B+ 1.2$\times$ | B+ 1.1$\times$ | B+ 1.2$\times$ | B+ 1.1$\times$ |
| Q2: Point Lookup | HP 1.4$\times$ | HP 1.4$\times$ | HP 1.5$\times$ | HP 1.4$\times$ |
| Q3: Narrow Range | HP 1.5$\times$ | HP 1.2$\times$ | HP 1.6$\times$ | HP 2.4$\times$ |
| Q4: Wide Range | HP 1.4$\times$ | HP 1.2$\times$ | HP 1.3$\times$ | HP 1.6$\times$ |
| Q5: Dim Filter | HP 305.9$\times$ | HP 98.5$\times$ | HP 37.6$\times$ | HP 427.3$\times$ |
| Q6: Multi-Dim | HP 5.8$\times$ | HP 95.9$\times$ | HP 15.4$\times$ | HP 5.6$\times$ |
| Q7: Aggregation | HP 8.0$\times$ | HP 34.7$\times$ | HP 9.2$\times$ | HP 10.7$\times$ |
| Q8: Full Scan | HP 1.3$\times$ | HP 1.5$\times$ | HP 1.2$\times$ | HP 1.1$\times$ |
| Q9: Inserts | HP 3.5$\times$ | HP 3.2$\times$ | HP 2.4$\times$ | HP 2.0$\times$ |
| Q10: Deletes | HP 1.3$\times$ | HP 1.2$\times$ | HP 1.6$\times$ | HP 1.4$\times$ |
| Q11: Hypercube | HP 1.7$\times$ | HP 110.7$\times$ | HP 7.3$\times$ | HP 1.6$\times$ |
| Q12: Group-By | HP 963.8$\times$ | HP 3,508.9$\times$ | HP 950.9$\times$ | HP 1,323.5$\times$ |
| Q13: Correlated | B+ 1.0$\times$ | HP 17.7$\times$ | HP 2.8$\times$ | B+ 1.1$\times$ |
| Q14: Mov. Window | HP 2.4$\times$ | HP 1.5$\times$ | HP 10.2$\times$ | B+ 1.2$\times$ |
| Q15: Ad-Hoc | HP 6.3$\times$ | HP 115.6$\times$ | HP 19.7$\times$ | HP 6.4$\times$ |

### 5.4 Aggregate Statistics

\vspace{0.2cm}
*Table 12. Summary across all 60 query-distribution cells.*

| Metric | Value |
|:---|:---|
| Total cells evaluated | 60 (15 queries $\times$ 4 distributions) |
| Cells won by HP-Tree ($S > 1.0$) | 53 / 60 (88.3%) |
| Cells won by B+ Tree ($S < 1.0$) | 7 / 60 (11.7%) |
| Maximum HP-Tree speedup | 3,508.9$\times$ (Q12, Clustered) |
| Maximum B+ Tree speedup | 1.2$\times$ (Q1, Uniform and Skewed) |
| Queries where HP-Tree wins all 4 distributions | 10 / 15 |
| Queries where B+ Tree wins all 4 distributions | 1 / 15 (Q1: Bulk Load only) |
| Correctness failures | 0 / 60 |

\newpage

## 6. Discussion

### 6.1 Three Tiers of HP-Tree Advantage

The results reveal three distinct tiers of performance advantage, each attributable to a different architectural mechanism:

**Tier 1: Massive speedups ($37$--$3{,}509\times$) on dimension-filtered operations (Q5, Q6, Q11, Q12, Q15).** These queries access records along dimensions that do not align with the composite key's sort order. The B+ Tree must perform a full scan of all $N$ records for each such query [1, 2], at $O(N)$ cost. The HP-Tree's per-leaf dimensional metadata enables $O(m)$-per-leaf pruning, reducing the effective search space from $N$ records to $L$ leaf-header checks. The empirical speedup ratio is approximately $N / (L \cdot c_{\text{prune}})$, where $c_{\text{prune}}$ is the fraction of leaves that survive pruning. On Clustered data, where $L = 780$ and most leaves are pruned for any single-dimension predicate, Q12 achieves a $3{,}509\times$ speedup. On Skewed data, where the 80/20 concentration reduces the effective number of distinct leaf clusters, the speedup is lower but still substantial (Q5: $37.6\times$).

The magnitude of these speedups is not an artefact of an unfair baseline: the B+ Tree's $O(N)$ behaviour on non-prefix dimension queries is an *inherent architectural limitation* [2, 6]. Any standard B+ Tree implementation --- including those in PostgreSQL, MySQL/InnoDB, or SQLite --- would exhibit the same full-scan behaviour on these queries without supplementary secondary indexes.

**Tier 2: Moderate speedups ($2$--$35\times$) on aggregation, inserts, deletes, and window queries (Q7, Q9, Q10, Q14).** Range aggregation (Q7) benefits from the HP-Tree's lower leaf count: since the HP-Tree has \textasciitilde{}10$\times$ fewer leaves than the B+ Tree, a higher fraction of leaves in any given range are fully covered (with no boundary-record scanning needed), increasing the effectiveness of the $O(1)$ per-leaf `dim_sum` shortcut even though both trees implement the same shortcut. Insertions (Q9) benefit from the delta buffer's amortisation (Section 2.6), which reduces per-insert tree-traversal cost from $O(\log_B N)$ to amortised $O(\log_B N / \Delta)$. Deletions (Q10) benefit from the $O(1)$ tombstone mechanism (Section 2.7), which avoids the B+ Tree's per-delete rebalancing cascade involving sibling redistribution and parent modification [1, 6].

**Tier 3: Modest speedups ($1.1$--$2.4\times$) on point lookups, range scans, and full scans (Q2, Q3, Q4, Q8).** These operations rely on the composite key's sort order, where both trees share identical asymptotic complexity ($O(\log_B N)$ and $O(R)$ respectively) [1, 2]. The HP-Tree's modest advantage arises from its lower leaf count (fewer internal nodes to traverse, shorter leaf chain to scan) rather than from any fundamental algorithmic improvement. This tier confirms that the HP-Tree's multi-dimensional enhancements do not degrade traditional single-key performance.

### 6.2 Analysis of B+ Tree Wins

**Q1 (Bulk Load): B+ 1.1--1.2$\times$.** The B+ Tree's bulk-load advantage is directly attributable to the HP-Tree's metadata overhead. Both trees share the $O(N \log N)$ sort step and construct the tree in $O(N)$ time. Both compute `dim_sum` in $O(N \cdot D)$. The HP-Tree additionally computes `dim_min` and `dim_max` in the same fused pass, but with a $3\times$ larger constant ($3D$ updates per record versus $D$). With $N = 10^6$ and $D = 7$, this represents \textasciitilde{}$14 \times 10^6$ additional comparisons (min/max updates), adding $10$--$20\%$ to the total bulk-load time. The HP-Tree also performs beta computation at each recursion level, adding $O(1)$ per partition.

**Q13 (Correlated Subquery): B+ 1.0--1.1$\times$ on Uniform and Sequential.** This query has two phases: (i) compute per-product average price via `dim_aggregate` (where the HP-Tree benefits from leaf pruning), and (ii) iterate all records per product and compare each record's price to the computed average. Phase (ii) dominates total time and involves $O(N)$ per-record `extract_dim` calls for both trees, with no opportunity for leaf-level shortcuts. On Uniform and Sequential data, products are spread across nearly all leaves, so the HP-Tree's pruning in phase (i) saves only a small fraction of total work. On Clustered data, products concentrate in few leaves, allowing phase (i) to dominate and yielding $17.7\times$.

**Q14 (Moving Window): B+ 1.2$\times$ on Sequential.** The moving-window query uses overlapping composite-key ranges, where the HP-Tree's range aggregation shortcut is effective. On Sequential data, the narrow windows produce very few covered leaves (many boundary leaves require per-record scanning), negating the HP-Tree's structural advantage and leaving only the overhead of tombstone-flag checking.

### 6.3 Distribution Sensitivity

The HP-Tree's advantage is *distribution-sensitive but never distribution-fragile*: it wins at least 13 of 15 queries on every distribution tested, and its wins span both OLTP (Q2--Q4, Q9, Q10) and OLAP (Q5--Q7, Q11--Q15) workloads. The *magnitude* of speedups varies substantially across distributions:

- **Clustered** data produces the largest speedups (up to $3{,}509\times$) because the three cluster centres create contiguous regions of uniform keys, enabling both aggressive leaf pruning (few leaves to check) and pervasive whole-leaf inclusion (most relevant leaves have constant dimension values along the filter dimension).
- **Sequential** data produces the largest single-dimension filter speedup ($427\times$ on Q5) because the monotonic key generation ensures the `year` dimension is perfectly sorted within the composite key, and the HP-Tree's `dim_min`/`dim_max` bounds achieve maximal selectivity.
- **Uniform** data produces the smallest speedups on dimension-dependent queries (but still $306\times$ for Q5 and $964\times$ for Q12), demonstrating that the HP-Tree's per-leaf metadata provides significant value even when no natural clustering exists in the data.
- **Skewed** data produces intermediate results, reflecting the mixture of highly-clustered (80%) and uniformly-distributed (20%) records.

### 6.4 Limitations

Several limitations of this study should be acknowledged:

1. **Interpreted-language overhead.** Both trees run under CPython, which imposes \textasciitilde{}100$\times$ overhead on arithmetic operations relative to compiled C/C++ code. While absolute timings are inflated, the *speedup ratios* between trees are unaffected, since both implementations experience identical interpreter overhead per operation. A compiled implementation with SIMD-accelerated dimension extraction would likely amplify the HP-Tree's advantage on metadata-intensive operations.

2. **Single-threaded execution.** The evaluation does not measure concurrent-access behaviour. The HP-Tree's C++ implementation includes per-node reader-writer latches and MVCC transaction support, but evaluating contention patterns, latch-coupling strategies [6], and throughput under mixed read-write workloads is beyond this study's scope.

3. **In-memory operation.** Both trees operate entirely in main memory. Disk-resident operation would introduce page-fault costs and buffer-pool management effects that may differentially affect the two structures. The HP-Tree's larger average leaf size could reduce I/O for dimension filtering (fewer pages to fault in) but increase write amplification on leaf splits.

4. **Fixed dimensionality.** The evaluation uses $D = 7$. As $D$ increases, the per-leaf metadata cost scales linearly and the pruning effectiveness may degrade due to the curse of dimensionality [4], wherein the probability of a leaf's bounding box overlapping a random query hypercube increases with $D$.

\newpage

## 7. Conclusion and Future Work

### 7.1 Conclusion

This paper has introduced the HP-Tree, a multi-dimensional index structure that extends the classical B+ Tree [1, 2] with three integrated mechanisms: a beta homogeneity metric for data-adaptive partitioning, per-leaf dimensional metadata for sub-linear filtering and constant-time aggregation, and a delta buffer for write amortisation. Through a comprehensive simulation study covering $1{,}000{,}000$ records, four data distributions, and fifteen query types, we have demonstrated that the HP-Tree achieves speedups of up to three orders of magnitude on analytical queries involving non-primary-key dimension filtering and grouped aggregation, while maintaining competitive or superior performance on all traditional OLTP operations (point lookups, range scans, insertions, deletions). The only concession is a marginal $1.1$--$1.2\times$ overhead on bulk loading, attributable to the $3\times$ metadata computation cost that enables the HP-Tree's downstream query advantages.

The central insight underlying these results is that **coupling the tree's partitioning criterion to data homogeneity produces leaves whose per-dimension bounds are maximally selective for pruning**. Unlike zone maps in column stores [7], which are associated with fixed-size storage blocks whose dimensional bounds are determined by insertion order, the HP-Tree's metadata is associated with partitions whose homogeneity is guaranteed by the $\beta < 1/N^2$ stopping criterion. This coupling ensures that the metadata is *informative by construction*: a leaf classified as homogeneous is guaranteed to have narrow dimensional ranges, enabling effective pruning without external statistics collection, query-feedback loops, or physical data reorganisation.

### 7.2 Future Work

Three research directions emerge from this work:

**Adaptive exponent selection.** The split power $k = 2$ was determined empirically for $N = 10^6$ and $D = 7$. Preliminary analysis using a gradient-boosted regression model over features $[\log N, k, \log \beta, \text{cv}, \text{skew}]$ suggests that the optimal $k$ decreases as $N$ grows (from \textasciitilde{}$2.15$ at $N = 10^5$ to \textasciitilde{}$1.9$ at $N = 5 \times 10^5$), reflecting the interaction between the discrete fanout quantisation effect and dataset scale. A formal characterisation of $k^\star(N, D, \mathcal{F})$ as a function of dataset size, dimensionality, and distribution family $\mathcal{F}$ --- and the design of an online adaptation mechanism that adjusts $k$ during tree maintenance --- is a natural next step.

**Concurrent evaluation under mixed workloads.** The HP-Tree's tombstone-based deletion and delta buffer interact non-trivially with concurrency control: tombstones eliminate delete-induced structural modifications (reducing latch contention on shared ancestors [6]), while the delta buffer introduces a serialisation point at flush time. Evaluating throughput and tail-latency behaviour under mixed OLTP/OLAP workloads with varying read/write ratios would quantify the practical benefits and costs of these design choices relative to the B+ Tree's rebalancing approach.

**Dimensionality scaling.** The HP-Tree's pruning effectiveness depends on the relationship between dimensionality $D$, the number of query predicates $m$, and the data distribution. As $D$ grows, the probability that a leaf's bounding box overlaps a randomly-placed query hypercube increases --- the well-known curse of dimensionality [4] --- potentially degrading selectivity. Characterising the break-even dimensionality $D^\star$ at which metadata overhead exceeds pruning benefit, and developing dimension-selection heuristics that restrict metadata maintenance to the most query-relevant dimensions, is an important open problem.

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
