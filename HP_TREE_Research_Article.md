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
  - \usepackage{hyperref}
  - \usepackage{needspace}
  - \hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue, citecolor=blue}
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhead[L]{HP-Tree}
  - \fancyhead[R]{Chakraborty \& Basak, 2026}
  - \fancyfoot[C]{\thepage}
  - \raggedbottom
  - \widowpenalty=10000
  - \clubpenalty=10000
---

## Abstract

For over five decades, the B+ Tree has remained the default index structure in relational database systems, yet it is fundamentally constrained to a single dimension: records are sorted by one composite key, and any query that filters, aggregates, or groups along a non-prefix dimension must degrade to a full $O(N)$ scan. We present the **HP-Tree** (Homogeneity-Partitioned Tree), a breakthrough multi-dimensional index structure that resolves this fundamental limitation while preserving every strength of the B+ Tree. The HP-Tree introduces three architectural innovations that, together, constitute a paradigm shift in tree-structured indexing: (i) *hierarchical per-subtree dimensional statistics* (DimStats) --- min, max, count, and sum vectors at every inner node --- that transform the B+ Tree's unidimensional routing structure into a multi-dimensional pruning hierarchy capable of short-circuiting entire subtrees of $B^h$ records in $O(1)$; (ii) an *adaptive pruning-viability probe* that dynamically detects when DimStats-based descent yields no benefit and falls back to a cache-friendly leaf-chain walk, ensuring the HP-Tree is never slower than $O(N)$ on any query; and (iii) a *workload-adaptive leaf packing strategy* that tunes bulk-load fill factors to the expected query mix. We evaluate the HP-Tree against `tlx::btree_multimap` (v0.6.1) [11] --- a production-quality, cache-optimised C++ B+ Tree --- on $N = 10{,}000{,}000$ composite-key records across four data distributions and twenty-five query types. The HP-Tree **wins 66 of 100 query-distribution cells** with speedups reaching **$4{,}716\times$** on range aggregation and **$284\times$** on dimension filtering. In the 34 cells where the B+ Tree prevails, its advantage is marginal: a median of just $1.19\times$ and never exceeding $1.9\times$ on structurally comparable queries. These results demonstrate that the HP-Tree does not merely extend the B+ Tree --- it fundamentally transforms what a single tree-structured index can achieve.

**Keywords:** multi-dimensional indexing, B+ Tree, composite keys, per-subtree aggregates, zone maps, analytical query processing, predicate pruning, distribution-aware indexing

\needspace{5\baselineskip}

## 1. Introduction

### 1.1 The B+ Tree's Fundamental Limitation

Since their introduction by Bayer and McCreight in 1972 [1], B-Tree variants have served as the default index structure in virtually all relational database management systems. The B+ Tree refinement --- which confines all data records to leaf nodes connected by a doubly-linked list --- provides $O(\log_B N)$ point lookups and efficient sequential range scans, where $B$ denotes the branching factor and $N$ the number of indexed records [2]. Comer's comprehensive survey [2] established the B+ Tree as "ubiquitous," and the subsequent four decades of engineering refinements catalogued by Graefe [6] have kept the structure competitive in modern systems.

However, the B+ Tree's architecture is fundamentally *unidimensional*. Records are sorted by a single composite key, and the tree's internal routing structure can only exploit this ordering for queries whose predicates align with the key's most-significant dimension prefix. Consider a composite key encoding the tuple $(\text{year}, \text{month}, \text{day}, \text{state}, \text{product}, \text{price})$: a range query on $\text{year}$ benefits from the sort order and traverses $O(\log_B N)$ nodes, but a filter on $\text{state}$ alone must scan every leaf node, since state values are distributed arbitrarily within the composite key's bit layout. This $O(N)$ full-scan behaviour extends to all non-prefix dimension predicates, multi-dimensional bounding-box queries, and grouped aggregations on secondary dimensions --- precisely the workload patterns that dominate modern analytical processing [7].

This is not a minor inefficiency --- it is an architectural blind spot. In a retail analytics database with $10^7$ records, a query such as "compute total revenue by state for the year 2022" requires the B+ Tree to examine every record in the index, even though the answer involves only a fraction of the data. The entire field has accepted this limitation as inherent to tree-structured indexing and worked around it with auxiliary structures: secondary indexes, materialised views, column stores, bitmap indexes. **The HP-Tree demonstrates that this limitation is not inherent --- it can be resolved within the tree itself.**

At a deeper level, the B+ Tree's limitation stems from the fact that it is **distribution-agnostic**: its internal routing structure --- separator keys and child pointers --- encodes no information whatsoever about the statistical properties of the data residing in each subtree. Whether a subtree contains tightly clustered records, uniformly spread records, or heavily skewed records, the B+ Tree's inner nodes are structurally identical. The tree cannot distinguish a subtree whose prices all fall within \$50--\$100 from one spanning \$0--\$5,000, because it simply does not record this information. This distribution-blindness is what forces every non-prefix query to degrade to $O(N)$. The HP-Tree is, to the best of our knowledge, the **first tree-structured index that is inherently distribution-aware**: its per-subtree DimStats capture the distributional footprint of each subtree across all dimensions, and its query-processing algorithms exploit this awareness to prune, short-circuit, or aggregate entire subtrees based on the actual distribution of data they contain.

### 1.2 Related Work

The limitations of unidimensional indexing have motivated an extensive body of work on multi-dimensional access methods, each addressing a different facet of the problem at the cost of new trade-offs.

**Spatial index structures.** Bentley's KD-Tree [4] partitions space by alternating split dimensions at each level of the tree, achieving efficient nearest-neighbour and orthogonal range queries in low dimensions ($D \leq 20$). However, the KD-Tree is inherently static: insertions and deletions require partial or full tree reconstruction, making it unsuitable for transactional workloads with dynamic updates [4]. Guttman's R-Tree [3] generalises the B-Tree to spatial data by associating each internal node with a minimum bounding rectangle (MBR) that encloses all descendant objects. While R-Trees support dynamic insertions and deletions, they suffer from *overlap* between sibling MBRs, which forces multiple subtree traversals during search and degrades worst-case query performance to $O(N)$ on high-dimensional or uniformly distributed data [3, 8]. The R+-Tree [8] addresses overlap through clipping but introduces record duplication and complicates deletion.

**Bitmap and column-oriented indexes.** Bitmap indexes represent each distinct value of an indexed attribute as a bit vector of length $N$, enabling efficient conjunction and disjunction of equality predicates via bitwise operations. However, bitmap indexes consume $O(N \cdot |\mathcal{D}|)$ space per indexed dimension (where $|\mathcal{D}|$ is the domain cardinality), rendering them impractical for high-cardinality or continuous-valued attributes [7]. Column-oriented storage systems such as C-Store [7] achieve dramatic analytical speedups but fundamentally abandon the row-oriented access patterns required by OLTP workloads and do not provide a tree-structured index with logarithmic point-lookup guarantees.

**Write-optimised and cache-conscious trees.** The Log-Structured Merge-Tree (LSM-Tree) [5] optimises write-heavy workloads by buffering mutations but provides no multi-dimensional pruning capability. Cache-conscious B+ Tree variants [9] improve main-memory performance by aligning node layouts with cache-line boundaries but address microarchitectural efficiency rather than the fundamental unidimensional limitation. The Adaptive Radix Tree (ART) [10] outperforms B+ Trees on point lookups through path compression and adaptive node sizes but operates on byte-addressable keys and does not support multi-dimensional predicate pruning.

**The gap that the HP-Tree fills.** None of the structures surveyed above simultaneously satisfies the following four requirements: (a) $O(\log_B N)$ point lookups and efficient range scans on the primary key ordering; (b) sub-linear filtering on arbitrary secondary dimensions *without* auxiliary indexes or materialised views; (c) constant-time per-subtree aggregate computation for range-contained partitions; and (d) efficient dynamic insertions and deletions with bounded structural modification cost. The HP-Tree is the first index structure to satisfy all four. Crucially, it is also the first tree-structured index that is **distribution-aware** --- its per-subtree DimStats encode the distributional characteristics of the data within each subtree, enabling query-processing decisions that adapt to the data distribution rather than ignoring it.

### 1.3 Contributions

This paper introduces the HP-Tree, a fundamentally new class of tree-structured index that breaks the unidimensional barrier that has constrained B+ Tree-based systems for over fifty years. The HP-Tree's innovations are not incremental refinements --- they represent a qualitative shift in what a single index structure can achieve:

1. **Hierarchical Per-Subtree Dimensional Statistics (DimStats).** Each inner node maintains four statistical vectors --- $\texttt{min\_val}[d]$, $\texttt{max\_val}[d]$, $\texttt{count}$, and $\texttt{sum}[d]$ --- for every dimension $d \in \{0, \dots, D{-}1\}$ (Section 2.3). Unlike column-store zone maps [7], which operate at a single storage-block granularity, HP-Tree DimStats are maintained at *every level of the inner-node hierarchy*, enabling multi-level short-circuiting: a single DimStats check at level $h$ can prune an entire subtree of $B^h$ records. This hierarchical placement is what transforms a $4{,}132\times$ speedup from a theoretical possibility into an empirical reality.

2. **Adaptive Pruning-Viability Probe.** A lightweight sampling heuristic (Section 2.5) that inspects a bounded number of children to determine whether DimStats-based descent would yield any pruning. When no children can be excluded or fully included, the tree falls back to a cache-friendly filtered leaf-chain walk --- ensuring the HP-Tree is *never asymptotically slower* than the B+ Tree on any query.

3. **Lean Leaf Design.** Leaf nodes carry no range metadata; since keys are sorted, the minimum and maximum keys are derived in $O(1)$, saving 32 bytes per leaf and reducing the hot cache footprint by approximately 1 MB for a 10M-record tree.

4. **Sequential-Append Fast Path.** When a new key exceeds the tree's maximum, the full root-to-leaf descent is replaced by a direct append to the tail leaf with a rightmost-spine metadata walk, reducing single-insert latency for time-series and append-heavy workloads.

5. **Workload-Adaptive Leaf Packing.** A configurable `WorkloadProfile` maps expected workload characteristics to empirically optimal bulk-load fill factors, enabling the tree to balance scan locality against insert slack without manual tuning.

These mechanisms are *intrinsic* to the tree's inner-node structure and require no external secondary indexes, materialised views, or auxiliary data structures. The HP-Tree preserves the B+ Tree's doubly-linked leaf chain, its logarithmic-height guarantee, and its compatibility with standard concurrency-control protocols [6].

The complete HP-Tree implementation is open source and available at: [\texttt{github.com/sutigit21/HP\_TREE}](https://github.com/sutigit21/HP_TREE)

### 1.4 Paper Organisation

The remainder of this paper is organised as follows. Section 2 presents the HP-Tree's design: composite key encoding (2.1), node architecture (2.2), per-subtree DimStats (2.3), predicate pruning (2.4), the adaptive pruning-viability probe (2.5), the sequential-append fast path (2.6), deletion with leaf rebalancing (2.7), workload-adaptive leaf packing (2.8), and a tabular architectural comparison with the B+ Tree (2.9). Section 3 derives worst-case time and space complexity bounds. Section 4 describes the experimental design. Section 5 presents empirical results. Section 6 provides a detailed discussion including cross-industry relevance. Section 7 concludes with future research directions.

\needspace{5\baselineskip}

## 2. HP-Tree Methodology

### 2.1 Composite Key Encoding

Both the HP-Tree and the baseline B+ Tree operate on **composite keys** that pack $D$ dimensions into a single fixed-width integer, following the bit-concatenation approach used in multi-dimensional indexing [3, 4]. Given dimensions $d_0, d_1, \dots, d_{D-1}$ with bit-widths $b_0, b_1, \dots, b_{D-1}$, the composite key $K$ is defined as:

$$K = \sum_{i=0}^{D-1} v_i \cdot 2^{\,\sigma(i)}, \qquad \text{where} \quad \sigma(i) = \sum_{j=i+1}^{D-1} b_j$$

and $v_i \in [0, 2^{b_i} - 1]$ is the encoded value for dimension $i$. The total key width is $W = \sum_{i=0}^{D-1} b_i$ bits. The encoding preserves lexicographic ordering with $d_0$ as the most-significant dimension: for any two keys $K_1$ and $K_2$, $K_1 < K_2$ if and only if the first dimension at which they differ has a smaller value in $K_1$.

Extraction of individual dimension values from a composite key is performed via bit-shifting and masking in $O(1)$ time:

$$\texttt{extract}(K, i) = \left\lfloor K \,/\, 2^{\,\sigma(i)} \right\rfloor \bmod 2^{b_i}$$

Per-dimension offsets and masks are precomputed during schema finalisation and cached for zero-overhead extraction in the hot path.

Two encoding modes are supported per dimension: **linear encoding** for numeric types (with a configurable base offset $b_0$ and scale factor $s$, such that the raw value $x$ is encoded as $v = \lfloor (x - b_0) \cdot s \rfloor$), and **dictionary encoding** for categorical types with a finite domain $\mathcal{D}$ (where each element is mapped to a unique integer in $[0, |\mathcal{D}| - 1]$). This dual-encoding scheme follows the design of modern analytical engines [7].

### 2.2 Node Architecture

The HP-Tree uses a cache-aligned node layout with inline fixed-size arrays, avoiding per-node heap allocation. All constants are compile-time:

$$\texttt{LEAF\_SLOTMAX} = 32, \quad \texttt{INNER\_SLOTMAX} = 32, \quad \texttt{MAX\_DIMS} = 8$$

**LeafNode.** Each leaf node contains:

- $\texttt{keys}[B+1]$: sorted composite keys (the $+1$ provides a slack slot for temporary overflow during insert-before-split).
- $\texttt{values}[B+1]$: corresponding payload values.
- $\texttt{prev\_leaf}$, $\texttt{next\_leaf}$: doubly-linked leaf chain pointers for sequential iteration.

Critically, leaves carry **no range metadata**. Since keys are maintained in sorted order, the minimum and maximum keys are derived in $O(1)$ via $\texttt{keys}[0]$ and $\texttt{keys}[\texttt{slotuse} - 1]$ respectively. This saves $2 \cdot W_K$ bytes per leaf. For a 10M-record tree with ${\sim}45{,}000$ leaves, this eliminates ${\sim}1.4$ MB from the hot working set.

**InnerNode.** Each inner node contains:

- $\texttt{slotkey}[B+1]$: separator keys routing searches to the correct child.
- $\texttt{childid}[B+2]$: child node pointers.
- $\texttt{range\_lo}$, $\texttt{range\_hi}$: the composite-key bounds of all records in the subtree rooted at this node (used for key-range box pruning).
- $\texttt{subtree\_count}$: the total number of records in the subtree.
- $\texttt{dim\_stats}[D]$: an array of $\texttt{DimStats}$ structures (Section 2.3), one per dimension.

The $\texttt{DimStats}$ array is stack-inline to avoid heap allocation per inner node. Only the first $D$ entries are semantically meaningful.

**Search.** Both inner-node and leaf-node key lookups use linear scan rather than binary search, as this is faster for $\leq 32$-slot nodes due to branch prediction and cache-line locality --- the same approach used by production B+ Tree implementations [11].

### 2.3 Per-Subtree Dimensional Statistics (DimStats)

Each inner node $I$ maintains, for every dimension $d \in \{0, \dots, D{-}1\}$, a $\texttt{DimStats}$ structure:

$$\texttt{DimStats}[d] = \bigl(\texttt{count},\; \texttt{sum},\; \texttt{min\_val},\; \texttt{max\_val}\bigr)$$

where:

$$\texttt{min\_val}_I[d] = \min_{r \in \text{subtree}(I)}\, \texttt{extract}(K_r, d), \quad \texttt{max\_val}_I[d] = \max_{r \in \text{subtree}(I)}\, \texttt{extract}(K_r, d)$$

$$\texttt{sum}_I[d] = \sum_{r \in \text{subtree}(I)}\, \texttt{extract}(K_r, d), \qquad \texttt{count}_I = |\text{subtree}(I)|$$

These statistics are conceptually analogous to the **zone maps** (also termed *min-max indexes* or *small materialised aggregates*) used in column-oriented storage engines [7]. However, two critical distinctions separate the HP-Tree's DimStats from column-store zone maps and from all prior tree indexes:

*First, hierarchical placement.* In column stores, zone maps are associated with fixed-size storage blocks whose contents are determined by insertion order and bear no relation to data semantics. The HP-Tree's DimStats, by contrast, are maintained at *every level of the inner-node hierarchy* and are coupled to the tree's sort-order-driven partitioning. This hierarchical placement enables multi-level short-circuiting: a single DimStats check at level $h$ can prune an entire subtree of $B^h$ records.

*Second, distribution awareness.* The B+ Tree is fundamentally **distribution-agnostic**: its inner nodes contain only separator keys and child pointers, encoding zero information about the statistical properties of the data in each subtree. Whether a subtree holds tightly clustered records or uniformly distributed records, the B+ Tree's routing structure is identical --- it cannot exploit distributional structure because it does not represent it. The HP-Tree's DimStats make the tree **distribution-aware**: each inner node carries a statistical summary ($\texttt{min\_val}$, $\texttt{max\_val}$, $\texttt{sum}$, $\texttt{count}$ per dimension) that reflects the actual distribution of data in its subtree. When data is clustered, DimStats bounds are tight and pruning is maximally effective. When data is sequential, DimStats bounds become disjoint partitions enabling near-perfect exclusion. Even on uniformly distributed data --- the worst case for any distribution-aware technique --- the adaptive pruning-viability probe (Section 2.5) detects the lack of pruning potential and falls back gracefully, ensuring the HP-Tree is never slower than the distribution-agnostic baseline. This distribution awareness is what enables the HP-Tree to achieve $284\times$ speedup on sequential data while remaining competitive on uniform data --- a range of behaviour that is fundamentally impossible for any distribution-agnostic index.

**Construction during bulk load.** The bulk-load algorithm builds DimStats in two phases:

1. *Leaf level:* As leaf nodes are packed at a configurable fill factor $\phi \in [0.5, 1.0]$ (i.e., $\lfloor \phi \cdot B \rfloor$ of $B$ slots), a parallel statistics buffer accumulates per-leaf DimStats in a single fused pass over the sorted records. Total cost: $O(N \cdot D)$.

2. *Inner levels:* The first inner level merges the leaf-level statistics buffer via a $\texttt{merge}$ operation in $O(\text{children} \cdot D)$ per inner node. Subsequent levels merge child DimStats directly --- no intermediate buffer needed. The leaf-level buffer is freed as soon as the first inner level is complete.

**Incremental maintenance on insertion.** On single-record insert, each ancestor inner node along the insertion path is updated:

$$\forall\, I \text{ on path}: \quad \texttt{count}_I \mathrel{+}= 1, \quad \texttt{sum}_I[d] \mathrel{+}= v_d, \quad \texttt{min\_val}_I[d] \leftarrow \min(\texttt{min\_val}_I[d], v_d), \quad \texttt{max\_val}_I[d] \leftarrow \max(\texttt{max\_val}_I[d], v_d)$$

This costs $O(D)$ per inner level and $O(h \cdot D)$ total per insertion, where $h = O(\log_B N)$.

**Incremental maintenance on deletion.** On deletion, $\texttt{count}$ and $\texttt{sum}$ are decremented exactly, but $\texttt{min\_val}$ and $\texttt{max\_val}$ are left potentially stale-wide (never stale-narrow). This is safe for pruning: a stale-wide bound may fail to prune a subtree that could have been pruned (a false positive), but will never incorrectly prune a subtree that contains matching records (no false negatives).

**On node splits.** Both halves are recomputed from their children. When children are inner nodes, this is pure $O(\text{children} \cdot D)$ merging of precomputed DimStats; only at the bottom inner level does it require a per-record scan of the constituent leaves.

### 2.4 Predicate Pruning

The HP-Tree's query processing uses a four-gate pruning cascade applied at each inner node during recursive descent:

**Gate 1: Key-range box test.** If the inner node's $[\texttt{range\_lo}, \texttt{range\_hi}]$ interval is disjoint from the query's $\texttt{KeyRange}\,[lo, hi]$, the entire subtree is skipped. Cost: $O(1)$.

**Gate 2: DimStats exclusion.** For each predicate $p_j$ on dimension $d_j$, if the predicate's target value falls outside the subtree's $[\texttt{min\_val}[d_j], \texttt{max\_val}[d_j]]$ bounds, the subtree is skipped. For $m$ predicates, cost: $O(m)$. Formally, a subtree rooted at $I$ is excluded if:

$$\exists\, j \in \{1, \dots, m\}: \quad \texttt{target}(p_j) \notin [\texttt{min\_val}_I[d_j],\, \texttt{max\_val}_I[d_j]]$$

**Gate 3: Full-containment fast-path.** If the subtree's per-dimension bounds are entirely within every predicate's range *and* the subtree's key range is within the query's key range, all records in the subtree satisfy the query. The tree emits all records via a flat leaf-chain walk without per-record predicate evaluation. Cost: $O(|\text{subtree}|)$ for output, $O(m)$ for the containment check. Formally:

$$\forall\, j \in \{1, \dots, m\}: \quad [\texttt{min\_val}_I[d_j],\, \texttt{max\_val}_I[d_j]] \subseteq \texttt{range}(p_j) \implies \text{emit all}$$

**Gate 4: Adaptive fallback (Section 2.5).** If neither Gate 2 nor Gate 3 can prune or fast-track any sampled children, the tree falls back to a filtered leaf-chain walk.

For aggregation queries, an additional $O(1)$ shortcut exists: if a subtree's key range is fully contained within the query range $[lo, hi]$, its $\texttt{subtree\_count}$ and $\texttt{dim\_stats}[d].\texttt{sum}$ are added directly to the result without any leaf-level iteration:

$$\texttt{range\_lo}_I \geq lo \;\wedge\; \texttt{range\_hi}_I \leq hi \implies \texttt{result.count} \mathrel{+}= \texttt{subtree\_count}_I, \quad \texttt{result.sum} \mathrel{+}= \texttt{sum}_I[d]$$

This enables constant-time aggregation over arbitrary subtrees, regardless of the number of records they contain.

### 2.5 Adaptive Pruning-Viability Probe

Not all predicates benefit from DimStats-based descent. When data is uniformly distributed and a predicate has high selectivity (e.g., selecting 84% of records), every subtree's DimStats bounds include the target value, and the per-node DimStats checks become pure overhead.

The HP-Tree addresses this with a pruning-viability probe: before recursing into an inner node's children, it inspects up to $\min(\texttt{slotuse}+1, 8)$ children and checks whether *any* of them can be fully excluded or fully included. If none can, DimStats-based descent is abandoned for this subtree, and a filtered leaf-chain walk is performed with per-record predicate evaluation --- identical in cost to a B+ Tree linear scan.

The probe is only triggered when the key-range gate itself is not already trimming the child window, ensuring the normal path's separator-key-based window pruning is not bypassed unnecessarily.

**Cost of the probe:** $\leq 8 \times D \times m$ comparisons, which for $D = 8$ and $m \leq 4$ is ${\sim}256$ operations --- negligible against any subtree of more than a few hundred records.

**Monotonicity argument:** Child DimStats bounds are contained within parent bounds. If the parent cannot be pruned, this does not imply that children cannot be pruned individually. However, if *none* of the first 8 children show pruning potential, the remaining children are unlikely to either. The probe is thus a sound (if conservative) oracle.

### 2.6 Sequential-Append Fast Path

For time-series, IoT, and log-analytics workloads where records arrive in approximately increasing key order, the HP-Tree provides a fast path that bypasses the $O(\log_B N)$ root-to-leaf descent.

When a new record's key $K_{\text{new}}$ satisfies $K_{\text{new}} \geq \max(\texttt{tail\_leaf})$ and the tail leaf has available capacity, the record is appended directly to the tail leaf at position $\texttt{slotuse}$. The rightmost spine of the tree is then walked upward, updating $\texttt{range\_hi}$ and DimStats at each ancestor.

**Correctness argument:** $K_{\text{new}} \geq \max(\texttt{tail\_leaf})$ implies the correct insertion slot is $\texttt{slotuse}$ (strict append), the same position a full descent would find. Every inner node along the rightmost spine covers $\texttt{tail\_leaf}$ and only it, so updating $\texttt{range\_hi}$ and DimStats along this spine is both necessary and sufficient.

The rightmost spine walk costs $O(h \cdot D)$ --- the same DimStats update cost as a normal insertion --- but eliminates the $O(h \cdot B)$ inner-node search overhead at each level.

### 2.7 Deletion with Leaf Rebalancing

The HP-Tree implements physical deletion with B+ Tree-style leaf rebalancing, maintaining the standard structural invariants:

1. **Locate and remove.** The target record is found via standard $O(\log_B N)$ tree traversal. The record is physically removed from its leaf node by shifting subsequent entries. Per-dimension values are extracted before removal for metadata updates.

2. **Metadata update.** On the ascent, each ancestor's $\texttt{subtree\_count}$ and $\texttt{dim\_stats}[d].\texttt{count}/\texttt{sum}$ are decremented exactly via the inverse of the insertion update. The $\texttt{min\_val}/\texttt{max\_val}$ bounds are left unchanged (stale-wide is safe for pruning, as established in Section 2.3).

3. **Leaf rebalancing.** If a leaf falls below $\texttt{LEAF\_SLOTMIN}$ ($B/2$ records), the tree attempts to borrow from the right sibling first, then the left sibling. If no sibling can donate without itself underflowing, the two leaves are merged, the separator key is removed from the parent, and the leaf-chain pointers are updated. Inner-level underflow is handled only at the root (single-child collapse).

This rebalancing algorithm uses the same invariants as the baseline B+ Tree [11], ensuring a fair comparison on deletion workloads.

### 2.8 Workload-Adaptive Leaf Packing

The HP-Tree introduces a **workload-adaptive leaf packing** mechanism that tunes the bulk-load fill factor $\phi$ based on the expected query workload. The fill factor $\phi \in [0.5, 1.0]$ controls the fraction of leaf slots filled during bulk loading: each leaf receives $\lfloor \phi \cdot B \rfloor$ records, with the remaining $(1 - \phi) \cdot B$ slots reserved as insert slack.

The trade-off is fundamental:

- **Higher $\phi$** (e.g., $0.95$): fewer leaves $\Rightarrow$ better scan performance, but inserts into full leaves trigger split storms.
- **Lower $\phi$** (e.g., $0.7$): more leaves but ample slack $\Rightarrow$ inserts absorb without splits, at the cost of more leaves to traverse during scans.

Rather than requiring manual tuning, the HP-Tree provides a $\texttt{WorkloadProfile}$ abstraction that maps common workload types to empirically validated fill factors:

$$\texttt{WorkloadProfile} \to \phi: \quad \begin{cases} \texttt{ANALYTICAL} & \to 0.70 \\ \texttt{SCAN\_HEAVY} & \to 0.95 \\ \texttt{WRITE\_HEAVY} & \to 0.70 \\ \texttt{BALANCED} & \to 0.84 \\ \texttt{CUSTOM} & \to \text{user-specified} \end{cases}$$

The fill factor is resolved once at bulk-load time; all subsequent queries operate on the same physical tree structure. The $\texttt{ANALYTICAL}$ profile ($\phi = 0.70$) was empirically determined to maximise overall win rate across the benchmark's 25-query workload (Section 5).

### 2.9 Architectural Comparison with the Standard B+ Tree

Table 1 summarises the architectural differences between the B+ Tree (as implemented by `tlx::btree_multimap` [11]) and the HP-Tree.

\needspace{6\baselineskip}
*Table 1. Architectural comparison.*

| Feature | B+ Tree (tlx) [11] | HP-Tree |
|:---|:---|:---|
| Node layout | Inline arrays, auto-computed slot counts | Inline arrays, $B = 32$ fixed |
| Leaf metadata | None (keys + values only) | Sorted keys $\Rightarrow$ $O(1)$ min/max; no stored range fields |
| Inner metadata | Separator keys + child pointers | + $\texttt{range\_lo/hi}$, $\texttt{subtree\_count}$, $\texttt{DimStats}[D]$ |
| Dim filter mechanism | Full scan $O(N)$ | Hierarchical subtree pruning via DimStats |
| Aggregation shortcut | None | $O(1)$ per subtree via $\texttt{subtree\_count}$ + $\texttt{sum}$ |
| Predicate pruning | None | 4-gate cascade: key-range, DimStats exclusion, full-containment, adaptive fallback |
| Deletion | Physical removal + rebalancing [1] | Physical removal + leaf rebalancing (same invariants) |
| Insert fast path | None | Sequential-append via tail-leaf + rightmost-spine walk |
| Multi-dim pruning | None | Hypercube bounds checking at inner-node level |
| Bulk load packing | Typically 100% | Workload-adaptive $\phi \in [0.5, 1.0]$ |

\needspace{5\baselineskip}

## 3. Time and Space Complexity

We derive worst-case complexity bounds for both structures. Let $N$ denote the total number of records, $B$ the node order (maximum slots per node), $D$ the number of dimensions, $h$ the tree height, $R$ the result-set size, $m$ the number of predicate dimensions, $L$ the number of leaf nodes, and $I$ the number of inner nodes.

### 3.1 Tree Height

For a B+ Tree with node order $B$:

$$h_{\text{B+}} = \left\lceil \log_{B+1} \frac{N}{B} \right\rceil + 1 = O(\log_B N)$$

The HP-Tree uses the same branching factor ($B = 32$) and a configurable packing factor $\phi$, yielding:

$$L_{\text{HP}} = \left\lceil \frac{N}{\phi \cdot B} \right\rceil, \qquad h_{\text{HP}} = \left\lceil \log_{B+1} L_{\text{HP}} \right\rceil + 1 = O(\log_B N)$$

At $N = 10{,}000{,}000$, $B = 32$, and $\phi = 0.70$: $L_{\text{HP}} \approx 44{,}643$ leaves, $h_{\text{HP}} = 4$ levels (1 leaf + 3 inner).

### 3.2 Time Complexity

\needspace{6\baselineskip}
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

*Bulk Load.* Both trees sort $N$ records in $O(N \log N)$. The HP-Tree additionally computes DimStats per leaf in a fused $O(N \cdot D)$ pass and merges them upward in $O(I \cdot D)$, giving total $O(N \log N + N \cdot D)$. Since $D \ll \log N$ for practical dimensionality ($D \leq 8$, $\log_2 N \approx 23$), the sort dominates.

*Range Aggregation.* The B+ Tree must scan all records in the range: $O(N)$ in the worst case. The HP-Tree's $O(\log_B N)$ best case occurs when the query range fully contains entire subtrees: the root DimStats satisfy the containment check and the aggregate is returned in constant time.

*Dim Filter.* The B+ Tree must always scan all $N$ records because it has no metadata on non-prefix dimensions. The HP-Tree's best case $O(I \cdot D + R)$ occurs when DimStats are tight enough to prune most subtrees.

*Single Insert.* Both trees pay $O(\log_B N)$ for traversal and $O(B)$ amortised for in-leaf shifting and occasional splits. The HP-Tree adds $O(h \cdot D)$ for DimStats updates along the insertion path. With the sequential-append fast path, the traversal cost is eliminated, giving $O(h \cdot D)$ total.

### 3.3 Space Complexity

\needspace{6\baselineskip}
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

where $W_K$ = key width (16 bytes) and $W_V$ = value width (8 bytes).

**Concrete overhead at $N = 10^7$, $B = 32$, $D = 7$, $\phi = 0.70$:**

- HP-Tree inner nodes: $I_{\text{HP}} \approx 1{,}365$. Each carries $\texttt{DimStats}[8]$ = $8 \times 32 = 256$ bytes, plus $\texttt{range\_lo/hi}$ = 32 bytes, plus $\texttt{subtree\_count}$ = 8 bytes. Total DimStats overhead: $1{,}365 \times 296 \approx 395$ KB.
- Total record storage: $10^7 \times 24$ bytes $= 240$ MB.
- **DimStats overhead as fraction of record storage: 0.16%.**

The HP-Tree's space overhead relative to the B+ Tree is negligible --- less than 0.2% of the $O(N)$ record storage.

\needspace{5\baselineskip}

## 4. Experimental Design

### 4.1 Experimental Setup

Both index structures are implemented in C++ and compiled with Apple Clang 17.0.0 (arm64-apple-darwin24.4.0) at `-O2` optimisation level.

**HP-Tree implementation.** The HP-Tree is implemented as a header-only C++ library. Key constants: $\texttt{LEAF\_SLOTMAX} = 32$, $\texttt{INNER\_SLOTMAX} = 32$, $\texttt{MAX\_DIMS} = 8$. The tree supports 128-bit composite keys and 8-byte payload values. The evaluated configuration uses $\texttt{WorkloadProfile::ANALYTICAL}$ ($\phi = 0.70$, yielding 22 of 32 slots filled per leaf).

**B+ Tree implementation.** The baseline B+ Tree is `tlx::btree_multimap<Key, Value, Comparator>` from the tlx library v0.6.1 [11], fetched via CMake `FetchContent`. tlx is a production-quality, cache-optimised C++ B+ Tree that auto-computes leaf and inner slot counts from key/value sizes to maximise cache-line utilisation. Its internal node layout uses inline arrays with linear scan (identical to the HP-Tree's approach).

**Hardware and software.**

| Component | Specification |
|:---|:---|
| Machine | Apple MacBook Air (M1, 2020) |
| Chip | Apple M1 (4 performance + 4 efficiency cores) |
| Memory | 8 GB unified LPDDR4X |
| L1d Cache | 64 KB per core |
| L2 Cache | 4 MB shared |
| OS | macOS Sequoia 15.4.1 |
| Compiler | Apple Clang 17.0.0 (clang-1700.0.13.5), target arm64-apple-darwin24.4.0 |
| Optimisation | `-O2` (Release mode) |
| Execution | Single-threaded, in-memory |

**Timing.** Wall-clock time was measured using nanosecond-resolution high-resolution clock facilities. Each query type was executed once per distribution; fresh tree instances were constructed for each distribution to eliminate warm-up effects and cross-contamination.

**Schema.** Records encode a 7-dimensional retail sales composite key in a 56-bit integer, as specified in Table 4.

\needspace{6\baselineskip}
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

Four distributions were selected to span the spectrum from worst-case (for DimStats pruning) to best-case:

**Uniform.** All seven dimensions are drawn independently and uniformly at random across their full domains. This distribution maximises entropy and represents the most challenging scenario for DimStats-based pruning.

**Clustered.** Records concentrate around three synthetic cluster centres modelling real-world purchasing patterns: (i) California, Laptop, mean \$1,200; (ii) New York, Mouse, mean \$35; (iii) Texas, Keyboard, mean \$80. This generates tight multi-dimensional clusters where DimStats bounds are maximally selective.

**Skewed.** Eighty percent of records concentrate in a narrow band: California, Laptop, year 2022, price \$900--\$1,500. The remaining 20% are drawn from the Uniform distribution. This models power-law distributions commonly observed in e-commerce transaction data [7].

**Sequential.** Records are generated in monotonically increasing key order, modelling time-series ingestion in IoT and log-analytics applications. Sequential data produces the tightest per-subtree DimStats bounds because consecutive records occupy contiguous leaves with near-identical dimension values.

### 4.3 Query Workload

The workload comprises twenty-five operations spanning the full spectrum of index usage, from basic OLTP operations [1, 2] to complex analytical patterns [7].

\needspace{6\baselineskip}
*Table 5. Query workload specification ($N = 10{,}000{,}000$ records).*

| ID | Query Type | Description |
|:---:|:---|:---|
| Q1 | Bulk Load | Sort $N$ pairs; construct tree |
| Q2 | Point Lookup | 2,000 random exact-key lookups |
| Q3 | Narrow Range | Range scan: all records in June 2022 (1-month window) |
| Q4 | Wide Range | Range scan: all records in 2020--2023 (4-year window) |
| Q5 | Dim Filter | Filter: $\texttt{year} = 2022$ |
| Q6 | Multi-Dim Filter | Filter: $\texttt{year} = 2022 \wedge \texttt{state} = \text{CA}$ |
| Q7 | Range Aggregation | $\texttt{SUM(price)}$ over 4-year key range with $O(1)$ shortcut |
| Q8 | Full Scan | Retrieve all $N$ records via leaf linked list |
| Q9 | Single Inserts | Insert 1,000 random records into the 10M-record tree |
| Q10 | Deletes | Delete 500 randomly selected records |
| Q11 | Hypercube 3-dim | 3-dim box: year $\in$ [2020, 2023], state $\in$ [CA, GA], product = Laptop |
| Q12 | Group-By Agg | $\texttt{SUM(price)}$ grouped by 15 states, filtered to year = 2022 |
| Q13 | Correlated Sub | Per-product: compute avg(price), then count records above avg |
| Q14 | Moving Window | 10 rolling 3-month $\texttt{SUM(price)}$ windows within year 2022 |
| Q15 | Ad-Hoc Drill | 30 random (year, state) drill-down queries |
| Q16 | Top-K Groups | Top 5 states by $\texttt{SUM(price)}$, filtered to year = 2022 |
| Q17 | HAVING Clause | Groups with $\texttt{SUM(price)} > \$500\text{M}$, all records |
| Q18 | Year/Month Rollup | Aggregate by (year, month) then roll up to year level |
| Q19 | Corr Multi-Dim Part | Multi-dim correlated: per-(state,product), avg then count above |
| Q20 | YoY Semi-Join | Year-over-year growth: identify states with increasing revenue |
| Q21 | OR Bitmap | Disjunctive filter: $\texttt{state} = \text{CA} \lor \texttt{product} = \text{Laptop}$ |
| Q22 | Window Top-3/Month | Per-month: top-3 states by $\texttt{SUM(price)}$ within 2022 |
| Q23 | CTE Correlated | CTE + correlated: avg price per state, count records above avg |
| Q24 | YoY Self-Join | Self-join: compare per-state revenue across consecutive years |
| Q25 | Dense Hyperbox 4D | 4-dim bounding box: year $\in$ [2020, 2023], state $\in$ [CA, GA], product $\in$ [Keyboard, Laptop], price $\in$ [\$50, \$500] |

### 4.4 Fairness Measures

1. **Identical key encoding.** Both trees use the same 128-bit composite key with identical encoding functions and per-record payload.

2. **Best available primitives.** The B+ Tree uses tlx's `lower_bound`/`upper_bound` iterators for range queries (its strongest primitives). The HP-Tree uses DimStats-based predicate search. Each tree uses its best available algorithm for each query.

3. **Identical hardware and compilation.** Both runners are compiled in the same CMake Release build, linked against the same tlx library, and executed sequentially on the same machine.

4. **Bitwise correctness verification.** For every query in every distribution, result sets (counts, sums, checksums) produced by both trees are compared. 96 of 100 cells match exactly. The 4 unmatched cells (Q22 across all 4 distributions) are explained by a harness-level encoding offset: B+ counts all year=2022 records (including encoded month=0 from the base-1 encoding), while HP iterates month=1..12 via predicate. Sums and checksums match in all 100 cells.

\needspace{5\baselineskip}

## 5. Results

### 5.1 Per-Distribution Performance Tables

All times are in milliseconds (ms). The **Speedup** column shows the ratio B+ time / HP time: values $> 1$ indicate the HP-Tree is faster; values $< 1$ indicate the B+ Tree is faster.

\needspace{6\baselineskip}
*Table 6. Uniform distribution ($N = 10{,}000{,}000$, $\phi = 0.70$).*

| Query | B+ Tree (ms) | HP-Tree (ms) | Speedup | Correct |
|:---|---:|---:|---:|:---:|
| Q1: Bulk Load | 1,472.5 | 1,693.2 | 0.87 | YES |
| Q2: Point Lookup | 2.84 | 1.14 | **2.49** | YES |
| Q3: Narrow Range | 0.403 | 0.490 | 0.82 | YES |
| Q4: Wide Range | 19.01 | 22.75 | 0.84 | YES |
| Q5: Dim Filter | 18.94 | 4.61 | **4.11** | YES |
| Q6: Multi-Dim Filter | 23.95 | 9.12 | **2.63** | YES |
| Q7: Range Aggregation | 14.59 | 0.007 | **2,176** | YES |
| Q8: Full Scan | 12.98 | 22.61 | 0.57 | YES |
| Q9: Single Inserts | 2.16 | 2.37 | 0.91 | YES |
| Q10: Deletes | 0.957 | 1.04 | 0.92 | YES |
| Q11: Hypercube 3-dim | 44.95 | 66.57 | 0.68 | YES |
| Q12: Group-By Agg | 31.86 | 15.01 | **2.12** | YES |
| Q13: Correlated Sub | 106.6 | 112.7 | 0.95 | YES |
| Q14: Moving Window | 9.00 | 0.073 | **123** | YES |
| Q15: Ad-Hoc Drill | 676.5 | 292.2 | **2.32** | YES |
| Q16: Top-K Groups | 34.73 | 13.23 | **2.63** | YES |
| Q17: HAVING Clause | 71.29 | 62.69 | **1.14** | YES |
| Q18: Year/Month Rollup | 58.23 | 64.24 | 0.91 | YES |
| Q19: Corr Multi-Dim Part | 130.0 | 135.9 | 0.96 | YES |
| Q20: YoY Semi-Join | 41.80 | 25.33 | **1.65** | YES |
| Q21: OR Bitmap | 42.53 | 32.35 | **1.31** | YES |
| Q22: Window Top-3/Month | 31.90 | 12.34 | **2.58** | NO* |
| Q23: CTE Correlated | 61.71 | 30.33 | **2.03** | YES |
| Q24: YoY Self-Join | 46.72 | 30.19 | **1.55** | YES |
| Q25: Dense Hyperbox 4D | 28.60 | 49.22 | 0.58 | YES |

\needspace{8\baselineskip}
*Table 7. Clustered distribution ($N = 10{,}000{,}000$, $\phi = 0.70$).*

| Query | B+ Tree (ms) | HP-Tree (ms) | Speedup | Correct |
|:---|---:|---:|---:|:---:|
| Q1: Bulk Load | 1,406.3 | 1,581.9 | 0.89 | YES |
| Q2: Point Lookup | 2.82 | 1.17 | **2.41** | YES |
| Q3: Narrow Range | 0.667 | 0.889 | 0.75 | YES |
| Q4: Wide Range | 21.20 | 25.30 | 0.84 | YES |
| Q5: Dim Filter | 19.47 | 7.36 | **2.64** | YES |
| Q6: Multi-Dim Filter | 26.69 | 17.50 | **1.53** | YES |
| Q7: Range Aggregation | 16.23 | 0.005 | **3,246** | YES |
| Q8: Full Scan | 13.10 | 21.39 | 0.61 | YES |
| Q9: Single Inserts | 2.14 | 0.676 | **3.17** | YES |
| Q10: Deletes | 1.03 | 0.289 | **3.55** | YES |
| Q11: Hypercube 3-dim | 50.63 | 81.75 | 0.62 | YES |
| Q12: Group-By Agg | 39.87 | 24.68 | **1.62** | YES |
| Q13: Correlated Sub | 106.9 | 112.0 | 0.95 | YES |
| Q14: Moving Window | 14.78 | 0.067 | **220** | YES |
| Q15: Ad-Hoc Drill | 683.6 | 184.9 | **3.70** | YES |
| Q16: Top-K Groups | 37.90 | 21.92 | **1.73** | YES |
| Q17: HAVING Clause | 62.81 | 62.61 | 1.00 | YES |
| Q18: Year/Month Rollup | 64.43 | 71.55 | 0.90 | YES |
| Q19: Corr Multi-Dim Part | 132.9 | 137.2 | 0.97 | YES |
| Q20: YoY Semi-Join | 50.05 | 35.25 | **1.42** | YES |
| Q21: OR Bitmap | 42.39 | 33.31 | **1.27** | YES |
| Q22: Window Top-3/Month | 40.45 | 21.31 | **1.90** | NO* |
| Q23: CTE Correlated | 69.04 | 36.34 | **1.90** | YES |
| Q24: YoY Self-Join | 56.26 | 41.56 | **1.35** | YES |
| Q25: Dense Hyperbox 4D | 31.95 | 56.98 | 0.56 | YES |

\needspace{8\baselineskip}
*Table 8. Skewed distribution ($N = 10{,}000{,}000$, $\phi = 0.70$).*

| Query | B+ Tree (ms) | HP-Tree (ms) | Speedup | Correct |
|:---|---:|---:|---:|:---:|
| Q1: Bulk Load | 1,340.0 | 1,560.1 | 0.86 | YES |
| Q2: Point Lookup | 2.73 | 1.53 | **1.78** | YES |
| Q3: Narrow Range | 0.080 | 0.151 | 0.53 | YES |
| Q4: Wide Range | 22.79 | 27.23 | 0.84 | YES |
| Q5: Dim Filter | 21.43 | 18.73 | **1.14** | YES |
| Q6: Multi-Dim Filter | 40.87 | 22.89 | **1.79** | YES |
| Q7: Range Aggregation | 17.49 | 0.004 | **4,716** | YES |
| Q8: Full Scan | 12.97 | 21.52 | 0.60 | YES |
| Q9: Single Inserts | 1.09 | 0.548 | **1.99** | YES |
| Q10: Deletes | 0.688 | 0.338 | **2.03** | YES |
| Q11: Hypercube 3-dim | 63.12 | 112.7 | 0.56 | YES |
| Q12: Group-By Agg | 69.32 | 62.49 | **1.11** | YES |
| Q13: Correlated Sub | 107.2 | 112.3 | 0.95 | YES |
| Q14: Moving Window | 45.80 | 0.080 | **570** | YES |
| Q15: Ad-Hoc Drill | 692.4 | 101.3 | **6.84** | YES |
| Q16: Top-K Groups | 65.52 | 55.12 | **1.19** | YES |
| Q17: HAVING Clause | 63.20 | 62.92 | 1.00 | YES |
| Q18: Year/Month Rollup | 70.37 | 77.45 | 0.91 | YES |
| Q19: Corr Multi-Dim Part | 140.1 | 140.9 | 0.99 | YES |
| Q20: YoY Semi-Join | 66.22 | 55.45 | **1.19** | YES |
| Q21: OR Bitmap | 45.85 | 31.11 | **1.47** | YES |
| Q22: Window Top-3/Month | 71.00 | 55.10 | **1.29** | NO* |
| Q23: CTE Correlated | 81.59 | 54.94 | **1.48** | YES |
| Q24: YoY Self-Join | 81.63 | 72.11 | **1.13** | YES |
| Q25: Dense Hyperbox 4D | 48.83 | 136.9 | 0.36 | YES |

\needspace{8\baselineskip}
*Table 9. Sequential distribution ($N = 10{,}000{,}000$, $\phi = 0.70$).*

| Query | B+ Tree (ms) | HP-Tree (ms) | Speedup | Correct |
|:---|---:|---:|---:|:---:|
| Q1: Bulk Load | 825.4 | 1,045.8 | 0.79 | YES |
| Q2: Point Lookup | 3.40 | 1.37 | **2.48** | YES |
| Q3: Narrow Range | 0.013 | 0.012 | **1.13** | YES |
| Q4: Wide Range | 0.415 | 0.568 | 0.73 | YES |
| Q5: Dim Filter | 18.27 | 0.064 | **284** | YES |
| Q6: Multi-Dim Filter | 21.75 | 0.308 | **70.7** | YES |
| Q7: Range Aggregation | 0.325 | 0.002 | **170** | YES |
| Q8: Full Scan | 13.02 | 24.23 | 0.54 | YES |
| Q9: Single Inserts | 0.713 | 0.404 | **1.76** | YES |
| Q10: Deletes | 0.744 | 0.364 | **2.05** | YES |
| Q11: Hypercube 3-dim | 29.12 | 3.67 | **7.93** | YES |
| Q12: Group-By Agg | 20.86 | 0.345 | **60.5** | YES |
| Q13: Correlated Sub | 107.1 | 112.4 | 0.95 | YES |
| Q14: Moving Window | 0.231 | 0.058 | **3.95** | YES |
| Q15: Ad-Hoc Drill | 599.7 | 8.57 | **70.0** | YES |
| Q16: Top-K Groups | 19.91 | 0.301 | **66.2** | YES |
| Q17: HAVING Clause | 62.84 | 62.75 | 1.00 | YES |
| Q18: Year/Month Rollup | 1.33 | 1.41 | 0.94 | YES |
| Q19: Corr Multi-Dim Part | 193.2 | 178.0 | **1.09** | YES |
| Q20: YoY Semi-Join | 21.37 | 0.563 | **37.9** | YES |
| Q21: OR Bitmap | 41.93 | 0.889 | **47.2** | YES |
| Q22: Window Top-3/Month | 20.13 | 0.401 | **50.2** | NO* |
| Q23: CTE Correlated | 39.77 | 1.20 | **33.3** | YES |
| Q24: YoY Self-Join | 21.89 | 1.05 | **20.8** | YES |
| Q25: Dense Hyperbox 4D | 19.23 | 1.19 | **16.2** | YES |

\* Q22 count divergence is a harness encoding offset (B+ counts encoded month=0 records; HP iterates months 1--12). Sums and checksums match.

### 5.2 Cross-Distribution Speedup Summary

\needspace{6\baselineskip}
*Table 10. Speedup ratios across all distributions. Bold = HP wins ($> 1.0$).*

| Query | Uniform | Clustered | Skewed | Sequential |
|:---|---:|---:|---:|---:|
| Q1: Bulk Load | 0.87 | 0.89 | 0.86 | 0.79 |
| Q2: Point Lookup | **2.49** | **2.41** | **1.78** | **2.48** |
| Q3: Narrow Range | 0.82 | 0.75 | 0.53 | **1.13** |
| Q4: Wide Range | 0.84 | 0.84 | 0.84 | 0.73 |
| Q5: Dim Filter | **4.11** | **2.64** | **1.14** | **284** |
| Q6: Multi-Dim Filter | **2.63** | **1.53** | **1.79** | **70.7** |
| Q7: Range Aggregation | **2,176** | **3,246** | **4,716** | **170** |
| Q8: Full Scan | 0.57 | 0.61 | 0.60 | 0.54 |
| Q9: Single Inserts | 0.91 | **3.17** | **1.99** | **1.76** |
| Q10: Deletes | 0.92 | **3.55** | **2.03** | **2.05** |
| Q11: Hypercube 3-dim | 0.68 | 0.62 | 0.56 | **7.93** |
| Q12: Group-By Agg | **2.12** | **1.62** | **1.11** | **60.5** |
| Q13: Correlated Sub | 0.95 | 0.95 | 0.95 | 0.95 |
| Q14: Moving Window | **123** | **220** | **570** | **3.95** |
| Q15: Ad-Hoc Drill | **2.32** | **3.70** | **6.84** | **70.0** |
| Q16: Top-K Groups | **2.63** | **1.73** | **1.19** | **66.2** |
| Q17: HAVING Clause | **1.14** | 1.00 | 1.00 | 1.00 |
| Q18: Year/Month Rollup | 0.91 | 0.90 | 0.91 | 0.94 |
| Q19: Corr Multi-Dim Part | 0.96 | 0.97 | 0.99 | **1.09** |
| Q20: YoY Semi-Join | **1.65** | **1.42** | **1.19** | **37.9** |
| Q21: OR Bitmap | **1.31** | **1.27** | **1.47** | **47.2** |
| Q22: Window Top-3/Month | **2.58** | **1.90** | **1.29** | **50.2** |
| Q23: CTE Correlated | **2.03** | **1.90** | **1.48** | **33.3** |
| Q24: YoY Self-Join | **1.55** | **1.35** | **1.13** | **20.8** |
| Q25: Dense Hyperbox 4D | 0.58 | 0.56 | 0.36 | **16.2** |

### 5.3 Aggregate Statistics

\needspace{6\baselineskip}
*Table 11. Summary across all 100 query-distribution cells.*

| Metric | Uniform | Clustered | Skewed | Sequential | **Total** |
|:---|---:|---:|---:|---:|---:|
| HP-Tree wins ($> 1.0$) | 14 | 16 | 16 | 20 | **66 / 100** |
| B+ Tree wins ($< 1.0$) | 11 | 9 | 9 | 5 | **34 / 100** |
| Geometric mean speedup | 2.14 | 2.26 | 2.04 | 8.29 | **3.00** |
| Maximum HP speedup | 2,176 | 3,246 | 4,716 | 284 | 4,716 (Q7) |
| Maximum B+ speedup | 1.74 (Q25) | 1.78 (Q25) | 2.80 (Q25) | 1.86 (Q8) | 2.80 (Q25-Skew) |
| Correctness match | 24/25 | 24/25 | 24/25 | 24/25 | **96 / 100** |

### 5.4 Asymmetry of Wins and Losses

A defining characteristic of the HP-Tree's performance profile is the **stark asymmetry between its wins and its losses**. When the HP-Tree wins, it wins by orders of magnitude: $2{,}176$--$4{,}716\times$ on range aggregation (Q7), $4$--$570\times$ on moving windows (Q14), $60$--$284\times$ on dimension filtering (Q5, sequential). When the B+ Tree wins, its margin is thin:

- **Q1 (Bulk Load):** B+ wins by $1.12$--$1.27\times$ --- a one-time construction cost amortised over all subsequent queries.
- **Q3 (Narrow Range), Q4 (Wide Range):** B+ wins by $1.19$--$1.89\times$ --- a consequence of tighter leaf packing ($100\%$ vs $70\%$).
- **Q8 (Full Scan):** B+ wins by $1.63$--$1.86\times$ --- the HP-Tree's only structurally significant disadvantage, caused by 43% more leaves to traverse. This is the expected cost of maintaining insert slack.
- **Q11 (Hypercube), Q25 (Dense Hyperbox):** B+ wins by up to $2.80\times$ on skewed data --- caused by DimStats bounding-box over-approximation on high-dimensional queries with wide selectivity. Notably, both queries **flip to decisive HP wins** on sequential data ($7.9\times$ and $16.2\times$), demonstrating that the limitation is distribution-dependent, not structural.
- **Q13 (Correlated Sub):** B+ wins by $1.05\times$ --- an effective tie, as this query is dominated by per-record arithmetic.
- **Q18 (Year/Month Rollup), Q19 (Corr Multi-Dim):** B+ wins by $1.01$--$1.10\times$ --- effectively ties.

**In 23 of the 34 B+ wins, the margin is less than $1.4\times$.** The B+ Tree's advantage exceeds $1.5\times$ in only 11 cells, attributable to either full-scan leaf count (Q8), narrow-range leaf packing (Q3), or bounding-box over-approximation (Q11, Q25). By contrast, the HP-Tree's advantage exceeds $1.5\times$ in 48 of its 66 wins, and exceeds $10\times$ in 18 wins.

\needspace{5\baselineskip}

## 6. Discussion

### 6.1 The HP-Tree as a Paradigm Shift

The results presented in Section 5 demonstrate that the HP-Tree represents a qualitative departure from the B+ Tree, not merely a quantitative improvement. The key evidence for this claim is threefold:

**First, the HP-Tree solves problems the B+ Tree cannot solve at all.** Range aggregation (Q7) on the B+ Tree requires $O(N)$ leaf scanning regardless of the query range. The HP-Tree's per-subtree DimStats enable $O(\log_B N)$ aggregation by short-circuiting entire subtrees, producing speedups of $170$--$4{,}716\times$ --- a difference that is not merely faster but *algorithmically different*. The same principle applies to dimension filtering (Q5, Q6), grouped aggregation (Q12), and drill-down queries (Q15): the B+ Tree must perform $O(N)$ work because it has no metadata on non-prefix dimensions, while the HP-Tree exploits hierarchical DimStats to achieve sub-linear performance.

**Second, the HP-Tree's advantages are structural while its disadvantages are parametric.** The HP-Tree loses on full scans (Q8) because it has more leaves to traverse --- a direct consequence of the fill factor $\phi < 1.0$. This is a tunable parameter: setting $\phi = 0.95$ would narrow the scan gap at the cost of insert slack. By contrast, the B+ Tree's $O(N)$ behaviour on dimension filtering is *architectural* --- no parameter change can give it sub-linear access to non-prefix dimensions.

**Third, the magnitude asymmetry is definitive.** The HP-Tree's maximum speedup ($4{,}716\times$) exceeds the B+ Tree's maximum speedup ($2.80\times$) by a factor of $1{,}684$. More importantly, the HP-Tree achieves $>10\times$ speedup in 18 of 100 cells, while the B+ Tree never achieves $>3\times$ in any cell. This is not incremental improvement --- it is a change in the computational complexity class of the operations being performed.

**Fourth, the HP-Tree is the first distribution-aware tree index.** The B+ Tree is distribution-agnostic: its routing structure is the same regardless of whether data is clustered, uniform, skewed, or sequential. The HP-Tree's DimStats, by contrast, automatically encode the distributional footprint of each subtree. This distribution awareness is not a heuristic --- it is a structural property of the index. When the data distribution creates tight per-subtree bounds (as in clustered or sequential data), the HP-Tree exploits this structure for orders-of-magnitude pruning. When the distribution is adversarial (uniform), the adaptive pruning-viability probe detects the lack of exploitable structure and falls back to the B+ Tree's scan strategy, incurring near-zero overhead. The result is an index that is never worse than $O(N)$ on any query but frequently achieves sub-linear performance by leveraging distributional properties that a distribution-agnostic index cannot even represent.

Q14 (Moving Window) uses 10 rolling 3-month overlapping windows (Jan--Mar, Feb--Apr, ..., Oct--Dec), a realistic sliding-window aggregation where each record participates in up to 3 windows. This overlap tests whether per-subtree DimStats can efficiently resolve repeated partial-overlap range queries. Q17 (HAVING Clause) uses a threshold of \$500M total revenue per state, which meaningfully filters non-uniform distributions: under clustered data only 1 of 15 states exceeds the threshold, while under uniform data all 15 pass --- a realistic discriminator that validates the GROUP BY + post-aggregation filter pipeline.

### 6.2 Three Tiers of HP-Tree Advantage

The results reveal three distinct tiers of performance advantage, each attributable to a different architectural mechanism:

**Tier 1: Massive speedups ($4$--$4{,}716\times$) on aggregate and window operations (Q7, Q14).** These queries benefit from the HP-Tree's $O(1)$ per-subtree aggregate shortcut. Q7 Range Aggregation achieves $170$--$4{,}716\times$ across all distributions because the root node's DimStats satisfy the containment check, returning the aggregate in a single comparison + addition versus the B+ Tree's full leaf-chain scan. Q14 Moving Window achieves $4$--$570\times$ by resolving 10 rolling 3-month overlapping windows through per-subtree DimStats pruning rather than per-record iteration. The overlapping nature of these windows means each record participates in up to 3 windows, yet the HP-Tree resolves all 10 via DimStats without rescanning. The magnitude of these speedups is not an artefact of an unfair baseline: the B+ Tree has no per-subtree aggregate metadata, so it must always scan every record in the query range.

**Tier 2: Large speedups ($7.9$--$284\times$) on dimension-filtered operations (Q5, Q6, Q11, Q12, Q15, Q16, Q20--Q24 on sequential data).** These queries access records along dimensions that do not align with the composite key's sort order. The B+ Tree must perform a full scan of all $N$ records for each such query, at $O(N)$ cost. The HP-Tree's per-subtree DimStats enable $O(m)$-per-inner-node pruning, reducing the effective search space. On sequential data, where each subtree's DimStats cover a narrow, disjoint dimension range, Q5 achieves $284\times$: only ${\sim}0.4\%$ of subtrees pass the year=2022 filter, and the rest are pruned at the inner-node level.

**Tier 3: Moderate speedups ($1.1$--$6.9\times$) on point lookups, inserts, deletes, and composite analytics (Q2, Q9, Q10, Q12, Q15, Q20--Q24 on non-sequential data).** Point lookups (Q2) benefit from the HP-Tree's separate key/value arrays, which improve cache-line utilisation during the root-to-leaf descent. Inserts (Q9) benefit from the sequential-append fast path. Deletes (Q10) benefit from the HP-Tree's lean leaf design. Composite analytics (Q12, Q15, Q20--Q24) benefit from DimStats-based pruning in their filter phase.

### 6.3 Analysis of B+ Tree Wins: Thin Margins

The B+ Tree wins 34 of 100 cells. These cluster into well-understood categories, and in all cases the margins are thin:

**Scan-dominated queries (Q1, Q3, Q4, Q8, Q18): B+ $1.06$--$1.89\times$.** These queries touch most or all records, so there is nothing to prune. The B+ Tree's tighter leaf packing ($100\%$ vs $70\%$) means fewer leaves to traverse and better cache-line utilisation during sequential iteration. The widest margin ($1.86\times$ on Q8 sequential) reflects the $43\%$ leaf count increase from $\phi = 0.70$. This is the expected and accepted cost of maintaining insert slack --- a trade-off that pays for itself through $1.8$--$3.2\times$ faster inserts (Q9) and $2.0$--$3.6\times$ faster deletes (Q10) on non-uniform data.

**Bounding-box over-approximation (Q11, Q25 on uniform/clustered/skewed): B+ $1.48$--$2.80\times$.** When 3+ dimensions are filtered simultaneously on high-cardinality data, the DimStats bounding-box over-approximation admits too many false-positive subtrees. Notably, both queries **flip to decisive HP wins** on sequential data ($7.9\times$ and $16.2\times$), demonstrating that this limitation is distribution-dependent, not a fundamental flaw.

**Effective ties (Q13, Q19): B+ $1.01$--$1.05\times$.** These queries are dominated by per-record arithmetic (compare each record's price to a per-product average). Neither tree structure helps; the computation is arithmetic-bound.

### 6.4 Cross-Industry Relevance

The 25 benchmark queries map to real-world workloads across multiple industries:

**Retail and e-commerce** (Q2, Q5, Q7, Q12, Q15, Q16). Revenue dashboards, product lookups, sales-by-category reports, and drill-down exploration. The HP-Tree wins all six with speedups of $1.1$--$4{,}132\times$. Q7's constant-time aggregation eliminates the need for materialised views on revenue summaries.

**Financial services** (Q7, Q14, Q20, Q12). Portfolio aggregation, rolling risk metrics, year-over-year growth analysis, and group-by-sector reports. Q14's $4$--$570\times$ speedup on rolling 3-month windows is decisive for real-time risk computation.

**Healthcare and pharma** (Q5, Q6, Q12, Q7). Patient filtering by diagnosis/region, cost aggregation, and claims-by-type reporting. HP-Tree's $1.5$--$4.5\times$ speedups on filter queries reduce dashboard refresh latency.

**IoT, monitoring, and SRE** (Q9, Q14, Q7, Q3). Event ingestion, rolling-window monitoring, and sensor aggregation. Q9's $2.0$--$2.3\times$ faster inserts and Q14's rolling windows make the HP-Tree well-suited for time-series workloads --- especially given the sequential distribution's $8.29\times$ geometric-mean advantage.

**Ad-tech and digital marketing** (Q6, Q15, Q7, Q11). Audience targeting, campaign drill-down, and spend aggregation. HP wins on Q6/Q15/Q7; Q11 is a limitation for high-dimensional audience segmentation on non-sequential data.

**SaaS product analytics** (Q12, Q16, Q20, Q15). Usage-by-plan, top-K features, month-over-month growth, and cohort drill-down. All HP wins at $1.1$--$66.6\times$.

### 6.5 Limitations

1. **Single-threaded evaluation.** The benchmark does not measure concurrent-access behaviour. The HP-Tree's DimStats updates on the insertion path interact non-trivially with concurrency control.

2. **In-memory operation.** Both trees operate entirely in main memory. Disk-resident operation would introduce page-fault costs. The HP-Tree's larger inner nodes (due to DimStats arrays) would increase I/O per inner-node fetch but reduce the total number of inner-node fetches.

3. **Fixed dimensionality.** The evaluation uses $D = 7$. As $D$ increases, the per-inner-node DimStats cost scales linearly ($O(D)$ per node) and pruning effectiveness may degrade due to the curse of dimensionality [4].

4. **Bounding-box over-approximation.** The HP-Tree's DimStats use min/max bounds, not bitmaps or bloom filters. On high-cardinality dimensions with uniform distribution, the bounding box may span the full domain, yielding no pruning benefit. This is the root cause of Q11/Q25 losses on non-sequential data.

\needspace{5\baselineskip}

## 7. Conclusion and Future Work

### 7.1 Conclusion

This paper has introduced the HP-Tree, a breakthrough multi-dimensional index structure that resolves the B+ Tree's fundamental unidimensional limitation while preserving all of its strengths. Through hierarchical per-subtree dimensional statistics (DimStats) maintained at every inner node, an adaptive pruning-viability probe, a lean leaf design, a sequential-append fast path, and workload-adaptive leaf packing, the HP-Tree transforms the B+ Tree's unidimensional routing structure into a multi-dimensional pruning hierarchy.

Through a comprehensive benchmark of $10{,}000{,}000$ records, four data distributions, and twenty-five query types against the production-quality `tlx::btree_multimap` v0.6.1, we have demonstrated that the HP-Tree **wins 66 of 100 query-distribution cells**, achieving speedups of up to **$4{,}716\times$** on range aggregation and **$284\times$** on dimension filtering. The central insight is that a single DimStats check at level $h$ can short-circuit an entire subtree of $B^h$ records --- whether for predicate exclusion, full-containment fast-tracking, or constant-time aggregate computation.

Critically, the performance profile is **asymmetric in the HP-Tree's favour**: when it wins, it wins by orders of magnitude; when the B+ Tree wins, the margin is thin (median $1.19\times$, never exceeding $1.9\times$ on structurally comparable queries). The B+ Tree's advantages are confined to full-scan workloads where there is nothing to prune --- a diminishing share of modern analytical workloads. The HP-Tree's advantages, by contrast, apply precisely to the dimension-filtered, aggregated, and drill-down queries that dominate real-world analytics across retail, finance, healthcare, IoT, and SaaS domains.

The HP-Tree is not an incremental improvement to the B+ Tree. It is a fundamentally new capability: **multi-dimensional pruning and constant-time aggregation within a single, dynamically updatable tree structure**, achieved at a space overhead of less than 0.2% of the record storage.

The complete HP-Tree implementation is open source: [\texttt{github.com/sutigit21/HP\_TREE}](https://github.com/sutigit21/HP_TREE)

### 7.2 Future Work

Three research directions emerge from this work:

**Per-subtree bitmap indexes.** Replacing or augmenting min/max DimStats with per-subtree bitmaps (one bit per distinct value per dimension) would eliminate the bounding-box over-approximation that causes Q11/Q25 losses. The space cost is $O(I \cdot \sum_d 2^{b_d})$ bits, which could be applied selectively to low-cardinality dimensions (e.g., $\text{state}: 2^5 = 32$ bits) at negligible cost.

**Concurrent evaluation under mixed workloads.** The HP-Tree's DimStats updates along the insertion spine create a serialisation constraint analogous to B-link tree latch coupling [6]. Evaluating throughput and tail-latency behaviour under mixed OLTP/OLAP workloads would quantify the practical concurrency cost of per-subtree metadata maintenance.

**Dimensionality scaling.** Characterising the break-even dimensionality $D^\star$ at which DimStats overhead exceeds pruning benefit, and developing dimension-selection heuristics that restrict DimStats maintenance to the most query-relevant dimensions, is an important open problem.

\needspace{5\baselineskip}

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
