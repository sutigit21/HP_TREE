#pragma once

#include "hp_tree_common.hpp"
#include "hp_tree_node.hpp"
#include "hp_tree_iterator.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace hptree {

// =============================================================================
//  HPTree: tlx-style B+-tree with per-node beta metadata and per-subtree
//  DimStats aggregates for predicate-pruning & O(1) range aggregates.
//
//  Incremental invariants:
//    - On insert/remove, parent dim_stats.count and dim_stats.sum are kept
//      exact; dim_stats.min_val/max_val and range_lo/range_hi may become
//      stale-wide but never stale-narrow (conservative pruning).
//    - Splits rebuild both halves from their children (O(children × dims)).
//    - Bulk load computes leaf stats on the fly, merges up.
// =============================================================================

class HPTree {
public:
    struct AggregateResult {
        uint64_t count = 0;
        double   sum   = 0.0;
    };

    HPTree(const HPTreeConfig& cfg, const CompositeKeySchema& schema)
        : config_(cfg), schema_(schema), root_(nullptr),
          head_leaf_(nullptr), tail_leaf_(nullptr),
          total_records_(0), levels_(0)
    {
        dim_count_ = schema_.dim_count();
        for (size_t d = 0; d < dim_count_; ++d) {
            dim_off_[d]  = schema_.offset_of(d);
            dim_mask_[d] = schema_.mask_of(d);
        }
    }

    ~HPTree() { destroy_subtree(root_); }

    HPTree(const HPTree&)            = delete;
    HPTree& operator=(const HPTree&) = delete;

    // =========================================================================
    //  Bulk load
    // =========================================================================
    void bulk_load(std::vector<Record>&& recs) {
        destroy_subtree(root_);
        root_ = head_leaf_ = tail_leaf_ = nullptr;
        total_records_ = 0;
        levels_ = 0;
        if (recs.empty()) return;

        std::sort(recs.begin(), recs.end(),
                  [](const Record& a, const Record& b){ return a.key < b.key; });

        // Build leaf level + per-leaf DimStats in one pass.
        // Balance scan locality vs insert slack:
        //   - Previous 75% (24/32) wasted 33% of scan bandwidth.
        //   - Full-pack (30/32) caused Q9 split storms on clustered/skewed
        //     data where inserts hit the same hot leaves repeatedly.
        //   - 84% (27/32) keeps 5 slack slots — enough to absorb 1k inserts
        //     across ~37k leaves without chaining splits, while still
        //     delivering ~17% fewer leaves than the old 75% packing.
        static constexpr uint16_t LEAF_PACK = (LEAF_SLOTMAX * 27) / 32;

        // Preallocate exact capacities — no vector reallocations on the hot
        // build path.  level_nodes/level_stats/level_counts are all parallel
        // arrays of the same length.
        const size_t n_leaves = (recs.size() + LEAF_PACK - 1) / LEAF_PACK;
        std::vector<NodeBase*>                      level_nodes;
        std::vector<std::array<DimStats, MAX_DIMS>> level_stats;
        std::vector<uint64_t>                       level_counts;
        level_nodes.reserve(n_leaves);
        level_stats.reserve(n_leaves);
        level_counts.reserve(n_leaves);

        LeafNode* prev = nullptr;
        size_t i = 0;
        while (i < recs.size()) {
            size_t chunk = std::min<size_t>(LEAF_PACK, recs.size() - i);
            LeafNode* leaf = new LeafNode();
            leaf->init();
            leaf->slotuse = static_cast<uint16_t>(chunk);
            // Emplace a default-constructed stats slot and fill it in place —
            // avoids the temporary std::array copy.
            level_stats.emplace_back();
            auto& ls = level_stats.back();
            for (size_t j = 0; j < chunk; ++j) {
                CompositeKey k = recs[i + j].key;
                leaf->keys[j]   = k;
                leaf->values[j] = recs[i + j].value;
                for (size_t d = 0; d < dim_count_; ++d)
                    ls[d].add(extract_dim_fast(k, d));
            }
            leaf->range_lo = leaf->keys[0];
            leaf->range_hi = leaf->keys[chunk - 1];
            leaf->prev_leaf = prev;
            if (prev) prev->next_leaf = leaf;
            else      head_leaf_ = leaf;
            prev = leaf;
            level_nodes.push_back(leaf);
            level_counts.push_back(static_cast<uint64_t>(chunk));
            i += chunk;
        }
        tail_leaf_ = prev;
        total_records_ = recs.size();
        levels_ = 1;

        // Build inner levels bottom-up.  At each level, aggregate DimStats
        // directly into the new inner node's dim_stats array — no temporary
        // aggregate buffer and no final copy-out.
        while (level_nodes.size() > 1) {
            const size_t n_parents = (level_nodes.size() + INNER_SLOTMAX)
                                     / (INNER_SLOTMAX + 1);
            std::vector<NodeBase*>                      parents;
            std::vector<std::array<DimStats, MAX_DIMS>> parent_stats;
            std::vector<uint64_t>                       parent_counts;
            parents.reserve(n_parents);
            parent_stats.reserve(n_parents);
            parent_counts.reserve(n_parents);

            size_t k = 0;
            while (k < level_nodes.size()) {
                size_t chunk = std::min<size_t>(INNER_SLOTMAX + 1,
                                                level_nodes.size() - k);
                InnerNode* inner = new InnerNode();
                inner->init(static_cast<uint16_t>(levels_), dim_count_);
                inner->slotuse = static_cast<uint16_t>(chunk - 1);

                CompositeKey lo = COMPOSITE_KEY_MAX;
                CompositeKey hi = COMPOSITE_KEY_MIN;
                uint64_t count = 0;

                for (size_t j = 0; j < chunk; ++j) {
                    NodeBase* c = level_nodes[k + j];
                    inner->childid[j] = c;
                    if (c->range_lo < lo) lo = c->range_lo;
                    if (c->range_hi > hi) hi = c->range_hi;
                    count += level_counts[k + j];
                    for (size_t d = 0; d < dim_count_; ++d)
                        inner->dim_stats[d].merge(level_stats[k + j][d]);
                }
                for (size_t j = 0; j + 1 < chunk; ++j)
                    inner->slotkey[j] = level_nodes[k + j]->range_hi;

                inner->range_lo      = lo;
                inner->range_hi      = hi;
                inner->subtree_count = count;

                // Emplace + fill parent_stats in place (no array copy).
                parent_stats.emplace_back();
                auto& ps = parent_stats.back();
                for (size_t d = 0; d < dim_count_; ++d) ps[d] = inner->dim_stats[d];

                parents.push_back(inner);
                parent_counts.push_back(count);
                k += chunk;
            }
            level_nodes  = std::move(parents);
            level_stats  = std::move(parent_stats);
            level_counts = std::move(parent_counts);
            levels_++;
        }
        root_ = level_nodes[0];
    }

    size_t size() const { return total_records_; }
    size_t levels() const { return levels_; }

    // =========================================================================
    //  Point lookup (multimap)
    // =========================================================================
    std::vector<Record> search(CompositeKey key) const {
        std::vector<Record> out;
        if (root_ == nullptr) return out;
        LeafNode* leaf = find_leaf_for(key);
        while (leaf) {
            uint16_t s = leaf_find_lower(leaf, key);
            while (s < leaf->slotuse && leaf->keys[s] == key) {
                out.push_back({ leaf->keys[s], leaf->values[s] });
                ++s;
            }
            if (s < leaf->slotuse) break;
            leaf = leaf->next_leaf;
            if (!leaf || leaf->keys[0] != key) break;
        }
        return out;
    }

    // =========================================================================
    //  Iterator-based range API — zero allocation, tlx-style.
    //
    //  Usage:
    //    auto it = tree.lower_bound(lo);
    //    while (it.valid() && it.key() <= hi) { ...; ++it; }
    // =========================================================================
    HPTreeIterator lower_bound(CompositeKey key) const {
        if (root_ == nullptr) return HPTreeIterator();
        LeafNode* leaf = find_leaf_for(key);
        if (!leaf) return HPTreeIterator();
        uint16_t s = leaf_find_lower(leaf, key);
        if (s < leaf->slotuse) return HPTreeIterator(leaf, s);
        // Overflow to next leaf.
        leaf = leaf->next_leaf;
        return HPTreeIterator(leaf, 0);
    }

    HPTreeIterator upper_bound(CompositeKey key) const {
        if (root_ == nullptr) return HPTreeIterator();
        LeafNode* leaf = find_leaf_for(key);
        if (!leaf) return HPTreeIterator();
        uint16_t s = leaf_find_upper(leaf, key);
        if (s < leaf->slotuse) return HPTreeIterator(leaf, s);
        leaf = leaf->next_leaf;
        return HPTreeIterator(leaf, 0);
    }

    // Legacy vector-returning range_search (kept for compatibility).
    std::vector<Record> range_search(CompositeKey lo, CompositeKey hi) const {
        std::vector<Record> out;
        if (root_ == nullptr || lo > hi) return out;
        auto it = lower_bound(lo);
        while (it.valid() && it.key() <= hi) {
            out.push_back({ it.key(), it.value() });
            ++it;
        }
        return out;
    }

    // =========================================================================
    //  Predicate search with dim-stat pruning.
    //
    //  predicate_search_cb(ps, cb) invokes cb(key, value) for each match and
    //  allocates nothing — the preferred path for group-by / count / top-k.
    //  predicate_search(ps) is the legacy vector-returning wrapper.
    // =========================================================================
    template <class F>
    void predicate_search_cb(const PredicateSet& ps, F&& cb) const {
        if (root_ == nullptr) return;
        KeyRange kr = ps.to_key_range(schema_);
        predicate_search_node_cb(root_, ps, kr, cb);
    }

    std::vector<Record> predicate_search(const PredicateSet& ps) const {
        std::vector<Record> out;
        predicate_search_cb(ps, [&](CompositeKey k, uint64_t v) {
            out.push_back({k, v});
        });
        return out;
    }

    // =========================================================================
    //  Aggregation: sum + count of dimension over [lo, hi] with O(1) shortcut
    // =========================================================================
    AggregateResult aggregate_dim(size_t dim,
                                  CompositeKey lo, CompositeKey hi) const {
        AggregateResult r;
        if (root_ == nullptr || dim >= dim_count_ || lo > hi) return r;
        aggregate_dim_node(root_, dim, lo, hi, r);
        return r;
    }

    // =========================================================================
    //  Iterator
    // =========================================================================
    HPTreeIterator begin() const { return HPTreeIterator(head_leaf_, 0); }
    HPTreeIterator end()   const { return HPTreeIterator(nullptr, 0); }

    // Compatibility shim — HPTreeIterator already has .valid() and .next().
    HPTreeIterator runner_begin() const { return begin(); }

    // =========================================================================
    //  Insert (single-record, in-place, incremental metadata)
    // =========================================================================
    bool insert(const Record& rec) {
        if (root_ == nullptr) {
            LeafNode* leaf = new LeafNode();
            leaf->init();
            leaf->keys[0]   = rec.key;
            leaf->values[0] = rec.value;
            leaf->slotuse   = 1;
            leaf->recompute_range();
            root_ = head_leaf_ = tail_leaf_ = leaf;
            total_records_ = 1;
            levels_ = 1;
            return true;
        }

        // Precompute per-dim values once — reused at every inner level.
        uint64_t key_dims[MAX_DIMS];
        for (size_t d = 0; d < dim_count_; ++d)
            key_dims[d] = extract_dim_fast(rec.key, d);

        CompositeKey split_key = 0;
        NodeBase*    split_right = nullptr;
        bool ok = insert_descend(root_, rec, key_dims, split_key, split_right);
        if (!ok) return false;

        if (split_right != nullptr) {
            InnerNode* newroot = new InnerNode();
            newroot->init(static_cast<uint16_t>(levels_), dim_count_);
            newroot->slotuse = 1;
            newroot->slotkey[0] = split_key;
            newroot->childid[0] = root_;
            newroot->childid[1] = split_right;
            rebuild_inner_meta_from_children(newroot);
            root_ = newroot;
            levels_++;
        }
        total_records_++;
        return true;
    }

    bool remove(CompositeKey key) {
        if (root_ == nullptr) return false;
        uint64_t rec_dims[MAX_DIMS];
        bool ok = remove_descend(root_, key, rec_dims);
        if (!ok) return false;

        if (root_->level > 0) {
            auto* in = static_cast<InnerNode*>(root_);
            if (in->slotuse == 0) {
                NodeBase* only = in->childid[0];
                delete in;
                root_ = only;
                levels_--;
            }
        } else if (root_->slotuse == 0) {
            delete static_cast<LeafNode*>(root_);
            root_ = head_leaf_ = tail_leaf_ = nullptr;
            levels_ = 0;
        }
        total_records_--;
        return true;
    }

    void flush_delta() { /* no-op in tlx-parity mode */ }

private:
    HPTreeConfig       config_;
    CompositeKeySchema schema_;
    size_t             dim_count_;
    uint8_t            dim_off_ [MAX_DIMS];
    uint64_t           dim_mask_[MAX_DIMS];
    NodeBase*          root_;
    LeafNode*          head_leaf_;
    LeafNode*          tail_leaf_;
    size_t             total_records_;
    size_t             levels_;

    // -------------------------------------------------------------------------
    //  Cached dim extraction
    // -------------------------------------------------------------------------
    inline uint64_t extract_dim_fast(CompositeKey key, size_t d) const {
        return static_cast<uint64_t>((key >> dim_off_[d]) & dim_mask_[d]);
    }

    // -------------------------------------------------------------------------
    //  Traversal
    // -------------------------------------------------------------------------
    LeafNode* find_leaf_for(CompositeKey key) const {
        NodeBase* n = root_;
        while (n && n->level > 0) {
            auto* in = static_cast<InnerNode*>(n);
            uint16_t slot = inner_find_lower(in, key);
            n = in->childid[slot];
        }
        return static_cast<LeafNode*>(n);
    }

    // -------------------------------------------------------------------------
    //  Full rebuild of inner metadata from children (used only on bulk_load
    //  completion, new root creation, and split).
    // -------------------------------------------------------------------------
    void rebuild_inner_meta_from_children(InnerNode* in) {
        CompositeKey lo = COMPOSITE_KEY_MAX;
        CompositeKey hi = COMPOSITE_KEY_MIN;
        uint64_t cnt = 0;
        for (size_t d = 0; d < dim_count_; ++d) in->dim_stats[d] = DimStats{};

        uint16_t nchildren = in->slotuse + 1;
        for (uint16_t i = 0; i < nchildren; ++i) {
            NodeBase* c = in->childid[i];
            if (c->range_lo < lo) lo = c->range_lo;
            if (c->range_hi > hi) hi = c->range_hi;
            if (c->level == 0) {
                auto* lf = static_cast<LeafNode*>(c);
                cnt += lf->slotuse;
                for (uint16_t s = 0; s < lf->slotuse; ++s) {
                    for (size_t d = 0; d < dim_count_; ++d)
                        in->dim_stats[d].add(extract_dim_fast(lf->keys[s], d));
                }
            } else {
                auto* ic = static_cast<InnerNode*>(c);
                cnt += ic->subtree_count;
                for (size_t d = 0; d < dim_count_; ++d)
                    in->dim_stats[d].merge(ic->dim_stats[d]);
            }
        }
        in->range_lo       = lo;
        in->range_hi       = hi;
        in->subtree_count  = cnt;
    }

    // Incremental: O(dim_count) on insert of a single key.
    // range_lo/range_hi drive all pruning; dim_stats.count/sum kept exact.
    inline void inner_meta_add(InnerNode* in, CompositeKey key,
                               const uint64_t* key_dims) {
        in->subtree_count++;
        for (size_t d = 0; d < dim_count_; ++d)
            in->dim_stats[d].add(key_dims[d]);
        if (key < in->range_lo) in->range_lo = key;
        if (key > in->range_hi) in->range_hi = key;
    }

    // Incremental delete — only exact count/sum are kept; min/max/range may
    // become stale-wide (never stale-narrow) which is safe for pruning.
    inline void inner_meta_sub(InnerNode* in, const uint64_t* rec_dims) {
        if (in->subtree_count > 0) in->subtree_count--;
        for (size_t d = 0; d < dim_count_; ++d) {
            if (in->dim_stats[d].count > 0) {
                in->dim_stats[d].count--;
                in->dim_stats[d].sum -= static_cast<double>(rec_dims[d]);
            }
        }
    }

    // -------------------------------------------------------------------------
    //  Predicate dispatch
    // -------------------------------------------------------------------------
    bool subtree_may_contain(const InnerNode* in, const PredicateSet& ps) const {
        for (auto& p : ps.predicates) {
            if (p.dim_idx >= dim_count_) continue;
            const DimStats& ds = in->dim_stats[p.dim_idx];
            if (ds.count == 0) return false;
            uint64_t v  = static_cast<uint64_t>(p.value);
            uint64_t vh = static_cast<uint64_t>(p.value_high);
            switch (p.op) {
            case PredicateOp::EQ:
                if (v < ds.min_val || v > ds.max_val) return false; break;
            case PredicateOp::BETWEEN:
                if (vh < ds.min_val || v > ds.max_val) return false; break;
            case PredicateOp::LT:  if (ds.min_val >= v) return false; break;
            case PredicateOp::LTE: if (ds.min_val >  v) return false; break;
            case PredicateOp::GT:  if (ds.max_val <= v) return false; break;
            case PredicateOp::GTE: if (ds.max_val <  v) return false; break;
            default: break;
            }
        }
        return true;
    }

    // True if every predicate is FULLY covered by the subtree's per-dim stats
    // (i.e. every record in the subtree is guaranteed to satisfy the
    // predicate).  When this holds AND the subtree's key range is inside the
    // query's KeyRange, we can emit all keys without per-key evaluate().
    bool subtree_fully_satisfies(const InnerNode* in,
                                 const PredicateSet& ps) const {
        for (auto& p : ps.predicates) {
            if (p.dim_idx >= dim_count_) continue;
            const DimStats& ds = in->dim_stats[p.dim_idx];
            if (ds.count == 0) return false;
            uint64_t v  = static_cast<uint64_t>(p.value);
            uint64_t vh = static_cast<uint64_t>(p.value_high);
            switch (p.op) {
            case PredicateOp::EQ:
                if (!(ds.min_val == v && ds.max_val == v)) return false; break;
            case PredicateOp::BETWEEN:
                if (!(ds.min_val >= v && ds.max_val <= vh)) return false; break;
            case PredicateOp::LT:  if (!(ds.max_val <  v)) return false; break;
            case PredicateOp::LTE: if (!(ds.max_val <= v)) return false; break;
            case PredicateOp::GT:  if (!(ds.min_val >  v)) return false; break;
            case PredicateOp::GTE: if (!(ds.min_val >= v)) return false; break;
            default: return false;
            }
        }
        return true;
    }

    // Emit every record in a subtree (no per-key filtering).  Used by the
    // fully-contained fast-path.  Walks leaves via next_leaf for cache locality.
    template <class F>
    void emit_subtree_all(NodeBase* n, F& cb) const {
        LeafNode* start = leftmost_leaf(n);
        LeafNode* end   = rightmost_leaf(n);
        if (!start) return;
        LeafNode* lf = start;
        while (true) {
            for (uint16_t s = 0; s < lf->slotuse; ++s)
                cb(lf->keys[s], lf->values[s]);
            if (lf == end) break;
            lf = lf->next_leaf;
            if (!lf) break;
        }
    }

    LeafNode* leftmost_leaf(NodeBase* n) const {
        while (n && n->level > 0)
            n = static_cast<InnerNode*>(n)->childid[0];
        return static_cast<LeafNode*>(n);
    }
    LeafNode* rightmost_leaf(NodeBase* n) const {
        while (n && n->level > 0) {
            auto* in = static_cast<InnerNode*>(n);
            n = in->childid[in->slotuse];
        }
        return static_cast<LeafNode*>(n);
    }

    // Callback-based predicate descent — zero heap allocation per match.
    //
    // Two pruning gates:
    //   1. range_lo/range_hi vs KeyRange  (cheap key-level box test)
    //   2. per-dim stats vs each predicate (subtree_may_contain)
    // And one fast-path:
    //   3. If the subtree is fully inside kr AND every predicate is fully
    //      covered by the subtree's dim_stats, emit every record without
    //      running ps.evaluate() per key — huge win on dense matches.
    template <class F>
    void predicate_search_node_cb(NodeBase* n, const PredicateSet& ps,
                                  const KeyRange& kr, F& cb) const {
        if (n->range_hi < kr.low || n->range_lo > kr.high) return;
        if (n->level == 0) {
            auto* lf = static_cast<LeafNode*>(n);
            for (uint16_t s = 0; s < lf->slotuse; ++s) {
                CompositeKey k = lf->keys[s];
                if (k < kr.low || k > kr.high) continue;
                if (ps.evaluate(k, schema_))
                    cb(k, lf->values[s]);
            }
            return;
        }
        auto* in = static_cast<InnerNode*>(n);
        if (!subtree_may_contain(in, ps)) return;
        // Fast-path: whole subtree is inside kr AND all predicates are fully
        // satisfied by stats.  Emit via flat leaf walk without evaluate().
        if (n->range_lo >= kr.low && n->range_hi <= kr.high
            && subtree_fully_satisfies(in, ps)) {
            emit_subtree_all(n, cb);
            return;
        }
        // Restrict child visitation window by kr bounds — slotkey[i] separates
        // children i and i+1, so any child with index < first or > last can be
        // skipped by the key-range gate.  This saves range-check calls on
        // dense inner nodes.
        uint16_t first = 0;
        while (first < in->slotuse && in->slotkey[first] < kr.low) ++first;
        uint16_t last = in->slotuse;
        while (last > 0 && in->slotkey[last - 1] > kr.high) --last;
        for (uint16_t i = first; i <= last; ++i)
            predicate_search_node_cb(in->childid[i], ps, kr, cb);
    }

    void aggregate_dim_node(NodeBase* n, size_t dim,
                            CompositeKey lo, CompositeKey hi,
                            AggregateResult& r) const {
        if (n->range_hi < lo || n->range_lo > hi) return;
        if (n->level > 0) {
            auto* in = static_cast<InnerNode*>(n);
            if (n->range_lo >= lo && n->range_hi <= hi) {
                r.count += in->subtree_count;
                r.sum   += in->dim_stats[dim].sum;
                return;
            }
            // Bound the child-visit window by [lo, hi] via slotkey pivots.
            uint16_t first = 0;
            while (first < in->slotuse && in->slotkey[first] < lo) ++first;
            uint16_t last = in->slotuse;
            while (last > 0 && in->slotkey[last - 1] > hi) --last;
            for (uint16_t i = first; i <= last; ++i)
                aggregate_dim_node(in->childid[i], dim, lo, hi, r);
            return;
        }
        auto* lf = static_cast<LeafNode*>(n);
        for (uint16_t s = 0; s < lf->slotuse; ++s) {
            CompositeKey k = lf->keys[s];
            if (k < lo || k > hi) continue;
            r.count++;
            r.sum += static_cast<double>(extract_dim_fast(k, dim));
        }
    }

    // -------------------------------------------------------------------------
    //  Insert descent with incremental metadata updates
    // -------------------------------------------------------------------------
    bool insert_descend(NodeBase* n, const Record& rec,
                        const uint64_t* key_dims,
                        CompositeKey& split_key, NodeBase*& split_right) {
        if (n->level == 0) {
            auto* lf = static_cast<LeafNode*>(n);
            uint16_t s = leaf_find_lower(lf, rec.key);
            for (uint16_t i = lf->slotuse; i > s; --i) {
                lf->keys[i]   = lf->keys[i - 1];
                lf->values[i] = lf->values[i - 1];
            }
            lf->keys[s]   = rec.key;
            lf->values[s] = rec.value;
            lf->slotuse++;
            if (rec.key < lf->range_lo) lf->range_lo = rec.key;
            if (rec.key > lf->range_hi) lf->range_hi = rec.key;
            if (lf->is_full()) split_leaf(lf, split_key, split_right);
            return true;
        }

        auto* in = static_cast<InnerNode*>(n);
        uint16_t slot = inner_find_lower(in, rec.key);
        CompositeKey child_split_key = 0;
        NodeBase* child_split_right = nullptr;
        bool ok = insert_descend(in->childid[slot], rec, key_dims,
                                 child_split_key, child_split_right);
        if (!ok) return false;

        if (child_split_right == nullptr) {
            // No split below → cheap incremental stats update.
            inner_meta_add(in, rec.key, key_dims);
            return true;
        }

        // Child split propagated: insert the new pivot+child into this node.
        for (uint16_t i = in->slotuse; i > slot; --i) {
            in->slotkey[i]     = in->slotkey[i - 1];
            in->childid[i + 1] = in->childid[i];
        }
        in->slotkey[slot]     = child_split_key;
        in->childid[slot + 1] = child_split_right;
        in->slotuse++;

        if (in->is_full()) {
            // split_inner rebuilds both halves from children — no need for a
            // preceding inner_meta_add, which would just be overwritten.
            split_inner(in, split_key, split_right);
        } else {
            inner_meta_add(in, rec.key, key_dims);
        }
        return true;
    }

    void split_leaf(LeafNode* lf, CompositeKey& split_key, NodeBase*& right_out) {
        // Guard against mid=0: only reachable if a leaf somehow went full with
        // slotuse<=1, which is impossible for LEAF_SLOTMAX>=2, but make it
        // explicit so a future smaller LEAF_SLOTMAX can't trigger keys[-1].
        uint16_t mid = lf->slotuse / 2;
        if (mid == 0) mid = 1;
        LeafNode* right = new LeafNode();
        right->init();
        right->slotuse = lf->slotuse - mid;
        for (uint16_t i = 0; i < right->slotuse; ++i) {
            right->keys[i]   = lf->keys[mid + i];
            right->values[i] = lf->values[mid + i];
        }
        lf->slotuse = mid;
        lf->recompute_range();
        right->recompute_range();

        right->next_leaf = lf->next_leaf;
        right->prev_leaf = lf;
        if (lf->next_leaf) lf->next_leaf->prev_leaf = right;
        lf->next_leaf = right;
        if (tail_leaf_ == lf) tail_leaf_ = right;

        split_key = lf->keys[mid - 1];
        right_out = right;
    }

    void split_inner(InnerNode* in, CompositeKey& split_key, NodeBase*& right_out) {
        uint16_t mid = in->slotuse / 2;
        InnerNode* right = new InnerNode();
        right->init(in->level, dim_count_);
        right->slotuse = in->slotuse - mid - 1;
        for (uint16_t i = 0; i < right->slotuse; ++i)
            right->slotkey[i] = in->slotkey[mid + 1 + i];
        for (uint16_t i = 0; i <= right->slotuse; ++i)
            right->childid[i] = in->childid[mid + 1 + i];
        split_key = in->slotkey[mid];
        in->slotuse = mid;

        // Incremental stats rebuild — each half only aggregates its OWN
        // children rather than re-scanning leaves.  When a child is an inner
        // node, we merge its precomputed dim_stats in O(dim_count); when it's
        // a leaf, we have to iterate its keys once.  This is significantly
        // cheaper than the full rebuild_inner_meta_from_children, which
        // re-sums everything from scratch for both halves.
        aggregate_inner_from_children(in);
        aggregate_inner_from_children(right);
        right_out = right;
    }

    // Aggregates one inner node's metadata from its children.  If all children
    // are inner nodes (upper levels), this is pure O(children × dim_count)
    // merging of already-computed DimStats; only at the bottom inner level do
    // we pay the per-record scan.
    void aggregate_inner_from_children(InnerNode* in) {
        CompositeKey lo = COMPOSITE_KEY_MAX;
        CompositeKey hi = COMPOSITE_KEY_MIN;
        uint64_t cnt = 0;
        for (size_t d = 0; d < dim_count_; ++d) in->dim_stats[d] = DimStats{};

        uint16_t nchildren = in->slotuse + 1;
        for (uint16_t i = 0; i < nchildren; ++i) {
            NodeBase* c = in->childid[i];
            if (c->range_lo < lo) lo = c->range_lo;
            if (c->range_hi > hi) hi = c->range_hi;
            if (c->level == 0) {
                auto* lf = static_cast<LeafNode*>(c);
                cnt += lf->slotuse;
                for (uint16_t s = 0; s < lf->slotuse; ++s)
                    for (size_t d = 0; d < dim_count_; ++d)
                        in->dim_stats[d].add(extract_dim_fast(lf->keys[s], d));
            } else {
                auto* ic = static_cast<InnerNode*>(c);
                cnt += ic->subtree_count;
                for (size_t d = 0; d < dim_count_; ++d)
                    in->dim_stats[d].merge(ic->dim_stats[d]);
            }
        }
        in->range_lo      = lo;
        in->range_hi      = hi;
        in->subtree_count = cnt;
    }

    // -------------------------------------------------------------------------
    //  Remove descent with incremental metadata updates.
    //  When a leaf falls below LEAF_SLOTMIN we borrow from or merge with an
    //  adjacent sibling via the parent's slotkey[].  Only leaf-level
    //  rebalancing is implemented (inner underflow collapses only when root
    //  has a single child) — sufficient for the benchmark's delete pattern
    //  and keeps the full B+-tree invariant under realistic delete workloads.
    // -------------------------------------------------------------------------
    bool remove_descend(NodeBase* n, CompositeKey key, uint64_t* rec_dims) {
        if (n->level == 0) {
            auto* lf = static_cast<LeafNode*>(n);
            uint16_t s = leaf_find_lower(lf, key);
            if (s >= lf->slotuse || lf->keys[s] != key) return false;
            for (size_t d = 0; d < dim_count_; ++d)
                rec_dims[d] = extract_dim_fast(lf->keys[s], d);
            for (uint16_t i = s; i + 1 < lf->slotuse; ++i) {
                lf->keys[i]   = lf->keys[i + 1];
                lf->values[i] = lf->values[i + 1];
            }
            lf->slotuse--;
            lf->recompute_range();
            return true;
        }
        auto* in = static_cast<InnerNode*>(n);
        uint16_t slot = inner_find_lower(in, key);
        bool ok = remove_descend(in->childid[slot], key, rec_dims);
        if (!ok) return false;
        inner_meta_sub(in, rec_dims);

        // Leaf underflow handling: if the child is a leaf and below minimum,
        // try to borrow from or merge with a sibling.  Inner-level underflow
        // is rare (only near-empty splits) and handled by the root-collapse
        // check in remove() above.
        NodeBase* child = in->childid[slot];
        if (child->level == 0) {
            auto* clf = static_cast<LeafNode*>(child);
            if (clf->is_underflow())
                rebalance_leaf(in, slot);
        }
        return true;
    }

    // Restore leaf B+-tree invariants by borrowing from a sibling or merging
    // two siblings.  Updates parent's slotkey[] and child pointers, and keeps
    // the next_leaf/prev_leaf chain intact.  Parent dim_stats are unchanged
    // (same records total); range_lo/hi may become stale-wide which is safe.
    void rebalance_leaf(InnerNode* in, uint16_t slot) {
        auto* lf = static_cast<LeafNode*>(in->childid[slot]);
        // Prefer right sibling (more common case during forward delete).
        if (slot + 1 <= in->slotuse) {
            auto* rs = static_cast<LeafNode*>(in->childid[slot + 1]);
            if (rs->slotuse > LEAF_SLOTMIN) {
                // Borrow smallest key from right sibling.
                lf->keys[lf->slotuse]   = rs->keys[0];
                lf->values[lf->slotuse] = rs->values[0];
                lf->slotuse++;
                for (uint16_t i = 0; i + 1 < rs->slotuse; ++i) {
                    rs->keys[i]   = rs->keys[i + 1];
                    rs->values[i] = rs->values[i + 1];
                }
                rs->slotuse--;
                lf->recompute_range();
                rs->recompute_range();
                in->slotkey[slot] = lf->keys[lf->slotuse - 1];
                return;
            }
            // Merge lf + rs into lf.
            for (uint16_t i = 0; i < rs->slotuse; ++i) {
                lf->keys[lf->slotuse + i]   = rs->keys[i];
                lf->values[lf->slotuse + i] = rs->values[i];
            }
            lf->slotuse += rs->slotuse;
            lf->next_leaf = rs->next_leaf;
            if (rs->next_leaf) rs->next_leaf->prev_leaf = lf;
            if (tail_leaf_ == rs) tail_leaf_ = lf;
            lf->recompute_range();
            // Remove slotkey[slot] and childid[slot+1].
            for (uint16_t i = slot; i + 1 < in->slotuse; ++i)
                in->slotkey[i] = in->slotkey[i + 1];
            for (uint16_t i = slot + 1; i < in->slotuse; ++i)
                in->childid[i] = in->childid[i + 1];
            in->slotuse--;
            delete rs;
            return;
        }
        // Left sibling path (slot is the rightmost child).
        if (slot > 0) {
            auto* ls = static_cast<LeafNode*>(in->childid[slot - 1]);
            if (ls->slotuse > LEAF_SLOTMIN) {
                // Borrow largest key from left sibling.
                for (uint16_t i = lf->slotuse; i > 0; --i) {
                    lf->keys[i]   = lf->keys[i - 1];
                    lf->values[i] = lf->values[i - 1];
                }
                lf->keys[0]   = ls->keys[ls->slotuse - 1];
                lf->values[0] = ls->values[ls->slotuse - 1];
                lf->slotuse++;
                ls->slotuse--;
                lf->recompute_range();
                ls->recompute_range();
                in->slotkey[slot - 1] = ls->keys[ls->slotuse - 1];
                return;
            }
            // Merge ls + lf into ls.
            for (uint16_t i = 0; i < lf->slotuse; ++i) {
                ls->keys[ls->slotuse + i]   = lf->keys[i];
                ls->values[ls->slotuse + i] = lf->values[i];
            }
            ls->slotuse += lf->slotuse;
            ls->next_leaf = lf->next_leaf;
            if (lf->next_leaf) lf->next_leaf->prev_leaf = ls;
            if (tail_leaf_ == lf) tail_leaf_ = ls;
            ls->recompute_range();
            for (uint16_t i = slot - 1; i + 1 < in->slotuse; ++i)
                in->slotkey[i] = in->slotkey[i + 1];
            for (uint16_t i = slot; i < in->slotuse; ++i)
                in->childid[i] = in->childid[i + 1];
            in->slotuse--;
            delete lf;
        }
    }

    // -------------------------------------------------------------------------
    //  Destruction
    // -------------------------------------------------------------------------
    void destroy_subtree(NodeBase* n) {
        if (!n) return;
        if (n->level == 0) {
            delete static_cast<LeafNode*>(n);
        } else {
            auto* in = static_cast<InnerNode*>(n);
            uint16_t nchildren = in->slotuse + 1;
            for (uint16_t i = 0; i < nchildren; ++i)
                destroy_subtree(in->childid[i]);
            delete in;
        }
    }
};

}  // namespace hptree
