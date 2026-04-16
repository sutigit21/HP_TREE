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
        // Pack leaves at 75% of LEAF_SLOTMAX to leave slack for insert Q9 —
        // otherwise every insert lands in a full leaf and causes a split.
        static constexpr uint16_t LEAF_PACK = (LEAF_SLOTMAX * 3) / 4;
        std::vector<NodeBase*> level_nodes;
        std::vector<std::array<DimStats, MAX_DIMS>> leaf_stats;
        LeafNode* prev = nullptr;
        size_t i = 0;
        while (i < recs.size()) {
            size_t chunk = std::min<size_t>(LEAF_PACK, recs.size() - i);
            LeafNode* leaf = new LeafNode();
            leaf->init();
            leaf->slotuse = static_cast<uint16_t>(chunk);
            std::array<DimStats, MAX_DIMS> ls{};
            for (size_t j = 0; j < chunk; ++j) {
                leaf->keys[j]   = recs[i + j].key;
                leaf->values[j] = recs[i + j].value;
                for (size_t d = 0; d < dim_count_; ++d)
                    ls[d].add(extract_dim_fast(recs[i + j].key, d));
            }
            leaf->recompute_range_beta();
            leaf->prev_leaf = prev;
            if (prev) prev->next_leaf = leaf;
            else      head_leaf_ = leaf;
            prev = leaf;
            level_nodes.push_back(leaf);
            leaf_stats.push_back(ls);
            i += chunk;
        }
        tail_leaf_ = prev;
        total_records_ = recs.size();
        levels_ = 1;

        // Build inner levels bottom-up, carrying DimStats via `level_stats`.
        std::vector<std::array<DimStats, MAX_DIMS>> level_stats = std::move(leaf_stats);
        std::vector<uint64_t>                      level_counts(level_nodes.size());
        for (size_t k = 0; k < level_nodes.size(); ++k) {
            auto* lf = static_cast<LeafNode*>(level_nodes[k]);
            level_counts[k] = lf->slotuse;
        }

        while (level_nodes.size() > 1) {
            std::vector<NodeBase*> parents;
            std::vector<std::array<DimStats, MAX_DIMS>> parent_stats;
            std::vector<uint64_t> parent_counts;
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
                bool all_hom = true;
                std::array<DimStats, MAX_DIMS> agg{};

                for (size_t j = 0; j < chunk; ++j) {
                    NodeBase* c = level_nodes[k + j];
                    inner->childid[j] = c;
                    if (c->range_lo < lo) lo = c->range_lo;
                    if (c->range_hi > hi) hi = c->range_hi;
                    if (!c->is_homogeneous) all_hom = false;
                    count += level_counts[k + j];
                    for (size_t d = 0; d < dim_count_; ++d)
                        agg[d].merge(level_stats[k + j][d]);
                }
                for (size_t j = 0; j + 1 < chunk; ++j)
                    inner->slotkey[j] = level_nodes[k + j]->range_hi;

                inner->range_lo       = lo;
                inner->range_hi       = hi;
                inner->beta_value     = BetaComputer::compute_beta(lo, hi);
                inner->is_homogeneous = all_hom && (lo == hi);
                inner->subtree_count  = count;
                for (size_t d = 0; d < dim_count_; ++d) inner->dim_stats[d] = agg[d];

                parents.push_back(inner);
                parent_stats.push_back(agg);
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
    //  Predicate search with beta-skip pruning
    // =========================================================================
    std::vector<Record> predicate_search(const PredicateSet& ps) const {
        std::vector<Record> out;
        if (root_ == nullptr) return out;
        KeyRange kr = ps.to_key_range(schema_);
        predicate_search_node(root_, ps, kr, out);
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
            leaf->recompute_range_beta();
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
        bool all_hom = true;
        for (size_t d = 0; d < dim_count_; ++d) in->dim_stats[d] = DimStats{};

        uint16_t nchildren = in->slotuse + 1;
        for (uint16_t i = 0; i < nchildren; ++i) {
            NodeBase* c = in->childid[i];
            if (c->range_lo < lo) lo = c->range_lo;
            if (c->range_hi > hi) hi = c->range_hi;
            if (!c->is_homogeneous) all_hom = false;
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
        in->beta_value     = BetaComputer::compute_beta(lo, hi);
        in->is_homogeneous = all_hom && (lo == hi);
        in->subtree_count  = cnt;
    }

    // Incremental: O(dim_count) on insert of a single key.
    // beta_value / is_homogeneous are NOT updated in the hot path — they are
    // derived fields not read by any query (range_lo/range_hi drive pruning).
    // They are refreshed only by rebuild_inner_meta_from_children on splits.
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
        in->subtree_count--;
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

    void predicate_search_node(NodeBase* n, const PredicateSet& ps,
                               const KeyRange& kr,
                               std::vector<Record>& out) const {
        if (n->range_hi < kr.low || n->range_lo > kr.high) return;
        if (n->level == 0) {
            auto* lf = static_cast<LeafNode*>(n);
            for (uint16_t s = 0; s < lf->slotuse; ++s) {
                CompositeKey k = lf->keys[s];
                if (k < kr.low || k > kr.high) continue;
                if (ps.evaluate(k, schema_))
                    out.push_back({ k, lf->values[s] });
            }
            return;
        }
        auto* in = static_cast<InnerNode*>(n);
        if (!subtree_may_contain(in, ps)) return;
        uint16_t nchildren = in->slotuse + 1;
        for (uint16_t i = 0; i < nchildren; ++i)
            predicate_search_node(in->childid[i], ps, kr, out);
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
            uint16_t nchildren = in->slotuse + 1;
            for (uint16_t i = 0; i < nchildren; ++i)
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
            // Fast range update — skip beta_value recompute (not read by queries).
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

        // Incremental stats update first.
        inner_meta_add(in, rec.key, key_dims);

        if (child_split_right != nullptr) {
            for (uint16_t i = in->slotuse; i > slot; --i) {
                in->slotkey[i]     = in->slotkey[i - 1];
                in->childid[i + 1] = in->childid[i];
            }
            in->slotkey[slot]     = child_split_key;
            in->childid[slot + 1] = child_split_right;
            in->slotuse++;
            if (in->is_full()) split_inner(in, split_key, split_right);
        }
        return true;
    }

    void split_leaf(LeafNode* lf, CompositeKey& split_key, NodeBase*& right_out) {
        uint16_t mid = lf->slotuse / 2;
        LeafNode* right = new LeafNode();
        right->init();
        right->slotuse = lf->slotuse - mid;
        for (uint16_t i = 0; i < right->slotuse; ++i) {
            right->keys[i]   = lf->keys[mid + i];
            right->values[i] = lf->values[mid + i];
        }
        lf->slotuse = mid;
        lf->recompute_range_beta();
        right->recompute_range_beta();

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

        // Both halves must recompute their stats from remaining children.
        rebuild_inner_meta_from_children(in);
        rebuild_inner_meta_from_children(right);
        right_out = right;
    }

    // -------------------------------------------------------------------------
    //  Remove descent with incremental metadata updates
    // -------------------------------------------------------------------------
    bool remove_descend(NodeBase* n, CompositeKey key, uint64_t* rec_dims) {
        if (n->level == 0) {
            auto* lf = static_cast<LeafNode*>(n);
            uint16_t s = leaf_find_lower(lf, key);
            if (s >= lf->slotuse || lf->keys[s] != key) return false;
            // Capture record dim values before removal for parent stat updates.
            for (size_t d = 0; d < dim_count_; ++d)
                rec_dims[d] = extract_dim_fast(lf->keys[s], d);
            for (uint16_t i = s; i + 1 < lf->slotuse; ++i) {
                lf->keys[i]   = lf->keys[i + 1];
                lf->values[i] = lf->values[i + 1];
            }
            lf->slotuse--;
            lf->recompute_range_beta();
            return true;
        }
        auto* in = static_cast<InnerNode*>(n);
        uint16_t slot = inner_find_lower(in, key);
        bool ok = remove_descend(in->childid[slot], key, rec_dims);
        if (ok) inner_meta_sub(in, rec_dims);
        return ok;
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
