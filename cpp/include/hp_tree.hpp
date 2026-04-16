#pragma once

#include "hp_tree_common.hpp"
#include "hp_tree_node.hpp"
#include "hp_tree_iterator.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace hptree {

// =============================================================================
//  HPTree: tlx-style B+-tree with per-node beta metadata and per-subtree
//  DimStats aggregates for predicate-pruning & O(1) range aggregates.
//
//  Single-threaded, no WAL/MVCC/delta-buffer by default.  Identical structural
//  cost to tlx::btree_multimap.
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
    }

    ~HPTree() { destroy_subtree(root_); }

    HPTree(const HPTree&)            = delete;
    HPTree& operator=(const HPTree&) = delete;

    // =========================================================================
    //  Bulk load: sort records then build leaves + inner levels bottom-up.
    //  Populates beta metadata and subtree DimStats along the way.
    // =========================================================================
    void bulk_load(std::vector<Record>&& recs) {
        destroy_subtree(root_);
        root_ = head_leaf_ = tail_leaf_ = nullptr;
        total_records_ = 0;
        levels_ = 0;
        if (recs.empty()) return;

        std::sort(recs.begin(), recs.end(),
                  [](const Record& a, const Record& b){ return a.key < b.key; });

        // Build leaf level
        std::vector<NodeBase*> level_nodes;
        LeafNode* prev = nullptr;
        size_t i = 0;
        while (i < recs.size()) {
            size_t chunk = std::min<size_t>(LEAF_SLOTMAX, recs.size() - i);
            LeafNode* leaf = new LeafNode();
            leaf->init();
            leaf->slotuse = static_cast<uint16_t>(chunk);
            for (size_t j = 0; j < chunk; ++j) {
                leaf->keys[j]   = recs[i + j].key;
                leaf->values[j] = recs[i + j].value;
            }
            leaf->recompute_range_beta();
            leaf->prev_leaf = prev;
            if (prev) prev->next_leaf = leaf;
            else      head_leaf_ = leaf;
            prev = leaf;
            level_nodes.push_back(leaf);
            i += chunk;
        }
        tail_leaf_ = prev;
        total_records_ = recs.size();
        levels_ = 1;

        // Build inner levels
        while (level_nodes.size() > 1) {
            std::vector<NodeBase*> parents;
            size_t k = 0;
            while (k < level_nodes.size()) {
                size_t chunk = std::min<size_t>(INNER_SLOTMAX + 1,
                                                level_nodes.size() - k);
                InnerNode* inner = new InnerNode();
                inner->init(static_cast<uint16_t>(levels_));
                inner->slotuse = static_cast<uint16_t>(chunk - 1);
                for (size_t j = 0; j < chunk; ++j)
                    inner->childid[j] = level_nodes[k + j];
                for (size_t j = 0; j + 1 < chunk; ++j) {
                    // separator = max key of child j (for multimap semantics
                    // use the first key of child j+1 minus one... use hi of j)
                    inner->slotkey[j] = level_nodes[k + j]->range_hi;
                }
                populate_inner_meta(inner);
                parents.push_back(inner);
                k += chunk;
            }
            level_nodes = std::move(parents);
            levels_++;
        }
        root_ = level_nodes[0];
    }

    size_t size() const { return total_records_; }
    size_t levels() const { return levels_; }

    // =========================================================================
    //  Point lookup (multimap): returns all matching records.
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
    //  Range scan.  Uses lower_bound + leaf chain walk (same as tlx).
    // =========================================================================
    std::vector<Record> range_search(CompositeKey lo, CompositeKey hi) const {
        std::vector<Record> out;
        if (root_ == nullptr || lo > hi) return out;
        LeafNode* leaf = find_leaf_for(lo);
        if (!leaf) return out;
        uint16_t s = leaf_find_lower(leaf, lo);
        while (leaf) {
            while (s < leaf->slotuse && leaf->keys[s] <= hi) {
                out.push_back({ leaf->keys[s], leaf->values[s] });
                ++s;
            }
            if (s < leaf->slotuse) break;        // hit a key > hi
            leaf = leaf->next_leaf;
            s = 0;
            if (leaf && leaf->keys[0] > hi) break;
        }
        return out;
    }

    // =========================================================================
    //  Predicate search with beta-skip pruning.
    //
    //  The KeyRange derived from the PredicateSet bounds the scan; inner nodes
    //  whose [range_lo, range_hi] does not overlap are skipped; additionally,
    //  for each EQ/BETWEEN predicate we can skip a subtree if the corresponding
    //  dim_stats[dim] min/max is outside the filter range.
    // =========================================================================
    std::vector<Record> predicate_search(const PredicateSet& ps) const {
        std::vector<Record> out;
        if (root_ == nullptr) return out;
        KeyRange kr = ps.to_key_range(schema_);
        predicate_search_node(root_, ps, kr, out);
        return out;
    }

    // =========================================================================
    //  Aggregation: sum + count of a given dimension over [lo, hi] key range.
    //  Uses per-InnerNode subtree DimStats for O(1) contribution of fully-
    //  contained children.
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

    // Runner-compat iterator surface
    class RunnerIterator {
    public:
        explicit RunnerIterator(HPTreeIterator it) : it_(it) {}
        bool valid() const { return !it_.is_end(); }
        void next() { ++it_; }
        CompositeKey key() const { return it_.key(); }
        uint64_t     value() const { return it_.value(); }
    private:
        HPTreeIterator it_;
    };

    RunnerIterator runner_begin() const { return RunnerIterator(begin()); }

    // =========================================================================
    //  Insert / remove (single-record, in-place).  tlx-style recursive split.
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

        CompositeKey split_key = 0;
        NodeBase*    split_right = nullptr;
        bool ok = insert_descend(root_, rec, split_key, split_right);
        if (!ok) return false;

        if (split_right != nullptr) {
            // Grow tree by one level
            InnerNode* newroot = new InnerNode();
            newroot->init(static_cast<uint16_t>(levels_));
            newroot->slotuse = 1;
            newroot->slotkey[0] = split_key;
            newroot->childid[0] = root_;
            newroot->childid[1] = split_right;
            populate_inner_meta(newroot);
            root_ = newroot;
            levels_++;
        }
        total_records_++;
        return true;
    }

    bool remove(CompositeKey key) {
        if (root_ == nullptr) return false;
        bool ok = remove_descend(root_, key);
        if (!ok) return false;

        // If root is an inner with 0 separators and 1 child, collapse level.
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
    NodeBase*          root_;
    LeafNode*          head_leaf_;
    LeafNode*          tail_leaf_;
    size_t             total_records_;
    size_t             levels_;

    // -------------------------------------------------------------------------
    //  Traversal helpers
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
    //  Inner metadata: range, beta, subtree_count, per-dim DimStats
    // -------------------------------------------------------------------------
    void populate_inner_meta(InnerNode* in) {
        uint16_t nchildren = in->slotuse + 1;
        CompositeKey lo = COMPOSITE_KEY_MAX;
        CompositeKey hi = COMPOSITE_KEY_MIN;
        uint64_t     cnt = 0;
        std::vector<DimStats> stats(dim_count_);
        bool all_homogeneous = true;

        for (uint16_t i = 0; i < nchildren; ++i) {
            NodeBase* c = in->childid[i];
            if (c->range_lo < lo) lo = c->range_lo;
            if (c->range_hi > hi) hi = c->range_hi;
            if (!c->is_homogeneous) all_homogeneous = false;
            if (c->level == 0) {
                auto* lf = static_cast<LeafNode*>(c);
                cnt += lf->slotuse;
                for (uint16_t s = 0; s < lf->slotuse; ++s) {
                    for (size_t d = 0; d < dim_count_; ++d) {
                        uint64_t v = extract_dim(lf->keys[s], d);
                        stats[d].add(v);
                    }
                }
            } else {
                auto* ic = static_cast<InnerNode*>(c);
                cnt += ic->subtree_count;
                for (size_t d = 0; d < dim_count_; ++d)
                    stats[d].merge(ic->dim_stats[d]);
            }
        }
        in->range_lo       = lo;
        in->range_hi       = hi;
        in->beta_value     = BetaComputer::compute_beta(lo, hi);
        in->is_homogeneous = all_homogeneous && (lo == hi);
        in->subtree_count  = cnt;
        in->dim_stats      = std::move(stats);
    }

    uint64_t extract_dim(CompositeKey key, size_t dim) const {
        uint8_t  off  = schema_.offset_of(dim);
        uint64_t mask = schema_.mask_of(dim);
        return static_cast<uint64_t>((key >> off) & mask);
    }

    // -------------------------------------------------------------------------
    //  Predicate dispatch helpers
    // -------------------------------------------------------------------------
    // Return false iff any predicate can be proven false for the whole subtree
    // by dim_stats.  On leaves this is skipped (we evaluate each key).
    bool subtree_may_contain(const InnerNode* in, const PredicateSet& ps) const {
        for (auto& p : ps.predicates) {
            if (p.dim_idx >= dim_count_) continue;
            const DimStats& ds = in->dim_stats[p.dim_idx];
            if (ds.count == 0) return false;
            uint64_t v  = static_cast<uint64_t>(p.value);
            uint64_t vh = static_cast<uint64_t>(p.value_high);
            switch (p.op) {
            case PredicateOp::EQ:
                if (v < ds.min_val || v > ds.max_val) return false;
                break;
            case PredicateOp::BETWEEN:
                if (vh < ds.min_val || v > ds.max_val) return false;
                break;
            case PredicateOp::LT:
                if (ds.min_val >= v) return false; break;
            case PredicateOp::LTE:
                if (ds.min_val > v)  return false; break;
            case PredicateOp::GT:
                if (ds.max_val <= v) return false; break;
            case PredicateOp::GTE:
                if (ds.max_val < v)  return false; break;
            default: break;
            }
        }
        return true;
    }

    void predicate_search_node(NodeBase* n, const PredicateSet& ps,
                               const KeyRange& kr,
                               std::vector<Record>& out) const {
        // Key-range pruning via beta metadata
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

    // -------------------------------------------------------------------------
    //  Range aggregate with O(1) shortcut on fully-contained subtrees.
    // -------------------------------------------------------------------------
    void aggregate_dim_node(NodeBase* n, size_t dim,
                            CompositeKey lo, CompositeKey hi,
                            AggregateResult& r) const {
        if (n->range_hi < lo || n->range_lo > hi) return;

        if (n->level > 0) {
            auto* in = static_cast<InnerNode*>(n);
            // Full containment — use precomputed stats
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
            r.sum += static_cast<double>(extract_dim(k, dim));
        }
    }

    // -------------------------------------------------------------------------
    //  Insert descent (recursive, split-bubbling; no MVCC/delta-buffer).
    // -------------------------------------------------------------------------
    bool insert_descend(NodeBase* n, const Record& rec,
                        CompositeKey& split_key, NodeBase*& split_right) {
        if (n->level == 0) {
            auto* lf = static_cast<LeafNode*>(n);
            uint16_t s = leaf_find_lower(lf, rec.key);
            // shift right
            for (uint16_t i = lf->slotuse; i > s; --i) {
                lf->keys[i]   = lf->keys[i - 1];
                lf->values[i] = lf->values[i - 1];
            }
            lf->keys[s]   = rec.key;
            lf->values[s] = rec.value;
            lf->slotuse++;
            lf->recompute_range_beta();
            if (lf->is_full()) split_leaf(lf, split_key, split_right);
            return true;
        }

        auto* in = static_cast<InnerNode*>(n);
        uint16_t slot = inner_find_lower(in, rec.key);
        CompositeKey child_split_key = 0;
        NodeBase* child_split_right = nullptr;
        bool ok = insert_descend(in->childid[slot], rec,
                                 child_split_key, child_split_right);
        if (!ok) return false;

        // Update metadata (cheap: derive from child update — but we recompute
        // incrementally for correctness).
        if (child_split_right != nullptr) {
            // Insert separator + child pointer
            for (uint16_t i = in->slotuse; i > slot; --i) {
                in->slotkey[i]     = in->slotkey[i - 1];
                in->childid[i + 1] = in->childid[i];
            }
            in->slotkey[slot]     = child_split_key;
            in->childid[slot + 1] = child_split_right;
            in->slotuse++;
            if (in->is_full()) split_inner(in, split_key, split_right);
        }

        // Refresh metadata for this inner node (range + dim_stats).
        populate_inner_meta(in);
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
        right->init(in->level);
        right->slotuse = in->slotuse - mid - 1;
        for (uint16_t i = 0; i < right->slotuse; ++i)
            right->slotkey[i] = in->slotkey[mid + 1 + i];
        for (uint16_t i = 0; i <= right->slotuse; ++i)
            right->childid[i] = in->childid[mid + 1 + i];
        split_key = in->slotkey[mid];
        in->slotuse = mid;
        populate_inner_meta(in);
        populate_inner_meta(right);
        right_out = right;
    }

    // -------------------------------------------------------------------------
    //  Remove descent (tlx semantics: remove first occurrence of key, no
    //  rebalance; stores underfull nodes which are recomputed).
    // -------------------------------------------------------------------------
    bool remove_descend(NodeBase* n, CompositeKey key) {
        if (n->level == 0) {
            auto* lf = static_cast<LeafNode*>(n);
            uint16_t s = leaf_find_lower(lf, key);
            if (s >= lf->slotuse || lf->keys[s] != key) return false;
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
        bool ok = remove_descend(in->childid[slot], key);
        if (ok) populate_inner_meta(in);
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
