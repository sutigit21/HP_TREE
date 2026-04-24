#pragma once

#include "hp_tree_common.hpp"

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <array>
#include <atomic>

namespace hptree {

// =============================================================================
//  tlx-style node layout with inline fixed-size arrays.
//  Beta metadata + per-subtree DimStats enable subtree skipping and O(1)
//  aggregate shortcuts without changing the core B+-tree structure.
// =============================================================================

static constexpr uint16_t LEAF_SLOTMAX  = 32;   // matches DEFAULT_MAX_LEAF_SIZE
static constexpr uint16_t LEAF_SLOTMIN  = 16;
static constexpr uint16_t INNER_SLOTMAX = 32;   // matches DEFAULT_BRANCHING_FACTOR
static constexpr uint16_t INNER_SLOTMIN = 16;

struct NodeBase {
    uint16_t level;       // 0 => leaf, >0 => inner
    uint16_t slotuse;

    bool is_leaf() const { return level == 0; }
};

struct LeafNode : public NodeBase {
    LeafNode*    prev_leaf;
    LeafNode*    next_leaf;

    // +1 slack slot for temporary overflow during insert-before-split.
    CompositeKey keys  [LEAF_SLOTMAX + 1];
    uint64_t     values[LEAF_SLOTMAX + 1];

    void init() {
        level = 0;
        slotuse = 0;
        prev_leaf = nullptr;
        next_leaf = nullptr;
    }

    bool is_full()      const { return slotuse >= LEAF_SLOTMAX; }
    bool is_few()       const { return slotuse <= LEAF_SLOTMIN; }
    bool is_underflow() const { return slotuse < LEAF_SLOTMIN; }

    // Leaf keys are kept sorted, so min/max are the first/last slots.
    // Callers must only invoke these when slotuse > 0.
    CompositeKey min_key() const { return keys[0]; }
    CompositeKey max_key() const { return keys[slotuse - 1]; }
};

struct InnerNode : public NodeBase {
    // range_lo / range_hi drive inner-node pruning.  They are maintained
    // explicitly only on inner nodes; for leaves min_key()/max_key() derive
    // them from the sorted key array without any extra storage.
    CompositeKey range_lo;
    CompositeKey range_hi;

    // +1 slack for temporary overflow during insert-before-split.
    CompositeKey slotkey[INNER_SLOTMAX + 1];
    NodeBase*    childid[INNER_SLOTMAX + 2];

    // Aggregated subtree stats (per-dimension) — enables beta-skip predicate
    // pruning and O(1) range aggregates.  Fixed-size to avoid per-node heap
    // allocation.  Only the first dim_count entries are meaningful.
    uint64_t                           subtree_count;
    std::array<DimStats, MAX_DIMS>     dim_stats;

    std::array<CommittedAgg, MAX_DIMS> committed_agg;
    Epoch                              last_commit_epoch = 0;
    SeqLock                            agg_seqlock;

    void init(uint16_t lvl, size_t dim_count) {
        level = lvl;
        slotuse = 0;
        range_lo = COMPOSITE_KEY_MAX;
        range_hi = COMPOSITE_KEY_MIN;
        subtree_count = 0;
        for (size_t d = 0; d < dim_count; ++d) {
            dim_stats[d] = DimStats{};
            committed_agg[d] = CommittedAgg{};
        }
        last_commit_epoch = 0;
        agg_seqlock.seq.store(0, std::memory_order_relaxed);
    }

    bool is_full()      const { return slotuse >= INNER_SLOTMAX; }
    bool is_few()       const { return slotuse <= INNER_SLOTMIN; }
    bool is_underflow() const { return slotuse < INNER_SLOTMIN; }
};

// Uniform accessors for a child node's key range — hides the
// LeafNode-derived-from-keys vs InnerNode-stored distinction so callers can
// work with NodeBase* directly.  Preconditions: leaves must be non-empty; the
// caller is responsible for checking slotuse > 0 (true for every node that
// lives in a valid B+-tree after bulk_load / split / rebalance).
inline CompositeKey node_range_lo(const NodeBase* n) {
    if (n->level == 0)
        return static_cast<const LeafNode*>(n)->min_key();
    return static_cast<const InnerNode*>(n)->range_lo;
}
inline CompositeKey node_range_hi(const NodeBase* n) {
    if (n->level == 0)
        return static_cast<const LeafNode*>(n)->max_key();
    return static_cast<const InnerNode*>(n)->range_hi;
}

// Linear scan beats std::lower_bound on nodes <= 32 slots due to branch
// prediction (this is exactly what tlx does).
inline uint16_t leaf_find_lower(const LeafNode* n, CompositeKey key) {
    uint16_t lo = 0;
    while (lo < n->slotuse && n->keys[lo] < key) ++lo;
    return lo;
}
inline uint16_t leaf_find_upper(const LeafNode* n, CompositeKey key) {
    uint16_t lo = 0;
    while (lo < n->slotuse && n->keys[lo] <= key) ++lo;
    return lo;
}
inline uint16_t inner_find_lower(const InnerNode* n, CompositeKey key) {
    uint16_t lo = 0;
    while (lo < n->slotuse && n->slotkey[lo] < key) ++lo;
    return lo;
}

}  // namespace hptree
