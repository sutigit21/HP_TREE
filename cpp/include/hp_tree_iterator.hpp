#pragma once

#include "hp_tree_common.hpp"
#include "hp_tree_node.hpp"

#include <cstdint>
#include <iterator>

namespace hptree {

// =============================================================================
//  Simple forward iterator: {LeafNode*, slot} pair, following next_leaf chain.
//  No MVCC, no predicate, no range check — identical semantics to tlx.
// =============================================================================

class HPTreeIterator {
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type        = std::pair<CompositeKey, uint64_t>;
    using difference_type   = std::ptrdiff_t;
    using pointer           = const value_type*;
    using reference         = const value_type&;

    HPTreeIterator() : leaf_(nullptr), slot_(0) {}
    HPTreeIterator(LeafNode* leaf, uint16_t slot) : leaf_(leaf), slot_(slot) {}

    bool operator==(const HPTreeIterator& o) const {
        return leaf_ == o.leaf_ && slot_ == o.slot_;
    }
    bool operator!=(const HPTreeIterator& o) const { return !(*this == o); }

    CompositeKey key()   const { return leaf_->keys[slot_]; }
    uint64_t     value() const { return leaf_->values[slot_]; }

    HPTreeIterator& operator++() {
        ++slot_;
        if (leaf_ != nullptr && slot_ >= leaf_->slotuse) {
            leaf_ = leaf_->next_leaf;
            slot_ = 0;
        }
        return *this;
    }
    HPTreeIterator operator++(int) {
        HPTreeIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    LeafNode* leaf() const { return leaf_; }
    uint16_t  slot() const { return slot_; }
    bool      is_end() const { return leaf_ == nullptr; }

    // Runner-style alias surface (no wrapper class needed).
    bool valid() const { return leaf_ != nullptr; }
    void next()        { ++(*this); }

private:
    LeafNode* leaf_;
    uint16_t  slot_;
};

}  // namespace hptree
