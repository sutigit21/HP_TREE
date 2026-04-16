#pragma once

#include "hp_tree_common.hpp"

namespace hptree {

struct NodeMeta {
    NodeId   id            = INVALID_NODE;
    NodeType type          = NodeType::LEAF;
    PageId   page_id       = INVALID_PAGE;
    uint32_t depth         = 0;
    double   beta_value    = std::numeric_limits<double>::infinity();
    double   beta_threshold= DEFAULT_BETA_STRICT;
    KeyRange range;
    LSN      page_lsn      = INVALID_LSN;
    uint64_t record_count  = 0;
    uint64_t tombstone_count = 0;
    bool     is_homogeneous= false;
};

class HPNode {
public:
    NodeMeta meta;
    NodeLatch latch;
    PerDimAggregates aggregates;

    virtual ~HPNode() = default;

    bool is_leaf() const {
        return meta.type == NodeType::LEAF || meta.type == NodeType::HOMOGENEOUS_LEAF;
    }

    bool is_internal() const {
        return meta.type == NodeType::INTERNAL;
    }

    bool is_homogeneous() const {
        return meta.is_homogeneous;
    }

    bool is_underfull(size_t min_size) const {
        return meta.record_count < min_size;
    }

    bool is_overfull(size_t max_size) const {
        return meta.record_count > max_size;
    }

    double tombstone_ratio() const {
        if (meta.record_count == 0) return 0.0;
        return static_cast<double>(meta.tombstone_count) / meta.record_count;
    }

    bool needs_compaction() const {
        return tombstone_ratio() > TOMBSTONE_COMPACT_RATIO;
    }

    virtual size_t memory_usage() const = 0;
};

class LeafNode : public HPNode {
public:
    std::vector<Record> records;
    NullBitmap          null_bitmap;
    NodeId              next_leaf = INVALID_NODE;
    NodeId              prev_leaf = INVALID_NODE;
    LeafNode*           next_leaf_ptr = nullptr;
    LeafNode*           prev_leaf_ptr = nullptr;

    LeafNode() { meta.type = NodeType::LEAF; }

    void insert_record(const Record& rec) {
        auto it = std::lower_bound(records.begin(), records.end(), rec,
            [](const Record& a, const Record& b) { return a.key < b.key; });
        size_t idx = it - records.begin();
        records.insert(it, rec);
        meta.record_count = records.size();
        update_range();
    }

    bool remove_record(CompositeKey key, TxnId txn) {
        for (auto& r : records) {
            if (r.key == key && !r.tombstone) {
                r.tombstone = true;
                r.version.xmax = txn;
                meta.tombstone_count++;
                return true;
            }
        }
        return false;
    }

    void hard_delete(CompositeKey key) {
        auto it = std::remove_if(records.begin(), records.end(),
            [key](const Record& r) { return r.key == key; });
        if (it != records.end()) {
            records.erase(it, records.end());
            meta.record_count = records.size();
            update_range();
        }
    }

    void compact() {
        records.erase(
            std::remove_if(records.begin(), records.end(),
                [](const Record& r) { return r.tombstone; }),
            records.end());
        meta.record_count = records.size();
        meta.tombstone_count = 0;
        update_range();
    }

    std::vector<Record*> search(CompositeKey key, TxnId reader_txn) {
        std::vector<Record*> result;
        auto range = std::equal_range(records.begin(), records.end(), Record{key},
            [](const Record& a, const Record& b) { return a.key < b.key; });
        for (auto it = range.first; it != range.second; ++it) {
            if (!it->tombstone && it->version.is_visible(reader_txn))
                result.push_back(&(*it));
        }
        return result;
    }

    // Fast path: assumes no tombstones in this leaf and reader_txn==TXN_COMMITTED.
    // Caller must verify those preconditions.
    std::vector<Record*> search_fast(CompositeKey key) {
        std::vector<Record*> result;
        auto range = std::equal_range(records.begin(), records.end(), Record{key},
            [](const Record& a, const Record& b) { return a.key < b.key; });
        for (auto it = range.first; it != range.second; ++it) {
            result.push_back(&(*it));
        }
        return result;
    }

    std::vector<Record*> range_search(const KeyRange& kr, TxnId reader_txn) {
        std::vector<Record*> result;
        Record lo{kr.low}; Record hi{kr.high};
        auto start = std::lower_bound(records.begin(), records.end(), lo,
            [](const Record& a, const Record& b) { return a.key < b.key; });
        auto end = std::upper_bound(records.begin(), records.end(), hi,
            [](const Record& a, const Record& b) { return a.key < b.key; });
        for (auto it = start; it != end; ++it) {
            if (!it->tombstone && it->version.is_visible(reader_txn))
                result.push_back(&(*it));
        }
        return result;
    }

    // Fast path: skip MVCC/tombstone checks. Caller guarantees clean tree.
    void range_search_fast(const KeyRange& kr, std::vector<Record*>& out) {
        Record lo{kr.low}; Record hi{kr.high};
        auto start = std::lower_bound(records.begin(), records.end(), lo,
            [](const Record& a, const Record& b) { return a.key < b.key; });
        auto end = std::upper_bound(records.begin(), records.end(), hi,
            [](const Record& a, const Record& b) { return a.key < b.key; });
        out.reserve(out.size() + static_cast<size_t>(end - start));
        for (auto it = start; it != end; ++it) {
            out.push_back(&(*it));
        }
    }

    std::vector<Record*> predicate_search(const PredicateSet& preds,
                                          const CompositeKeySchema& schema,
                                          TxnId reader_txn) {
        std::vector<Record*> result;
        for (size_t i = 0; i < records.size(); ++i) {
            auto& r = records[i];
            if (r.tombstone || !r.version.is_visible(reader_txn)) continue;
            if (preds.evaluate_record(r.key, schema, &null_bitmap, i))
                result.push_back(&r);
        }
        return result;
    }

    // Fast path: skip MVCC/tombstone checks. Caller guarantees clean.
    void predicate_search_fast(const PredicateSet& preds,
                               const CompositeKeySchema& schema,
                               std::vector<Record*>& out) {
        const size_t n = records.size();
        for (size_t i = 0; i < n; ++i) {
            auto& r = records[i];
            if (preds.evaluate_record(r.key, schema, &null_bitmap, i))
                out.push_back(&r);
        }
    }

    std::pair<std::unique_ptr<LeafNode>, std::unique_ptr<LeafNode>>
    split(const CompositeKeySchema& schema) {
        auto left  = std::make_unique<LeafNode>();
        auto right = std::make_unique<LeafNode>();

        size_t mid = records.size() / 2;

        left->records.assign(records.begin(), records.begin() + mid);
        right->records.assign(records.begin() + mid, records.end());

        left->meta.record_count = left->records.size();
        right->meta.record_count = right->records.size();
        left->meta.depth = meta.depth;
        right->meta.depth = meta.depth;

        left->update_range();
        right->update_range();

        left->null_bitmap.init(left->records.size(), schema.dim_count());
        right->null_bitmap.init(right->records.size(), schema.dim_count());

        left->recompute_aggregates(schema);
        right->recompute_aggregates(schema);

        left->recompute_beta();
        right->recompute_beta();

        left->next_leaf = right->meta.id;
        right->prev_leaf = left->meta.id;
        left->prev_leaf = prev_leaf;
        right->next_leaf = next_leaf;

        return {std::move(left), std::move(right)};
    }

    void merge_from(LeafNode& other) {
        records.insert(records.end(), other.records.begin(), other.records.end());
        std::sort(records.begin(), records.end());
        meta.record_count = records.size();
        meta.tombstone_count += other.meta.tombstone_count;
        next_leaf = other.next_leaf;
        update_range();
    }

    void recompute_beta() {
        if (records.empty()) {
            meta.beta_value = 0.0;
            return;
        }
        meta.beta_value = BetaComputer::compute_beta(
            records.front().key, records.back().key);
    }

    void recompute_aggregates(const CompositeKeySchema& schema) {
        aggregates.init(schema.dim_count());
        for (size_t i = 0; i < records.size(); ++i) {
            if (!records[i].tombstone)
                aggregates.add_record(records[i].key, schema, &null_bitmap, i);
        }
    }

    void mark_homogeneous() {
        meta.type = NodeType::HOMOGENEOUS_LEAF;
        meta.is_homogeneous = true;
    }

    void sort_records() {
        std::sort(records.begin(), records.end());
        update_range();
    }

    size_t memory_usage() const override {
        size_t s = sizeof(LeafNode);
        s += records.capacity() * sizeof(Record);
        for (auto& r : records) s += r.payload.capacity();
        s += null_bitmap.words.capacity() * sizeof(uint64_t);
        s += aggregates.dims.capacity() * sizeof(AggregateStats);
        return s;
    }

private:
    void update_range() {
        if (records.empty()) {
            meta.range.low = COMPOSITE_KEY_MAX;
            meta.range.high = COMPOSITE_KEY_MIN;
        } else {
            meta.range.low  = records.front().key;
            meta.range.high = records.back().key;
        }
    }
};

class InternalNode : public HPNode {
public:
    std::vector<CompositeKey>           separator_keys;
    std::vector<std::shared_ptr<HPNode>> children;

    InternalNode() { meta.type = NodeType::INTERNAL; }

    size_t find_child_index(CompositeKey key) const {
        auto it = std::upper_bound(separator_keys.begin(), separator_keys.end(), key);
        return static_cast<size_t>(it - separator_keys.begin());
    }

    HPNode* find_child(CompositeKey key) const {
        size_t idx = find_child_index(key);
        if (idx < children.size()) return children[idx].get();
        return nullptr;
    }

    void insert_child(size_t pos, CompositeKey sep, std::shared_ptr<HPNode> child) {
        if (pos > 0 || !separator_keys.empty()) {
            if (pos <= separator_keys.size())
                separator_keys.insert(separator_keys.begin() + pos, sep);
            else
                separator_keys.push_back(sep);
        }
        if (pos <= children.size())
            children.insert(children.begin() + pos, std::move(child));
        else
            children.push_back(std::move(child));
        update_meta();
    }

    void replace_child(size_t pos, std::shared_ptr<HPNode> new_child) {
        if (pos < children.size())
            children[pos] = std::move(new_child);
        update_meta();
    }

    void remove_child(size_t pos) {
        if (pos < children.size()) {
            children.erase(children.begin() + pos);
            if (pos > 0 && pos - 1 < separator_keys.size())
                separator_keys.erase(separator_keys.begin() + pos - 1);
            else if (!separator_keys.empty() && pos < separator_keys.size())
                separator_keys.erase(separator_keys.begin() + pos);
        }
        update_meta();
    }

    std::pair<std::unique_ptr<InternalNode>, std::unique_ptr<InternalNode>>
    split() {
        auto left  = std::make_unique<InternalNode>();
        auto right = std::make_unique<InternalNode>();

        size_t mid = separator_keys.size() / 2;

        left->separator_keys.assign(
            separator_keys.begin(), separator_keys.begin() + mid);
        right->separator_keys.assign(
            separator_keys.begin() + mid + 1, separator_keys.end());

        left->children.assign(
            children.begin(), children.begin() + mid + 1);
        right->children.assign(
            children.begin() + mid + 1, children.end());

        left->meta.depth = meta.depth;
        right->meta.depth = meta.depth;

        left->update_meta();
        right->update_meta();

        return {std::move(left), std::move(right)};
    }

    CompositeKey get_split_key() const {
        if (separator_keys.empty()) return 0;
        return separator_keys[separator_keys.size() / 2];
    }

    void merge_from(InternalNode& other, CompositeKey merge_key) {
        separator_keys.push_back(merge_key);
        separator_keys.insert(separator_keys.end(),
            other.separator_keys.begin(), other.separator_keys.end());
        children.insert(children.end(),
            other.children.begin(), other.children.end());
        update_meta();
    }

    bool is_overfull(size_t max_keys) const {
        return separator_keys.size() >= max_keys;
    }

    bool is_underfull(size_t min_keys) const {
        return separator_keys.size() < min_keys;
    }

    size_t child_count() const { return children.size(); }

    void recompute_range() {
        if (children.empty()) return;
        meta.range.low = children.front()->meta.range.low;
        meta.range.high = children.back()->meta.range.high;
        for (auto& c : children) {
            if (c->meta.range.low < meta.range.low)
                meta.range.low = c->meta.range.low;
            if (c->meta.range.high > meta.range.high)
                meta.range.high = c->meta.range.high;
        }
    }

    void recompute_aggregates_from_children() {
        if (children.empty()) return;
        size_t nd = children[0]->aggregates.dims.size();
        aggregates.init(nd);
        meta.record_count = 0;
        for (auto& c : children) {
            aggregates.merge(c->aggregates);
            meta.record_count += c->meta.record_count;
        }
    }

    size_t memory_usage() const override {
        size_t s = sizeof(InternalNode);
        s += separator_keys.capacity() * sizeof(CompositeKey);
        s += children.capacity() * sizeof(std::shared_ptr<HPNode>);
        for (auto& c : children) {
            if (c) s += c->memory_usage();
        }
        return s;
    }

private:
    void update_meta() {
        meta.record_count = 0;
        for (auto& c : children) {
            if (c) meta.record_count += c->meta.record_count;
        }
        recompute_range();
    }
};

inline std::vector<LeafNode*> collect_leaves(HPNode* root) {
    std::vector<LeafNode*> leaves;
    std::function<void(HPNode*)> walk = [&](HPNode* node) {
        if (!node) return;
        if (node->is_leaf()) {
            leaves.push_back(static_cast<LeafNode*>(node));
        } else {
            auto* internal = static_cast<InternalNode*>(node);
            for (auto& c : internal->children) walk(c.get());
        }
    };
    walk(root);
    return leaves;
}

inline LeafNode* find_leftmost_leaf(HPNode* node) {
    if (!node) return nullptr;
    if (node->is_leaf()) return static_cast<LeafNode*>(node);
    auto* internal = static_cast<InternalNode*>(node);
    if (internal->children.empty()) return nullptr;
    return find_leftmost_leaf(internal->children.front().get());
}

inline LeafNode* find_rightmost_leaf(HPNode* node) {
    if (!node) return nullptr;
    if (node->is_leaf()) return static_cast<LeafNode*>(node);
    auto* internal = static_cast<InternalNode*>(node);
    if (internal->children.empty()) return nullptr;
    return find_rightmost_leaf(internal->children.back().get());
}

}  // namespace hptree
