#pragma once

#include "hp_tree_common.hpp"
#include "hp_tree_node.hpp"
#include "hp_tree_wal.hpp"
#include "hp_tree_buffer_pool.hpp"
#include "hp_tree_delta_buffer.hpp"
#include "hp_tree_iterator.hpp"
#include "hp_tree_stats.hpp"

namespace hptree {

class HPTree {
    HPTreeConfig           config_;
    CompositeKeySchema     schema_;
    std::shared_ptr<HPNode> root_;
    std::unique_ptr<WalManager> wal_;
    DeltaBuffer            delta_buffer_;
    StatisticsCollector    stats_collector_;
    BetaComputer::Thresholds thresholds_;

    std::unique_ptr<DiskManager> disk_mgr_;
    std::unique_ptr<BufferPool>  buffer_pool_;

    std::atomic<NodeId>  next_node_id_{1};
    std::atomic<TxnId>   next_txn_id_{1};
    std::atomic<uint64_t> total_records_{0};
    std::atomic<uint64_t> total_tombstones_{0};
    std::atomic<uint64_t> total_splits_{0};
    std::atomic<uint64_t> total_merges_{0};
    std::atomic<uint64_t> total_rebalances_{0};
    std::atomic<uint64_t> delta_flushes_{0};

    mutable std::shared_mutex tree_latch_;

    std::unordered_map<NodeId, LeafNode*> leaf_map_;
    std::mutex leaf_map_mtx_;

    NodeId alloc_node_id() { return next_node_id_.fetch_add(1); }

    void register_leaf(LeafNode* leaf) {
        std::lock_guard<std::mutex> lock(leaf_map_mtx_);
        leaf_map_[leaf->meta.id] = leaf;
    }

    void unregister_leaf(NodeId id) {
        std::lock_guard<std::mutex> lock(leaf_map_mtx_);
        leaf_map_.erase(id);
    }

    void rebuild_leaf_map() {
        std::lock_guard<std::mutex> lock(leaf_map_mtx_);
        leaf_map_.clear();
        if (!root_) return;
        auto leaves = collect_leaves(root_.get());
        for (auto* l : leaves) leaf_map_[l->meta.id] = l;
    }

    void link_leaves() {
        auto leaves = collect_leaves(root_.get());
        for (size_t i = 0; i < leaves.size(); ++i) {
            leaves[i]->prev_leaf = (i > 0) ? leaves[i - 1]->meta.id : INVALID_NODE;
            leaves[i]->next_leaf = (i + 1 < leaves.size())
                                 ? leaves[i + 1]->meta.id : INVALID_NODE;
            leaves[i]->prev_leaf_ptr = (i > 0) ? leaves[i - 1] : nullptr;
            leaves[i]->next_leaf_ptr = (i + 1 < leaves.size()) ? leaves[i + 1] : nullptr;
        }
    }

    std::shared_ptr<LeafNode> make_leaf() {
        auto leaf = std::make_shared<LeafNode>();
        leaf->meta.id = alloc_node_id();
        leaf->null_bitmap.init(0, schema_.dim_count());
        if (config_.enable_aggregates)
            leaf->aggregates.init(schema_.dim_count());
        return leaf;
    }

    std::shared_ptr<InternalNode> make_internal() {
        auto node = std::make_shared<InternalNode>();
        node->meta.id = alloc_node_id();
        if (config_.enable_aggregates)
            node->aggregates.init(schema_.dim_count());
        return node;
    }

    double get_active_threshold() const {
        return BetaComputer::select_threshold(
            thresholds_, config_.beta_strategy, config_.beta_strict);
    }

    bool should_stop_splitting(const std::vector<Record>& recs,
                               double parent_beta = std::numeric_limits<double>::infinity(),
                               size_t num_children = 2) const {
        if (recs.size() <= config_.max_leaf_size) return true;
        if (recs.empty()) return true;
        double beta = BetaComputer::compute_beta(recs.front().key, recs.back().key);
        if (config_.beta_strategy == BetaStrategy::ADAPTIVE_LOCAL) {
            return BetaComputer::adaptive_should_stop(
                beta, parent_beta, recs.size(), num_children);
        }
        return beta < get_active_threshold();
    }

    std::shared_ptr<HPNode> build_recursive(std::vector<Record>& recs,
                                            uint32_t depth,
                                            double parent_beta = std::numeric_limits<double>::infinity()) {
        size_t num_children_hint = std::min(config_.branching_factor, recs.size());
        if (num_children_hint < 2) num_children_hint = 2;

        if (recs.size() <= config_.max_leaf_size || depth >= MAX_TREE_DEPTH
            || should_stop_splitting(recs, parent_beta, num_children_hint)) {
            auto leaf = make_leaf();
            leaf->meta.depth = depth;
            leaf->records = std::move(recs);
            leaf->meta.record_count = leaf->records.size();
            leaf->null_bitmap.init(leaf->records.size(), schema_.dim_count());
            leaf->recompute_beta();
            if (config_.enable_aggregates)
                leaf->recompute_aggregates(schema_);
            bool is_homo = (config_.beta_strategy == BetaStrategy::ADAPTIVE_LOCAL)
                ? BetaComputer::adaptive_should_stop(
                      leaf->meta.beta_value, parent_beta,
                      leaf->records.size(), num_children_hint)
                : (leaf->meta.beta_value < get_active_threshold());
            if (is_homo && leaf->records.size() > 1) {
                leaf->mark_homogeneous();
            }
            leaf->sort_records();
            register_leaf(leaf.get());
            return leaf;
        }

        std::sort(recs.begin(), recs.end());

        double current_beta = BetaComputer::compute_beta(
            recs.front().key, recs.back().key);

        size_t num_children = std::min(config_.branching_factor, recs.size());
        if (num_children < 2) num_children = 2;
        size_t per_child = recs.size() / num_children;
        size_t remainder = recs.size() % num_children;

        auto internal = make_internal();
        internal->meta.depth = depth;

        size_t pos = 0;
        for (size_t i = 0; i < num_children; ++i) {
            size_t chunk_size = per_child + (i < remainder ? 1 : 0);
            if (chunk_size == 0) continue;

            std::vector<Record> chunk(
                std::make_move_iterator(recs.begin() + pos),
                std::make_move_iterator(recs.begin() + pos + chunk_size));
            pos += chunk_size;

            auto child = build_recursive(chunk, depth + 1, current_beta);

            if (i > 0 && !internal->children.empty()) {
                internal->separator_keys.push_back(child->meta.range.low);
            }
            internal->children.push_back(std::move(child));
        }

        internal->recompute_range();
        if (config_.enable_aggregates)
            internal->recompute_aggregates_from_children();
        internal->meta.record_count = 0;
        for (auto& c : internal->children)
            internal->meta.record_count += c->meta.record_count;

        return internal;
    }

    struct InsertResult {
        bool     split_occurred = false;
        CompositeKey split_key  = 0;
        std::shared_ptr<HPNode> new_sibling;
    };

    InsertResult insert_recursive(HPNode* node, const Record& rec, TxnId txn) {
        InsertResult result;

        if (node->is_leaf()) {
            auto* leaf = static_cast<LeafNode*>(node);
            leaf->latch.lock();

            Record r = rec;
            r.version.xmin = txn;
            r.version.xmax = TXN_COMMITTED;
            leaf->insert_record(r);

            if (config_.enable_aggregates)
                leaf->aggregates.add_record(r.key, schema_, &leaf->null_bitmap,
                                            leaf->records.size() - 1);
            leaf->recompute_beta();

            if (leaf->records.size() > config_.max_leaf_size) {
                auto [left, right] = leaf->split(schema_);
                left->meta.id = node->meta.id;
                right->meta.id = alloc_node_id();

                result.split_occurred = true;
                result.split_key = right->records.front().key;
                result.new_sibling = std::move(right);

                register_leaf(static_cast<LeafNode*>(result.new_sibling.get()));

                leaf->records = std::move(left->records);
                leaf->meta = left->meta;
                leaf->null_bitmap = std::move(left->null_bitmap);
                leaf->aggregates = std::move(left->aggregates);
                leaf->next_leaf = result.new_sibling->meta.id;

                auto* right_leaf = static_cast<LeafNode*>(result.new_sibling.get());
                right_leaf->prev_leaf = leaf->meta.id;

                if (config_.enable_wal) {
                    wal_->log_split(txn, leaf->meta.id, right_leaf->meta.id);
                }
                total_splits_++;
            }

            leaf->latch.unlock();
            return result;
        }

        auto* internal = static_cast<InternalNode*>(node);
        size_t child_idx = internal->find_child_index(rec.key);
        if (child_idx >= internal->children.size())
            child_idx = internal->children.size() - 1;

        auto child_result = insert_recursive(
            internal->children[child_idx].get(), rec, txn);

        if (child_result.split_occurred) {
            internal->latch.lock();

            size_t insert_pos = child_idx + 1;
            if (insert_pos <= internal->separator_keys.size()) {
                internal->separator_keys.insert(
                    internal->separator_keys.begin() + child_idx,
                    child_result.split_key);
                internal->children.insert(
                    internal->children.begin() + insert_pos,
                    std::move(child_result.new_sibling));
            } else {
                internal->separator_keys.push_back(child_result.split_key);
                internal->children.push_back(
                    std::move(child_result.new_sibling));
            }

            internal->recompute_range();
            if (config_.enable_aggregates)
                internal->recompute_aggregates_from_children();

            if (internal->separator_keys.size() >= config_.branching_factor) {
                auto [left, right] = internal->split();
                left->meta.id = internal->meta.id;
                right->meta.id = alloc_node_id();

                CompositeKey split_key = internal->get_split_key();

                result.split_occurred = true;
                result.split_key = split_key;
                result.new_sibling = std::move(right);

                internal->separator_keys = std::move(left->separator_keys);
                internal->children = std::move(left->children);
                internal->recompute_range();
                if (config_.enable_aggregates)
                    internal->recompute_aggregates_from_children();
            }

            internal->latch.unlock();
        } else {
            internal->recompute_range();
            if (config_.enable_aggregates)
                internal->recompute_aggregates_from_children();
        }

        return result;
    }

    bool delete_recursive(HPNode* node, CompositeKey key, TxnId txn) {
        if (node->is_leaf()) {
            auto* leaf = static_cast<LeafNode*>(node);
            leaf->latch.lock();
            bool removed = leaf->remove_record(key, txn);
            if (removed) {
                if (config_.enable_wal) {
                    wal_->log_delete(txn, INVALID_PAGE, leaf->meta.id, key, {});
                }
                if (config_.enable_aggregates)
                    leaf->recompute_aggregates(schema_);
            }
            leaf->latch.unlock();
            return removed;
        }

        auto* internal = static_cast<InternalNode*>(node);
        size_t child_idx = internal->find_child_index(key);
        if (child_idx >= internal->children.size()) return false;

        bool removed = delete_recursive(internal->children[child_idx].get(), key, txn);

        if (removed) {
            auto* child = internal->children[child_idx].get();
            if (child->is_leaf()) {
                auto* child_leaf = static_cast<LeafNode*>(child);
                if (child_leaf->needs_compaction())
                    child_leaf->compact();

                if (child_leaf->records.size() < config_.min_leaf_size
                    && internal->children.size() > 1) {
                    try_rebalance_or_merge(internal, child_idx);
                }
            }
            internal->recompute_range();
            if (config_.enable_aggregates)
                internal->recompute_aggregates_from_children();
        }

        return removed;
    }

    void try_rebalance_or_merge(InternalNode* parent, size_t child_idx) {
        if (child_idx >= parent->children.size()) return;
        auto* target = parent->children[child_idx].get();
        if (!target->is_leaf()) return;
        auto* target_leaf = static_cast<LeafNode*>(target);

        if (child_idx + 1 < parent->children.size()) {
            auto* right = parent->children[child_idx + 1].get();
            if (right->is_leaf()) {
                auto* right_leaf = static_cast<LeafNode*>(right);
                size_t combined = target_leaf->records.size() + right_leaf->records.size();
                if (combined <= config_.max_leaf_size) {
                    target_leaf->latch.lock();
                    right_leaf->latch.lock();
                    target_leaf->merge_from(*right_leaf);
                    target_leaf->recompute_beta();
                    if (config_.enable_aggregates)
                        target_leaf->recompute_aggregates(schema_);
                    unregister_leaf(right_leaf->meta.id);
                    right_leaf->latch.unlock();
                    target_leaf->latch.unlock();
                    parent->remove_child(child_idx + 1);
                    total_merges_++;
                    if (config_.enable_wal)
                        wal_->log_merge(INVALID_TXN, target_leaf->meta.id,
                                       right_leaf->meta.id);
                    return;
                }

                if (right_leaf->records.size() > config_.min_leaf_size + 1) {
                    target_leaf->latch.lock();
                    right_leaf->latch.lock();
                    auto& donor_rec = right_leaf->records.front();
                    target_leaf->insert_record(donor_rec);
                    right_leaf->records.erase(right_leaf->records.begin());
                    right_leaf->meta.record_count = right_leaf->records.size();
                    if (child_idx < parent->separator_keys.size())
                        parent->separator_keys[child_idx] =
                            right_leaf->records.front().key;
                    right_leaf->latch.unlock();
                    target_leaf->latch.unlock();
                    total_rebalances_++;
                    return;
                }
            }
        }

        if (child_idx > 0) {
            auto* left = parent->children[child_idx - 1].get();
            if (left->is_leaf()) {
                auto* left_leaf = static_cast<LeafNode*>(left);
                size_t combined = target_leaf->records.size() + left_leaf->records.size();
                if (combined <= config_.max_leaf_size) {
                    left_leaf->latch.lock();
                    target_leaf->latch.lock();
                    left_leaf->merge_from(*target_leaf);
                    left_leaf->recompute_beta();
                    if (config_.enable_aggregates)
                        left_leaf->recompute_aggregates(schema_);
                    unregister_leaf(target_leaf->meta.id);
                    target_leaf->latch.unlock();
                    left_leaf->latch.unlock();
                    parent->remove_child(child_idx);
                    total_merges_++;
                    return;
                }

                if (left_leaf->records.size() > config_.min_leaf_size + 1) {
                    left_leaf->latch.lock();
                    target_leaf->latch.lock();
                    auto& donor_rec = left_leaf->records.back();
                    target_leaf->insert_record(donor_rec);
                    left_leaf->records.pop_back();
                    left_leaf->meta.record_count = left_leaf->records.size();
                    if (child_idx - 1 < parent->separator_keys.size())
                        parent->separator_keys[child_idx - 1] =
                            target_leaf->records.front().key;
                    target_leaf->latch.unlock();
                    left_leaf->latch.unlock();
                    total_rebalances_++;
                    return;
                }
            }
        }
    }

    std::vector<Record*> search_recursive(HPNode* node, CompositeKey key,
                                          TxnId txn) const {
        if (!node) return {};
        if (node->is_leaf()) {
            auto* leaf = static_cast<LeafNode*>(node);
            leaf->latch.lock_shared();
            auto result = leaf->search(key, txn);
            leaf->latch.unlock_shared();
            return result;
        }
        auto* internal = static_cast<InternalNode*>(node);
        internal->latch.lock_shared();
        size_t idx = internal->find_child_index(key);
        if (idx >= internal->children.size()) {
            internal->latch.unlock_shared();
            return {};
        }
        auto* child = internal->children[idx].get();
        internal->latch.unlock_shared();
        return search_recursive(child, key, txn);
    }

    // Fast path: no per-node latches, no MVCC check. Caller guarantees
    // single-threaded access and a clean tree (no tombstones, reader=COMMITTED).
    std::vector<Record*> search_fast_recursive(HPNode* node,
                                               CompositeKey key) const {
        while (node && !node->is_leaf()) {
            auto* internal = static_cast<InternalNode*>(node);
            size_t idx = internal->find_child_index(key);
            if (idx >= internal->children.size()) return {};
            node = internal->children[idx].get();
        }
        if (!node) return {};
        return static_cast<LeafNode*>(node)->search_fast(key);
    }

    void range_search_recursive(HPNode* node, const KeyRange& range,
                                TxnId txn, std::vector<Record*>& results) const {
        if (!node) return;
        if (!node->meta.range.overlaps(range)) return;

        if (node->is_leaf()) {
            auto* leaf = static_cast<LeafNode*>(node);
            leaf->latch.lock_shared();
            auto partial = leaf->range_search(range, txn);
            results.insert(results.end(), partial.begin(), partial.end());
            leaf->latch.unlock_shared();
            return;
        }

        auto* internal = static_cast<InternalNode*>(node);
        internal->latch.lock_shared();
        for (auto& child : internal->children) {
            if (child->meta.range.overlaps(range))
                range_search_recursive(child.get(), range, txn, results);
        }
        internal->latch.unlock_shared();
    }

    // Fast path: no per-node latches, no MVCC check. Caller guarantees
    // single-threaded access and a clean tree.
    void range_search_fast_recursive(HPNode* node, const KeyRange& range,
                                     std::vector<Record*>& results) const {
        if (!node) return;
        if (!node->meta.range.overlaps(range)) return;
        if (node->is_leaf()) {
            static_cast<LeafNode*>(node)->range_search_fast(range, results);
            return;
        }
        auto* internal = static_cast<InternalNode*>(node);
        for (auto& child : internal->children) {
            if (child->meta.range.overlaps(range))
                range_search_fast_recursive(child.get(), range, results);
        }
    }

    static bool predicates_can_match_node(const PredicateSet& preds,
                                          const HPNode* node) {
        const auto& dims = node->aggregates.dims;
        if (dims.empty()) return true;  // aggregates not populated
        for (const auto& p : preds.predicates) {
            if (p.dim_idx >= dims.size()) continue;
            const auto& stats = dims[p.dim_idx];
            if (stats.count_non_null == 0) {
                // only nulls or empty: EQ/BETWEEN/ranges on a non-null value cannot match
                switch (p.op) {
                case PredicateOp::EQ:
                case PredicateOp::NEQ:
                case PredicateOp::LT:
                case PredicateOp::LTE:
                case PredicateOp::GT:
                case PredicateOp::GTE:
                case PredicateOp::BETWEEN:
                case PredicateOp::IN:
                case PredicateOp::IS_NOT_NULL:
                    return false;
                default: break;
                }
                continue;
            }
            uint64_t dmin = static_cast<uint64_t>(stats.min_val);
            uint64_t dmax = static_cast<uint64_t>(stats.max_val);
            switch (p.op) {
            case PredicateOp::EQ: {
                uint64_t v = static_cast<uint64_t>(p.value);
                if (v < dmin || v > dmax) return false;
                break;
            }
            case PredicateOp::BETWEEN: {
                uint64_t lo = static_cast<uint64_t>(p.value);
                uint64_t hi = static_cast<uint64_t>(p.value_high);
                if (hi < dmin || lo > dmax) return false;
                break;
            }
            case PredicateOp::LT: {
                uint64_t v = static_cast<uint64_t>(p.value);
                if (dmin >= v) return false;
                break;
            }
            case PredicateOp::LTE: {
                uint64_t v = static_cast<uint64_t>(p.value);
                if (dmin > v) return false;
                break;
            }
            case PredicateOp::GT: {
                uint64_t v = static_cast<uint64_t>(p.value);
                if (dmax <= v) return false;
                break;
            }
            case PredicateOp::GTE: {
                uint64_t v = static_cast<uint64_t>(p.value);
                if (dmax < v) return false;
                break;
            }
            case PredicateOp::IN: {
                bool any = false;
                for (auto& vv : p.in_values) {
                    uint64_t v = static_cast<uint64_t>(vv);
                    if (v >= dmin && v <= dmax) { any = true; break; }
                }
                if (!any) return false;
                break;
            }
            default: break;
            }
        }
        return true;
    }

    void predicate_search_recursive(HPNode* node, const PredicateSet& preds,
                                    TxnId txn,
                                    std::vector<Record*>& results,
                                    bool clean = false) const {
        if (!node) return;

        KeyRange approx_range = preds.to_key_range(schema_);
        if (!node->meta.range.overlaps(approx_range)) return;
        if (!predicates_can_match_node(preds, node)) return;

        if (node->is_leaf()) {
            auto* leaf = static_cast<LeafNode*>(node);
            if (clean) {
                leaf->predicate_search_fast(preds, schema_, results);
            } else {
                leaf->latch.lock_shared();
                auto partial = leaf->predicate_search(preds, schema_, txn);
                results.insert(results.end(), partial.begin(), partial.end());
                leaf->latch.unlock_shared();
            }
            return;
        }

        auto* internal = static_cast<InternalNode*>(node);
        if (clean) {
            for (auto& child : internal->children) {
                if (!child->meta.range.overlaps(approx_range)) continue;
                if (!predicates_can_match_node(preds, child.get())) continue;
                predicate_search_recursive(child.get(), preds, txn, results, true);
            }
        } else {
            internal->latch.lock_shared();
            for (auto& child : internal->children) {
                if (!child->meta.range.overlaps(approx_range)) continue;
                if (!predicates_can_match_node(preds, child.get())) continue;
                predicate_search_recursive(child.get(), preds, txn, results, false);
            }
            internal->latch.unlock_shared();
        }
    }

    void flush_delta_buffer_locked() {
        if (delta_buffer_.is_empty()) return;

        auto entries = delta_buffer_.drain();
        delta_flushes_++;

        if (config_.enable_wal)
            wal_->append(WalOpType::DELTA_FLUSH, INVALID_TXN,
                        INVALID_PAGE, INVALID_NODE, 0);

        for (auto& entry : entries) {
            switch (entry.op) {
            case DeltaOpType::INSERT:
                if (root_) {
                    auto res = insert_recursive(root_.get(), entry.record, entry.txn_id);
                    if (res.split_occurred) grow_root(res);
                    total_records_++;
                }
                break;
            case DeltaOpType::DELETE:
                if (root_)
                    if (delete_recursive(root_.get(), entry.record.key, entry.txn_id))
                        total_records_--;
                break;
            case DeltaOpType::UPDATE:
                if (root_) {
                    delete_recursive(root_.get(), entry.old_record.key, entry.txn_id);
                    auto res = insert_recursive(root_.get(), entry.record, entry.txn_id);
                    if (res.split_occurred) grow_root(res);
                }
                break;
            }
        }

        link_leaves();
        rebuild_leaf_map();
    }

    void grow_root(const InsertResult& res) {
        auto new_root = make_internal();
        new_root->children.push_back(root_);
        new_root->separator_keys.push_back(res.split_key);
        new_root->children.push_back(res.new_sibling);
        new_root->recompute_range();
        if (config_.enable_aggregates)
            new_root->recompute_aggregates_from_children();
        new_root->meta.record_count = 0;
        for (auto& c : new_root->children)
            new_root->meta.record_count += c->meta.record_count;
        root_ = std::move(new_root);
    }

    void recompute_thresholds() {
        std::vector<CompositeKey> all_keys;
        if (root_) {
            auto leaves = collect_leaves(root_.get());
            for (auto* l : leaves) {
                for (auto& r : l->records) {
                    if (!r.tombstone) all_keys.push_back(r.key);
                }
            }
        }
        thresholds_ = BetaComputer::compute_dynamic_thresholds(
            all_keys, config_.beta_strict);
    }

public:
    HPTree() : delta_buffer_(DEFAULT_DELTA_BUFFER_CAP) {
        schema_ = make_default_sales_schema();
        stats_collector_.set_schema(&schema_);
        thresholds_ = {config_.beta_strict, config_.beta_strict,
                       config_.beta_strict, config_.beta_strict};
    }

    explicit HPTree(const HPTreeConfig& config)
        : config_(config), delta_buffer_(config.delta_buffer_cap) {
        schema_ = make_default_sales_schema();
        stats_collector_.set_schema(&schema_);
        thresholds_ = {config_.beta_strict, config_.beta_strict,
                       config_.beta_strict, config_.beta_strict};

        if (config_.enable_wal)
            wal_ = std::make_unique<WalManager>(config_.wal_path, true);

        if (config_.enable_buffer_pool) {
            disk_mgr_ = std::make_unique<DiskManager>(
                config_.data_path, config_.page_size);
            buffer_pool_ = std::make_unique<BufferPool>(
                config_.buffer_pool_pages, disk_mgr_.get());
        }
    }

    HPTree(const HPTreeConfig& config, const CompositeKeySchema& schema)
        : config_(config), schema_(schema), delta_buffer_(config.delta_buffer_cap) {
        stats_collector_.set_schema(&schema_);
        thresholds_ = {config_.beta_strict, config_.beta_strict,
                       config_.beta_strict, config_.beta_strict};

        if (config_.enable_wal)
            wal_ = std::make_unique<WalManager>(config_.wal_path, true);

        if (config_.enable_buffer_pool) {
            disk_mgr_ = std::make_unique<DiskManager>(
                config_.data_path, config_.page_size);
            buffer_pool_ = std::make_unique<BufferPool>(
                config_.buffer_pool_pages, disk_mgr_.get());
        }
    }

    ~HPTree() = default;

    HPTree(HPTree&&) = delete;
    HPTree& operator=(HPTree&&) = delete;

    // ======================================================================
    //  BULK LOAD
    // ======================================================================
    void bulk_load(std::vector<Record> records) {
        std::unique_lock<std::shared_mutex> lock(tree_latch_);

        if (records.empty()) {
            root_ = make_leaf();
            link_leaves();
            rebuild_leaf_map();
            return;
        }

        for (auto& r : records) {
            if (r.version.xmin == INVALID_TXN) r.version.xmin = 1;
            if (r.version.xmax == INVALID_TXN) r.version.xmax = TXN_COMMITTED;
        }

        std::sort(records.begin(), records.end());
        total_records_ = records.size();

        std::vector<CompositeKey> keys;
        keys.reserve(records.size());
        for (auto& r : records) keys.push_back(r.key);
        thresholds_ = BetaComputer::compute_dynamic_thresholds(
            keys, config_.beta_strict);

        root_ = build_recursive(records, 0);
        link_leaves();
        rebuild_leaf_map();

        if (config_.enable_wal) {
            wal_->append(WalOpType::BULK_LOAD, INVALID_TXN,
                        INVALID_PAGE, INVALID_NODE,
                        static_cast<CompositeKey>(total_records_.load()));
        }
    }

    // ======================================================================
    //  INSERT
    // ======================================================================
    bool insert(const Record& rec) {
        return insert(rec, next_txn_id_.fetch_add(1));
    }

    bool insert(const Record& rec, TxnId txn) {
        if (config_.enable_delta_buffer) {
            Record r = rec;
            r.version.xmin = txn;
            r.version.xmax = TXN_COMMITTED;
            delta_buffer_.add_insert(std::move(r), txn);
            if (delta_buffer_.needs_flush(total_records_.load())) {
                if (config_.single_threaded) {
                    flush_delta_buffer_locked();
                } else {
                    std::unique_lock<std::shared_mutex> lock(tree_latch_);
                    flush_delta_buffer_locked();
                }
            }
            return true;
        }

        auto do_insert = [&]() -> bool {
            if (!root_) {
                auto leaf = make_leaf();
                Record r = rec;
                r.version.xmin = txn;
                r.version.xmax = TXN_COMMITTED;
                leaf->insert_record(r);
                leaf->recompute_beta();
                if (config_.enable_aggregates)
                    leaf->recompute_aggregates(schema_);
                root_ = std::move(leaf);
                register_leaf(static_cast<LeafNode*>(root_.get()));
                total_records_++;
                if (config_.enable_wal)
                    wal_->log_insert(txn, INVALID_PAGE, root_->meta.id, rec.key, {});
                return true;
            }

            if (config_.enable_wal)
                wal_->log_insert(txn, INVALID_PAGE, INVALID_NODE, rec.key, {});

            auto result = insert_recursive(root_.get(), rec, txn);
            if (result.split_occurred) grow_root(result);

            total_records_++;
            return true;
        };

        if (config_.single_threaded) return do_insert();
        std::unique_lock<std::shared_mutex> lock(tree_latch_);
        return do_insert();
    }

    // ======================================================================
    //  DELETE
    // ======================================================================
    bool remove(CompositeKey key) {
        return remove(key, next_txn_id_.fetch_add(1));
    }

    bool remove(CompositeKey key, TxnId txn) {
        if (config_.enable_delta_buffer) {
            delta_buffer_.add_delete(key, txn);
            total_tombstones_.fetch_add(1, std::memory_order_relaxed);
            if (delta_buffer_.needs_flush(total_records_.load())) {
                if (config_.single_threaded) {
                    flush_delta_buffer_locked();
                } else {
                    std::unique_lock<std::shared_mutex> lock(tree_latch_);
                    flush_delta_buffer_locked();
                }
            }
            return true;
        }

        auto do_remove = [&]() -> bool {
            if (!root_) return false;
            bool removed = delete_recursive(root_.get(), key, txn);
            if (removed) {
                total_records_--;
                total_tombstones_.fetch_add(1, std::memory_order_relaxed);
            }

            if (root_->is_internal()) {
                auto* internal = static_cast<InternalNode*>(root_.get());
                if (internal->children.size() == 1)
                    root_ = internal->children[0];
            }
            return removed;
        };

        if (config_.single_threaded) return do_remove();
        std::unique_lock<std::shared_mutex> lock(tree_latch_);
        return do_remove();
    }

    // ======================================================================
    //  UPDATE
    // ======================================================================
    bool update(CompositeKey old_key, const Record& new_rec) {
        return update(old_key, new_rec, next_txn_id_.fetch_add(1));
    }

    bool update(CompositeKey old_key, const Record& new_rec, TxnId txn) {
        if (old_key == new_rec.key) {
            std::unique_lock<std::shared_mutex> lock(tree_latch_);
            if (!root_) return false;
            flush_if_needed_locked();
            return update_in_place(root_.get(), old_key, new_rec, txn);
        }

        if (config_.enable_delta_buffer) {
            Record old_rec;
            old_rec.key = old_key;
            delta_buffer_.add_update(old_rec, new_rec, txn);
            if (delta_buffer_.needs_flush(total_records_.load())) {
                std::unique_lock<std::shared_mutex> lock(tree_latch_);
                flush_delta_buffer_locked();
            }
            return true;
        }

        std::unique_lock<std::shared_mutex> lock(tree_latch_);
        if (!root_) return false;

        if (config_.enable_wal) {
            wal_->log_update(txn, INVALID_PAGE, INVALID_NODE, old_key, {}, {});
        }

        delete_recursive(root_.get(), old_key, txn);
        auto result = insert_recursive(root_.get(), new_rec, txn);
        if (result.split_occurred) grow_root(result);

        return true;
    }

    // ======================================================================
    //  SEARCH (point lookup)
    // ======================================================================
    std::vector<Record*> search(CompositeKey key) const {
        return search(key, TXN_COMMITTED);
    }

    std::vector<Record*> search(CompositeKey key, TxnId txn) const {
        const bool delta_empty = !config_.enable_delta_buffer
                              || delta_buffer_.is_empty();
        const bool clean = (total_tombstones_.load(std::memory_order_relaxed) == 0
                         && txn == TXN_COMMITTED);

        // Fast path: single-threaded + clean tree + empty delta buffer.
        if (config_.single_threaded && clean && delta_empty) {
            if (!root_) return {};
            return search_fast_recursive(root_.get(), key);
        }

        std::vector<Record*> combined;

        if (!delta_empty) {
            auto delta_results = const_cast<DeltaBuffer&>(delta_buffer_).search(key, txn);
            combined.insert(combined.end(), delta_results.begin(), delta_results.end());
        }

        auto descend = [&]() {
            if (!root_) return;
            std::vector<Record*> tree_results =
                (clean ? search_fast_recursive(root_.get(), key)
                       : search_recursive(root_.get(), key, txn));
            if (!delta_empty) {
                auto del_keys = delta_buffer_.deleted_keys();
                for (auto* r : tree_results) {
                    if (del_keys.find(r->key) == del_keys.end())
                        combined.push_back(r);
                }
            } else {
                combined.insert(combined.end(),
                                tree_results.begin(), tree_results.end());
            }
        };

        if (config_.single_threaded) {
            descend();
        } else {
            std::shared_lock<std::shared_mutex> lock(tree_latch_);
            descend();
        }
        return combined;
    }

    // ======================================================================
    //  RANGE SEARCH
    // ======================================================================
    std::vector<Record*> range_search(CompositeKey low, CompositeKey high) const {
        return range_search(low, high, TXN_COMMITTED);
    }

    std::vector<Record*> range_search(CompositeKey low, CompositeKey high,
                                      TxnId txn) const {
        KeyRange range{low, high};

        const bool delta_empty = !config_.enable_delta_buffer
                              || delta_buffer_.is_empty();
        const bool clean = (total_tombstones_.load(std::memory_order_relaxed) == 0
                         && txn == TXN_COMMITTED);

        // Fast path: single-threaded + clean tree + empty delta buffer.
        if (config_.single_threaded && clean && delta_empty) {
            std::vector<Record*> out;
            if (!root_) return out;
            range_search_fast_recursive(root_.get(), range, out);
            return out;
        }

        std::vector<Record*> combined;

        if (!delta_empty) {
            auto delta_results =
                const_cast<DeltaBuffer&>(delta_buffer_).range_search(range, txn);
            combined.insert(combined.end(), delta_results.begin(), delta_results.end());
        }

        auto descend = [&]() {
            if (!root_) return;
            std::vector<Record*> tree_results;
            if (clean) {
                range_search_fast_recursive(root_.get(), range, tree_results);
            } else {
                range_search_recursive(root_.get(), range, txn, tree_results);
            }
            if (!delta_empty) {
                auto del_keys = delta_buffer_.deleted_keys();
                for (auto* r : tree_results) {
                    if (del_keys.find(r->key) == del_keys.end())
                        combined.push_back(r);
                }
            } else {
                combined.insert(combined.end(),
                                tree_results.begin(), tree_results.end());
            }
        };

        if (config_.single_threaded) {
            descend();
        } else {
            std::shared_lock<std::shared_mutex> lock(tree_latch_);
            descend();
        }

        return combined;
    }

    // ======================================================================
    //  PREDICATE SEARCH (complex predicates: EQ, NEQ, IN, LIKE, NULL, etc.)
    // ======================================================================
    std::vector<Record*> predicate_search(const PredicateSet& preds) const {
        return predicate_search(preds, TXN_COMMITTED);
    }

    std::vector<Record*> predicate_search(const PredicateSet& preds,
                                          TxnId txn) const {
        std::vector<Record*> results;
        const bool clean = (total_tombstones_.load(std::memory_order_relaxed) == 0
                         && txn == TXN_COMMITTED);
        auto run = [&]() {
            if (root_)
                predicate_search_recursive(root_.get(), preds, txn, results, clean);
        };
        if (config_.single_threaded) {
            run();
        } else {
            std::shared_lock<std::shared_mutex> lock(tree_latch_);
            run();
        }
        return results;
    }

    // ======================================================================
    //  ITERATOR (forward/reverse scan via leaf linked list)
    // ======================================================================
    HPTreeIterator begin(CompositeKey start = COMPOSITE_KEY_MIN,
                         CompositeKey end = COMPOSITE_KEY_MAX,
                         TxnId txn = TXN_COMMITTED) {
        if (!delta_buffer_.is_empty()) flush_delta_if_needed();
        const bool clean = (total_tombstones_.load(std::memory_order_relaxed) == 0
                         && txn == TXN_COMMITTED);
        auto build = [&]() -> HPTreeIterator {
            if (!root_) return HPTreeIterator();
            auto* leaf = find_leaf_for_key(root_.get(), start);
            if (!leaf) return HPTreeIterator();
            size_t idx = 0;
            for (size_t i = 0; i < leaf->records.size(); ++i) {
                if (leaf->records[i].key >= start) { idx = i; break; }
            }
            return HPTreeIterator(leaf, idx, KeyRange{start, end}, txn,
                                  ScanDirection::FORWARD, nullptr, &schema_,
                                  &leaf_map_, clean);
        };
        if (config_.single_threaded) return build();
        std::shared_lock<std::shared_mutex> lock(tree_latch_);
        return build();
    }

    HPTreeIterator rbegin(CompositeKey start = COMPOSITE_KEY_MAX,
                          CompositeKey end = COMPOSITE_KEY_MIN,
                          TxnId txn = TXN_COMMITTED) {
        if (!delta_buffer_.is_empty()) flush_delta_if_needed();
        const bool clean = (total_tombstones_.load(std::memory_order_relaxed) == 0
                         && txn == TXN_COMMITTED);
        auto build = [&]() -> HPTreeIterator {
            if (!root_) return HPTreeIterator();
            auto* leaf = find_leaf_for_key(root_.get(), start);
            if (!leaf) {
                leaf = find_rightmost_leaf(root_.get());
                if (!leaf) return HPTreeIterator();
            }
            size_t idx = leaf->records.empty() ? 0 : leaf->records.size() - 1;
            for (size_t i = leaf->records.size(); i > 0; --i) {
                if (leaf->records[i - 1].key <= start) { idx = i - 1; break; }
            }
            return HPTreeIterator(leaf, idx, KeyRange{end, start}, txn,
                                  ScanDirection::REVERSE, nullptr, &schema_,
                                  &leaf_map_, clean);
        };
        if (config_.single_threaded) return build();
        std::shared_lock<std::shared_mutex> lock(tree_latch_);
        return build();
    }

    // ======================================================================
    //  AGGREGATE QUERIES
    // ======================================================================
    uint64_t count(CompositeKey low = COMPOSITE_KEY_MIN,
                   CompositeKey high = COMPOSITE_KEY_MAX,
                   TxnId txn = TXN_COMMITTED) const {
        auto results = range_search(low, high, txn);
        return results.size();
    }

    uint64_t count_predicate(const PredicateSet& preds,
                             TxnId txn = TXN_COMMITTED) const {
        auto results = predicate_search(preds, txn);
        return results.size();
    }

    struct AggResult {
        uint64_t count  = 0;
        double   sum    = 0;
        double   avg    = 0;
        double   min_v  = std::numeric_limits<double>::max();
        double   max_v  = std::numeric_limits<double>::lowest();
    };

    AggResult aggregate_dim(size_t dim_idx, CompositeKey low = COMPOSITE_KEY_MIN,
                            CompositeKey high = COMPOSITE_KEY_MAX,
                            TxnId txn = TXN_COMMITTED) const {
        AggResult agg;
        if (!root_) return agg;
        aggregate_dim_recursive(root_.get(), dim_idx, low, high, txn, agg);
        if (agg.count > 0) agg.avg = agg.sum / static_cast<double>(agg.count);
        return agg;
    }

private:
    void aggregate_dim_recursive(HPNode* node, size_t dim_idx,
                                 CompositeKey low, CompositeKey high,
                                 TxnId txn, AggResult& agg) const {
        if (!node) return;
        const auto& r = node->meta.range;
        if (r.high < low || r.low > high) return;

        bool fully_contained = (r.low >= low && r.high <= high);

        if (node->is_leaf()) {
            auto* leaf = static_cast<LeafNode*>(node);
            bool can_use_precomputed =
                config_.enable_aggregates
                && fully_contained
                && txn == TXN_COMMITTED
                && leaf->meta.tombstone_count == 0
                && dim_idx < leaf->aggregates.dims.size();
            if (can_use_precomputed) {
                const auto& stats = leaf->aggregates.dims[dim_idx];
                if (stats.count_non_null > 0) {
                    agg.count += stats.count_non_null;
                    agg.sum   += stats.sum;
                    if (stats.min_val < agg.min_v) agg.min_v = stats.min_val;
                    if (stats.max_val > agg.max_v) agg.max_v = stats.max_val;
                }
                return;
            }
        }

        if (node->is_leaf()) {
            auto* leaf = static_cast<LeafNode*>(node);
            CompositeKeyEncoder encoder(schema_);
            Record lo_rec{low}, hi_rec{high};
            auto start = std::lower_bound(leaf->records.begin(), leaf->records.end(),
                lo_rec, [](const Record& a, const Record& b){ return a.key < b.key; });
            auto end   = std::upper_bound(leaf->records.begin(), leaf->records.end(),
                hi_rec, [](const Record& a, const Record& b){ return a.key < b.key; });
            for (auto it = start; it != end; ++it) {
                if (it->tombstone || !it->version.is_visible(txn)) continue;
                double v = static_cast<double>(encoder.extract_dim(it->key, dim_idx));
                agg.count++;
                agg.sum += v;
                if (v < agg.min_v) agg.min_v = v;
                if (v > agg.max_v) agg.max_v = v;
            }
            return;
        }

        auto* internal = static_cast<InternalNode*>(node);
        for (auto& child : internal->children) {
            aggregate_dim_recursive(child.get(), dim_idx, low, high, txn, agg);
        }
    }

public:

    std::unordered_map<uint64_t, uint64_t>
    group_by_count(size_t dim_idx, const PredicateSet& preds = {},
                   TxnId txn = TXN_COMMITTED) const {
        std::unordered_map<uint64_t, uint64_t> groups;
        auto results = preds.predicates.empty()
            ? range_search(COMPOSITE_KEY_MIN, COMPOSITE_KEY_MAX, txn)
            : predicate_search(preds, txn);
        CompositeKeyEncoder encoder(schema_);
        for (auto* r : results) {
            uint64_t val = encoder.extract_dim(r->key, dim_idx);
            groups[val]++;
        }
        return groups;
    }

    // ======================================================================
    //  TRANSACTION SUPPORT
    // ======================================================================
    TxnId begin_transaction() {
        TxnId txn = next_txn_id_.fetch_add(1);
        if (config_.enable_wal) wal_->log_txn_begin(txn);
        return txn;
    }

    void commit_transaction(TxnId txn) {
        if (config_.enable_wal) wal_->log_txn_commit(txn);
    }

    void abort_transaction(TxnId txn) {
        if (config_.enable_wal) wal_->log_txn_abort(txn);
    }

    // ======================================================================
    //  MAINTENANCE
    // ======================================================================
    void flush_delta() {
        if (config_.single_threaded) {
            flush_delta_buffer_locked();
            return;
        }
        std::unique_lock<std::shared_mutex> lock(tree_latch_);
        flush_delta_buffer_locked();
    }

    void compact() {
        std::unique_lock<std::shared_mutex> lock(tree_latch_);
        if (!root_) return;
        auto leaves = collect_leaves(root_.get());
        for (auto* l : leaves) {
            if (l->needs_compaction()) {
                l->latch.lock();
                l->compact();
                l->recompute_beta();
                if (config_.enable_aggregates)
                    l->recompute_aggregates(schema_);
                l->latch.unlock();
            }
        }
    }

    void rebuild() {
        std::unique_lock<std::shared_mutex> lock(tree_latch_);
        flush_delta_buffer_locked();
        if (!root_) return;

        std::vector<Record> all_records;
        auto leaves = collect_leaves(root_.get());
        for (auto* l : leaves) {
            for (auto& r : l->records) {
                if (!r.tombstone) all_records.push_back(r);
            }
        }

        leaf_map_.clear();
        std::sort(all_records.begin(), all_records.end());
        total_records_ = all_records.size();

        std::vector<CompositeKey> keys;
        keys.reserve(all_records.size());
        for (auto& r : all_records) keys.push_back(r.key);
        thresholds_ = BetaComputer::compute_dynamic_thresholds(
            keys, config_.beta_strict);

        root_ = build_recursive(all_records, 0);
        link_leaves();
        rebuild_leaf_map();
    }

    void recalculate_thresholds() {
        std::unique_lock<std::shared_mutex> lock(tree_latch_);
        recompute_thresholds();
        if (config_.enable_wal)
            wal_->append(WalOpType::BETA_RECALC, INVALID_TXN,
                        INVALID_PAGE, INVALID_NODE, 0);
    }

    void checkpoint() {
        flush_delta();
        compact();
        std::set<TxnId> active_txns;
        std::vector<PageId> dirty;
        if (buffer_pool_) dirty = buffer_pool_->dirty_pages();
        if (config_.enable_wal)
            wal_->checkpoint(active_txns, dirty, thresholds_);
        if (buffer_pool_) buffer_pool_->flush_all();
    }

    // ======================================================================
    //  STATISTICS & COST MODEL
    // ======================================================================
    TreeStatistics statistics() const {
        std::shared_lock<std::shared_mutex> lock(tree_latch_);
        auto stats = stats_collector_.collect(root_.get(), config_.max_leaf_size);
        stats.current_thresholds = thresholds_;
        stats.delta_buffer_size = delta_buffer_.size();
        stats.delta_total_flushes = delta_flushes_.load();
        stats.total_splits = total_splits_.load();
        stats.total_merges = total_merges_.load();
        stats.total_rebalances = total_rebalances_.load();
        return stats;
    }

    QueryCost estimate_query_cost(const PredicateSet& preds) const {
        std::shared_lock<std::shared_mutex> lock(tree_latch_);
        auto stats = stats_collector_.collect(root_.get(), config_.max_leaf_size);
        return stats_collector_.estimate_cost(root_.get(), preds, stats);
    }

    // ======================================================================
    //  ACCESSORS
    // ======================================================================
    uint64_t size() const { return total_records_.load(); }
    bool     empty() const { return total_records_.load() == 0; }

    const HPTreeConfig&     config() const { return config_; }
    const CompositeKeySchema& schema() const { return schema_; }
    const BetaComputer::Thresholds& thresholds() const { return thresholds_; }

    HPNode* root() const { return root_.get(); }

    void set_beta_strategy(BetaStrategy strategy) {
        config_.beta_strategy = strategy;
    }

    void set_schema(const CompositeKeySchema& schema) {
        schema_ = schema;
        stats_collector_.set_schema(&schema_);
    }

    // ======================================================================
    //  DEBUG / PRINT
    // ======================================================================
    void print_tree(std::ostream& os, size_t max_depth = 5) const {
        std::shared_lock<std::shared_mutex> lock(tree_latch_);
        os << "=== HP-Tree ===\n";
        os << "Records: " << total_records_.load()
           << ", Strategy: " << static_cast<int>(config_.beta_strategy)
           << ", BF: " << config_.branching_factor
           << ", MaxLeaf: " << config_.max_leaf_size << "\n";
        os << "Thresholds: AM=" << thresholds_.am
           << " MED=" << thresholds_.median
           << " 2STD=" << thresholds_.stddev_2x
           << " 6STD=" << thresholds_.stddev_6x << "\n";
        if (root_) print_node(os, root_.get(), 0, max_depth);
    }

    void print_stats(std::ostream& os) const {
        auto s = statistics();
        os << "--- HP-Tree Statistics ---\n"
           << "Total Records:     " << s.total_records << "\n"
           << "Total Leaves:      " << s.total_leaves << "\n"
           << "Total Internal:    " << s.total_internal << "\n"
           << "Homogeneous Leaves:" << s.total_homogeneous << "\n"
           << "Tree Depth:        " << s.tree_depth << "\n"
           << "Memory (bytes):    " << s.memory_bytes << "\n"
           << "Avg Leaf Fill:     " << s.avg_leaf_fill << "\n"
           << "Avg Beta:          " << s.avg_beta << "\n"
           << "Beta p50:          " << s.beta_dist.p50 << "\n"
           << "Beta p99:          " << s.beta_dist.p99 << "\n"
           << "Total Splits:      " << s.total_splits << "\n"
           << "Total Merges:      " << s.total_merges << "\n"
           << "Delta Flushes:     " << s.delta_total_flushes << "\n";
    }

private:
    void flush_if_needed_locked() {
        if (config_.enable_delta_buffer &&
            delta_buffer_.needs_flush(total_records_.load())) {
            flush_delta_buffer_locked();
        }
    }

    void flush_delta_if_needed() {
        if (config_.enable_delta_buffer &&
            delta_buffer_.needs_flush(total_records_.load())) {
            if (config_.single_threaded) {
                flush_delta_buffer_locked();
                return;
            }
            std::unique_lock<std::shared_mutex> lock(tree_latch_);
            flush_delta_buffer_locked();
        }
    }

    bool update_in_place(HPNode* node, CompositeKey key,
                         const Record& new_rec, TxnId txn) {
        if (!node) return false;
        if (node->is_leaf()) {
            auto* leaf = static_cast<LeafNode*>(node);
            leaf->latch.lock();
            for (auto& r : leaf->records) {
                if (r.key == key && !r.tombstone && r.version.is_visible(txn)) {
                    if (config_.enable_wal) {
                        wal_->log_update(txn, INVALID_PAGE, leaf->meta.id,
                                        key, r.payload, new_rec.payload);
                    }
                    r.payload = new_rec.payload;
                    r.version.xmin = txn;
                    leaf->latch.unlock();
                    return true;
                }
            }
            leaf->latch.unlock();
            return false;
        }

        auto* internal = static_cast<InternalNode*>(node);
        size_t idx = internal->find_child_index(key);
        if (idx >= internal->children.size()) return false;
        return update_in_place(internal->children[idx].get(), key, new_rec, txn);
    }

    LeafNode* find_leaf_for_key(HPNode* node, CompositeKey key) const {
        if (!node) return nullptr;
        if (node->is_leaf()) return static_cast<LeafNode*>(node);
        auto* internal = static_cast<InternalNode*>(node);
        size_t idx = internal->find_child_index(key);
        if (idx >= internal->children.size()) {
            if (internal->children.empty()) return nullptr;
            idx = internal->children.size() - 1;
        }
        return find_leaf_for_key(internal->children[idx].get(), key);
    }

    void print_node(std::ostream& os, HPNode* node,
                    size_t indent, size_t max_depth) const {
        if (!node || indent > max_depth) return;
        std::string pad(indent * 2, ' ');

        if (node->is_leaf()) {
            auto* leaf = static_cast<LeafNode*>(node);
            os << pad << (leaf->is_homogeneous() ? "[HomoLeaf" : "[Leaf")
               << " id=" << leaf->meta.id
               << " n=" << leaf->records.size()
               << " beta=" << leaf->meta.beta_value
               << " range=[" << static_cast<uint64_t>(leaf->meta.range.low)
               << ".." << static_cast<uint64_t>(leaf->meta.range.high) << "]"
               << " prev=" << (leaf->prev_leaf == INVALID_NODE ? -1LL :
                               static_cast<long long>(leaf->prev_leaf))
               << " next=" << (leaf->next_leaf == INVALID_NODE ? -1LL :
                               static_cast<long long>(leaf->next_leaf))
               << "]\n";
        } else {
            auto* internal = static_cast<InternalNode*>(node);
            os << pad << "[Internal id=" << internal->meta.id
               << " keys=" << internal->separator_keys.size()
               << " children=" << internal->children.size()
               << " records=" << internal->meta.record_count
               << "]\n";
            for (size_t i = 0; i < internal->children.size() &&
                               i < 8; ++i) {
                print_node(os, internal->children[i].get(),
                           indent + 1, max_depth);
            }
            if (internal->children.size() > 8)
                os << pad << "  ... (" << internal->children.size() - 8
                   << " more children)\n";
        }
    }
};

}  // namespace hptree
