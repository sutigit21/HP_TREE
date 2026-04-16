#pragma once

#include "hp_tree_common.hpp"
#include <map>
#include <set>

namespace hptree {

enum class DeltaOpType : uint8_t {
    INSERT = 0,
    DELETE = 1,
    UPDATE = 2,
};

struct DeltaEntry {
    DeltaOpType op;
    Record      record;
    Record      old_record;
    TxnId       txn_id = INVALID_TXN;
    uint64_t    seq    = 0;

    bool operator<(const DeltaEntry& o) const {
        if (record.key != o.record.key) return record.key < o.record.key;
        return seq < o.seq;
    }
};

class DeltaBuffer {
    std::vector<DeltaEntry> entries_;
    size_t capacity_;
    std::atomic<uint64_t> seq_counter_{0};
    mutable std::shared_mutex mtx_;
    size_t total_inserts_ = 0;
    size_t total_deletes_ = 0;
    size_t total_updates_ = 0;

public:
    explicit DeltaBuffer(size_t cap = DEFAULT_DELTA_BUFFER_CAP) : capacity_(cap) {
        entries_.reserve(cap);
    }

    bool is_full() const {
        std::shared_lock<std::shared_mutex> lock(mtx_);
        return entries_.size() >= capacity_;
    }

    bool is_empty() const {
        std::shared_lock<std::shared_mutex> lock(mtx_);
        return entries_.empty();
    }

    size_t size() const {
        std::shared_lock<std::shared_mutex> lock(mtx_);
        return entries_.size();
    }

    double fill_ratio() const {
        std::shared_lock<std::shared_mutex> lock(mtx_);
        return capacity_ > 0 ?
            static_cast<double>(entries_.size()) / capacity_ : 0.0;
    }

    bool needs_flush(size_t tree_size) const {
        std::shared_lock<std::shared_mutex> lock(mtx_);
        if (entries_.size() >= capacity_) return true;
        if (tree_size > 0 &&
            static_cast<double>(entries_.size()) / tree_size > DELTA_MERGE_RATIO)
            return true;
        return false;
    }

    void add_insert(const Record& rec, TxnId txn) {
        std::unique_lock<std::shared_mutex> lock(mtx_);
        DeltaEntry e;
        e.op = DeltaOpType::INSERT;
        e.record = rec;
        e.txn_id = txn;
        e.seq = seq_counter_.fetch_add(1);
        entries_.push_back(std::move(e));
        total_inserts_++;
    }

    void add_delete(CompositeKey key, TxnId txn) {
        std::unique_lock<std::shared_mutex> lock(mtx_);
        DeltaEntry e;
        e.op = DeltaOpType::DELETE;
        e.record.key = key;
        e.record.tombstone = true;
        e.txn_id = txn;
        e.seq = seq_counter_.fetch_add(1);
        entries_.push_back(std::move(e));
        total_deletes_++;
    }

    void add_update(const Record& old_rec, const Record& new_rec, TxnId txn) {
        std::unique_lock<std::shared_mutex> lock(mtx_);
        DeltaEntry e;
        e.op = DeltaOpType::UPDATE;
        e.record = new_rec;
        e.old_record = old_rec;
        e.txn_id = txn;
        e.seq = seq_counter_.fetch_add(1);
        entries_.push_back(std::move(e));
        total_updates_++;
    }

    std::vector<Record*> search(CompositeKey key, TxnId reader_txn) {
        std::shared_lock<std::shared_mutex> lock(mtx_);
        std::vector<Record*> result;
        for (auto& e : entries_) {
            if (e.record.key == key && !e.record.tombstone
                && e.op != DeltaOpType::DELETE
                && e.record.version.is_visible(reader_txn)) {
                result.push_back(&e.record);
            }
        }
        return result;
    }

    std::vector<Record*> range_search(const KeyRange& kr, TxnId reader_txn) {
        std::shared_lock<std::shared_mutex> lock(mtx_);
        std::vector<Record*> result;
        for (auto& e : entries_) {
            if (e.record.key >= kr.low && e.record.key <= kr.high
                && !e.record.tombstone
                && e.op != DeltaOpType::DELETE
                && e.record.version.is_visible(reader_txn)) {
                result.push_back(&e.record);
            }
        }
        return result;
    }

    std::set<CompositeKey> deleted_keys() const {
        std::shared_lock<std::shared_mutex> lock(mtx_);
        std::set<CompositeKey> keys;
        for (auto& e : entries_) {
            if (e.op == DeltaOpType::DELETE) keys.insert(e.record.key);
        }
        return keys;
    }

    std::vector<DeltaEntry> drain() {
        std::unique_lock<std::shared_mutex> lock(mtx_);
        auto drained = std::move(entries_);
        entries_.clear();
        entries_.reserve(capacity_);
        std::sort(drained.begin(), drained.end());
        return drained;
    }

    std::vector<DeltaEntry> snapshot() const {
        std::shared_lock<std::shared_mutex> lock(mtx_);
        auto copy = entries_;
        std::sort(copy.begin(), copy.end());
        return copy;
    }

    struct Stats {
        size_t current_size;
        size_t capacity;
        size_t total_inserts;
        size_t total_deletes;
        size_t total_updates;
        double fill_ratio;
    };

    Stats stats() const {
        std::shared_lock<std::shared_mutex> lock(mtx_);
        return {
            entries_.size(), capacity_,
            total_inserts_, total_deletes_, total_updates_,
            capacity_ > 0 ?
                static_cast<double>(entries_.size()) / capacity_ : 0.0
        };
    }

    void clear() {
        std::unique_lock<std::shared_mutex> lock(mtx_);
        entries_.clear();
    }

    void set_capacity(size_t cap) {
        std::unique_lock<std::shared_mutex> lock(mtx_);
        capacity_ = cap;
    }
};

}  // namespace hptree
