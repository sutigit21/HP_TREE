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
    std::atomic<size_t>   size_hint_{0};
    mutable std::mutex    mtx_;
    size_t total_inserts_ = 0;
    size_t total_deletes_ = 0;
    size_t total_updates_ = 0;

public:
    explicit DeltaBuffer(size_t cap = DEFAULT_DELTA_BUFFER_CAP) : capacity_(cap) {
        entries_.reserve(cap);
    }

    bool is_full() const {
        return size_hint_.load(std::memory_order_relaxed) >= capacity_;
    }

    bool is_empty() const {
        return size_hint_.load(std::memory_order_relaxed) == 0;
    }

    size_t size() const {
        return size_hint_.load(std::memory_order_relaxed);
    }

    size_t size_hint() const {
        return size_hint_.load(std::memory_order_relaxed);
    }

    double fill_ratio() const {
        size_t s = size_hint_.load(std::memory_order_relaxed);
        return capacity_ > 0 ? static_cast<double>(s) / capacity_ : 0.0;
    }

    bool needs_flush(size_t tree_size) const {
        size_t s = size_hint_.load(std::memory_order_relaxed);
        if (s >= capacity_) return true;
        if (tree_size > 0 &&
            static_cast<double>(s) / tree_size > DELTA_MERGE_RATIO)
            return true;
        return false;
    }

    void add_insert(const Record& rec, TxnId txn) {
        std::lock_guard<std::mutex> lock(mtx_);
        DeltaEntry e;
        e.op = DeltaOpType::INSERT;
        e.record = rec;
        e.txn_id = txn;
        e.seq = seq_counter_.fetch_add(1, std::memory_order_relaxed);
        entries_.push_back(std::move(e));
        total_inserts_++;
        size_hint_.store(entries_.size(), std::memory_order_relaxed);
    }

    void add_insert(Record&& rec, TxnId txn) {
        std::lock_guard<std::mutex> lock(mtx_);
        DeltaEntry e;
        e.op = DeltaOpType::INSERT;
        e.record = std::move(rec);
        e.txn_id = txn;
        e.seq = seq_counter_.fetch_add(1, std::memory_order_relaxed);
        entries_.push_back(std::move(e));
        total_inserts_++;
        size_hint_.store(entries_.size(), std::memory_order_relaxed);
    }

    void add_delete(CompositeKey key, TxnId txn) {
        std::lock_guard<std::mutex> lock(mtx_);
        DeltaEntry e;
        e.op = DeltaOpType::DELETE;
        e.record.key = key;
        e.record.tombstone = true;
        e.txn_id = txn;
        e.seq = seq_counter_.fetch_add(1, std::memory_order_relaxed);
        entries_.push_back(std::move(e));
        total_deletes_++;
        size_hint_.store(entries_.size(), std::memory_order_relaxed);
    }

    void add_update(const Record& old_rec, const Record& new_rec, TxnId txn) {
        std::lock_guard<std::mutex> lock(mtx_);
        DeltaEntry e;
        e.op = DeltaOpType::UPDATE;
        e.record = new_rec;
        e.old_record = old_rec;
        e.txn_id = txn;
        e.seq = seq_counter_.fetch_add(1, std::memory_order_relaxed);
        entries_.push_back(std::move(e));
        total_updates_++;
        size_hint_.store(entries_.size(), std::memory_order_relaxed);
    }

    std::vector<Record*> search(CompositeKey key, TxnId reader_txn) {
        if (size_hint_.load(std::memory_order_relaxed) == 0) return {};
        std::lock_guard<std::mutex> lock(mtx_);
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
        if (size_hint_.load(std::memory_order_relaxed) == 0) return {};
        std::lock_guard<std::mutex> lock(mtx_);
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
        if (size_hint_.load(std::memory_order_relaxed) == 0) return {};
        std::lock_guard<std::mutex> lock(mtx_);
        std::set<CompositeKey> keys;
        for (auto& e : entries_) {
            if (e.op == DeltaOpType::DELETE) keys.insert(e.record.key);
        }
        return keys;
    }

    std::vector<DeltaEntry> drain() {
        std::lock_guard<std::mutex> lock(mtx_);
        auto drained = std::move(entries_);
        entries_.clear();
        entries_.reserve(capacity_);
        std::sort(drained.begin(), drained.end());
        size_hint_.store(0, std::memory_order_relaxed);
        return drained;
    }

    std::vector<DeltaEntry> snapshot() const {
        std::lock_guard<std::mutex> lock(mtx_);
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
        std::lock_guard<std::mutex> lock(mtx_);
        return {
            entries_.size(), capacity_,
            total_inserts_, total_deletes_, total_updates_,
            capacity_ > 0 ?
                static_cast<double>(entries_.size()) / capacity_ : 0.0
        };
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mtx_);
        entries_.clear();
        size_hint_.store(0, std::memory_order_relaxed);
    }

    void set_capacity(size_t cap) {
        std::lock_guard<std::mutex> lock(mtx_);
        capacity_ = cap;
    }
};

}  // namespace hptree
