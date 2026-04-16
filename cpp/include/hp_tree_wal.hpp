#pragma once

#include "hp_tree_common.hpp"
#include <fstream>
#include <deque>
#include <set>

namespace hptree {

struct WalRecord {
    LSN       lsn          = INVALID_LSN;
    TxnId     txn_id       = INVALID_TXN;
    WalOpType op           = WalOpType::INSERT;
    PageId    page_id      = INVALID_PAGE;
    NodeId    node_id      = INVALID_NODE;
    CompositeKey key       = 0;
    std::vector<uint8_t> before_image;
    std::vector<uint8_t> after_image;
    LSN       prev_lsn     = INVALID_LSN;
    uint64_t  timestamp    = 0;

    size_t serialized_size() const {
        return sizeof(lsn) + sizeof(txn_id) + sizeof(op) + sizeof(page_id)
             + sizeof(node_id) + sizeof(key) + sizeof(uint32_t)
             + before_image.size() + sizeof(uint32_t) + after_image.size()
             + sizeof(prev_lsn) + sizeof(timestamp);
    }

    void serialize(std::vector<uint8_t>& buf) const {
        auto append = [&](const void* p, size_t n) {
            const uint8_t* b = static_cast<const uint8_t*>(p);
            buf.insert(buf.end(), b, b + n);
        };
        append(&lsn, sizeof(lsn));
        append(&txn_id, sizeof(txn_id));
        append(&op, sizeof(op));
        append(&page_id, sizeof(page_id));
        append(&node_id, sizeof(node_id));
        append(&key, sizeof(key));
        uint32_t bs = static_cast<uint32_t>(before_image.size());
        append(&bs, sizeof(bs));
        if (bs > 0) append(before_image.data(), bs);
        uint32_t as = static_cast<uint32_t>(after_image.size());
        append(&as, sizeof(as));
        if (as > 0) append(after_image.data(), as);
        append(&prev_lsn, sizeof(prev_lsn));
        append(&timestamp, sizeof(timestamp));
    }

    static WalRecord deserialize(const uint8_t* data, size_t& offset) {
        WalRecord r;
        auto read = [&](void* p, size_t n) {
            std::memcpy(p, data + offset, n);
            offset += n;
        };
        read(&r.lsn, sizeof(r.lsn));
        read(&r.txn_id, sizeof(r.txn_id));
        read(&r.op, sizeof(r.op));
        read(&r.page_id, sizeof(r.page_id));
        read(&r.node_id, sizeof(r.node_id));
        read(&r.key, sizeof(r.key));
        uint32_t bs = 0;
        read(&bs, sizeof(bs));
        if (bs > 0) {
            r.before_image.resize(bs);
            read(r.before_image.data(), bs);
        }
        uint32_t as = 0;
        read(&as, sizeof(as));
        if (as > 0) {
            r.after_image.resize(as);
            read(r.after_image.data(), as);
        }
        read(&r.prev_lsn, sizeof(r.prev_lsn));
        read(&r.timestamp, sizeof(r.timestamp));
        return r;
    }
};

struct CheckpointInfo {
    LSN                checkpoint_lsn = INVALID_LSN;
    std::set<TxnId>    active_txns;
    std::vector<PageId> dirty_pages;
    double             beta_am        = 0.0;
    double             beta_median    = 0.0;
    double             beta_stddev_2x = 0.0;
    double             beta_stddev_6x = 0.0;
};

class WalManager {
    std::string   filepath_;
    std::ofstream log_file_;
    std::mutex    mtx_;
    std::atomic<LSN> next_lsn_{1};
    std::deque<WalRecord> buffer_;
    size_t        buffer_limit_ = 1024;
    bool          enabled_     = true;
    CheckpointInfo last_checkpoint_;

    std::unordered_map<TxnId, LSN> txn_last_lsn_;

    uint64_t now_micros() const {
        auto tp = std::chrono::steady_clock::now();
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                tp.time_since_epoch()).count());
    }

public:
    WalManager() = default;

    explicit WalManager(const std::string& path, bool enabled = true)
        : filepath_(path), enabled_(enabled) {
        if (enabled_) {
            log_file_.open(filepath_, std::ios::binary | std::ios::app);
        }
    }

    ~WalManager() {
        if (log_file_.is_open()) {
            flush();
            log_file_.close();
        }
    }

    WalManager(WalManager&&) = delete;
    WalManager& operator=(WalManager&&) = delete;
    WalManager(const WalManager&) = delete;
    WalManager& operator=(const WalManager&) = delete;

    bool is_enabled() const { return enabled_; }

    LSN append(WalOpType op, TxnId txn, PageId page, NodeId node,
               CompositeKey key,
               const std::vector<uint8_t>& before = {},
               const std::vector<uint8_t>& after = {}) {
        if (!enabled_) return INVALID_LSN;

        std::lock_guard<std::mutex> lock(mtx_);
        WalRecord rec;
        rec.lsn = next_lsn_.fetch_add(1);
        rec.txn_id = txn;
        rec.op = op;
        rec.page_id = page;
        rec.node_id = node;
        rec.key = key;
        rec.before_image = before;
        rec.after_image = after;
        rec.timestamp = now_micros();

        auto it = txn_last_lsn_.find(txn);
        rec.prev_lsn = (it != txn_last_lsn_.end()) ? it->second : INVALID_LSN;
        txn_last_lsn_[txn] = rec.lsn;

        buffer_.push_back(std::move(rec));
        if (buffer_.size() >= buffer_limit_) flush_locked();

        return rec.lsn;
    }

    LSN log_insert(TxnId txn, PageId page, NodeId node, CompositeKey key,
                   const std::vector<uint8_t>& record_data) {
        return append(WalOpType::INSERT, txn, page, node, key, {}, record_data);
    }

    LSN log_delete(TxnId txn, PageId page, NodeId node, CompositeKey key,
                   const std::vector<uint8_t>& record_data) {
        return append(WalOpType::DELETE, txn, page, node, key, record_data, {});
    }

    LSN log_update(TxnId txn, PageId page, NodeId node, CompositeKey key,
                   const std::vector<uint8_t>& before,
                   const std::vector<uint8_t>& after) {
        return append(WalOpType::UPDATE, txn, page, node, key, before, after);
    }

    LSN log_split(TxnId txn, NodeId parent, NodeId child_new) {
        return append(WalOpType::SPLIT, txn, INVALID_PAGE, parent, child_new);
    }

    LSN log_merge(TxnId txn, NodeId survivor, NodeId removed) {
        return append(WalOpType::MERGE, txn, INVALID_PAGE, survivor, removed);
    }

    LSN log_txn_begin(TxnId txn) {
        return append(WalOpType::TXN_BEGIN, txn, INVALID_PAGE, INVALID_NODE, 0);
    }

    LSN log_txn_commit(TxnId txn) {
        LSN lsn = append(WalOpType::TXN_COMMIT, txn, INVALID_PAGE, INVALID_NODE, 0);
        std::lock_guard<std::mutex> lock(mtx_);
        txn_last_lsn_.erase(txn);
        return lsn;
    }

    LSN log_txn_abort(TxnId txn) {
        LSN lsn = append(WalOpType::TXN_ABORT, txn, INVALID_PAGE, INVALID_NODE, 0);
        std::lock_guard<std::mutex> lock(mtx_);
        txn_last_lsn_.erase(txn);
        return lsn;
    }

    void checkpoint(const std::set<TxnId>& active_txns,
                    const std::vector<PageId>& dirty_pages,
                    const BetaComputer::Thresholds& thresholds) {
        std::lock_guard<std::mutex> lock(mtx_);
        flush_locked();

        WalRecord cp_rec;
        cp_rec.lsn = next_lsn_.fetch_add(1);
        cp_rec.txn_id = INVALID_TXN;
        cp_rec.op = WalOpType::CHECKPOINT;
        cp_rec.timestamp = now_micros();

        std::vector<uint8_t> cp_data;
        uint32_t num_txns = static_cast<uint32_t>(active_txns.size());
        cp_data.insert(cp_data.end(),
            reinterpret_cast<const uint8_t*>(&num_txns),
            reinterpret_cast<const uint8_t*>(&num_txns) + sizeof(num_txns));
        for (auto t : active_txns) {
            cp_data.insert(cp_data.end(),
                reinterpret_cast<const uint8_t*>(&t),
                reinterpret_cast<const uint8_t*>(&t) + sizeof(t));
        }
        cp_rec.after_image = std::move(cp_data);

        buffer_.push_back(std::move(cp_rec));
        flush_locked();

        last_checkpoint_.checkpoint_lsn = cp_rec.lsn;
        last_checkpoint_.active_txns = active_txns;
        last_checkpoint_.dirty_pages = dirty_pages;
        last_checkpoint_.beta_am = thresholds.am;
        last_checkpoint_.beta_median = thresholds.median;
        last_checkpoint_.beta_stddev_2x = thresholds.stddev_2x;
        last_checkpoint_.beta_stddev_6x = thresholds.stddev_6x;
    }

    void flush() {
        std::lock_guard<std::mutex> lock(mtx_);
        flush_locked();
    }

    LSN current_lsn() const { return next_lsn_.load(); }

    const CheckpointInfo& last_checkpoint_info() const { return last_checkpoint_; }

    struct RecoveryResult {
        std::vector<WalRecord> redo_records;
        std::set<TxnId>        committed_txns;
        std::set<TxnId>        aborted_txns;
        std::set<TxnId>        active_txns;
        LSN                    max_lsn = INVALID_LSN;
    };

    RecoveryResult recover(const std::string& path) {
        RecoveryResult result;
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) return result;

        std::vector<uint8_t> file_data(
            (std::istreambuf_iterator<char>(in)),
            std::istreambuf_iterator<char>());
        in.close();

        size_t offset = 0;
        std::set<TxnId> begun;
        while (offset + sizeof(uint32_t) <= file_data.size()) {
            try {
                uint32_t rec_len = 0;
                std::memcpy(&rec_len, file_data.data() + offset, sizeof(rec_len));
                offset += sizeof(rec_len);
                if (offset + rec_len > file_data.size()) break;
                auto rec = WalRecord::deserialize(file_data.data(), offset);
                if (rec.lsn > result.max_lsn) result.max_lsn = rec.lsn;

                switch (rec.op) {
                case WalOpType::TXN_BEGIN:
                    begun.insert(rec.txn_id);
                    break;
                case WalOpType::TXN_COMMIT:
                    result.committed_txns.insert(rec.txn_id);
                    begun.erase(rec.txn_id);
                    break;
                case WalOpType::TXN_ABORT:
                    result.aborted_txns.insert(rec.txn_id);
                    begun.erase(rec.txn_id);
                    break;
                default:
                    break;
                }
                result.redo_records.push_back(std::move(rec));
            } catch (...) {
                break;
            }
        }

        result.active_txns = begun;
        return result;
    }

private:
    void flush_locked() {
        if (!log_file_.is_open() || buffer_.empty()) return;
        for (auto& rec : buffer_) {
            std::vector<uint8_t> buf;
            rec.serialize(buf);
            uint32_t rec_len = static_cast<uint32_t>(buf.size());
            log_file_.write(reinterpret_cast<const char*>(&rec_len), sizeof(rec_len));
            log_file_.write(reinterpret_cast<const char*>(buf.data()), buf.size());
        }
        log_file_.flush();
        buffer_.clear();
    }
};

}  // namespace hptree
