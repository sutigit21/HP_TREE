#pragma once

#include "hp_tree_node.hpp"

namespace hptree {

class HPTreeIterator {
public:
    enum class State { VALID, END, INVALID };

private:
    LeafNode*      current_leaf_ = nullptr;
    size_t         record_idx_   = 0;
    KeyRange       range_;
    TxnId          reader_txn_   = TXN_COMMITTED;
    ScanDirection  direction_    = ScanDirection::FORWARD;
    State          state_        = State::INVALID;
    const PredicateSet* predicates_ = nullptr;
    const CompositeKeySchema* schema_ = nullptr;
    bool           clean_        = false;

    std::unordered_map<NodeId, LeafNode*>* leaf_map_ = nullptr;

public:
    HPTreeIterator() = default;

    HPTreeIterator(LeafNode* start_leaf, size_t start_idx,
                   const KeyRange& range, TxnId txn,
                   ScanDirection dir = ScanDirection::FORWARD,
                   const PredicateSet* preds = nullptr,
                   const CompositeKeySchema* schema = nullptr,
                   std::unordered_map<NodeId, LeafNode*>* lmap = nullptr,
                   bool clean = false)
        : current_leaf_(start_leaf), record_idx_(start_idx),
          range_(range), reader_txn_(txn), direction_(dir),
          predicates_(preds), schema_(schema), clean_(clean), leaf_map_(lmap) {
        if (current_leaf_ && !current_leaf_->records.empty()) {
            state_ = State::VALID;
            advance_to_valid();
        } else {
            state_ = State::END;
        }
    }

    bool valid() const { return state_ == State::VALID; }
    bool at_end() const { return state_ == State::END; }

    Record* current() {
        if (!valid()) return nullptr;
        return &current_leaf_->records[record_idx_];
    }

    const Record* current() const {
        if (!valid()) return nullptr;
        return &current_leaf_->records[record_idx_];
    }

    CompositeKey key() const {
        if (!valid()) return 0;
        return current_leaf_->records[record_idx_].key;
    }

    void next() {
        if (!valid()) return;
        if (direction_ == ScanDirection::FORWARD)
            advance_forward();
        else
            advance_reverse();
        advance_to_valid();
    }

    void seek(CompositeKey target) {
        if (!current_leaf_) { state_ = State::END; return; }

        if (direction_ == ScanDirection::FORWARD) {
            while (current_leaf_) {
                auto it = std::lower_bound(
                    current_leaf_->records.begin(),
                    current_leaf_->records.end(),
                    Record{target},
                    [](const Record& a, const Record& b) {
                        return a.key < b.key;
                    });
                if (it != current_leaf_->records.end()) {
                    record_idx_ = it - current_leaf_->records.begin();
                    state_ = State::VALID;
                    advance_to_valid();
                    return;
                }
                move_to_next_leaf();
            }
        } else {
            while (current_leaf_) {
                auto it = std::upper_bound(
                    current_leaf_->records.begin(),
                    current_leaf_->records.end(),
                    Record{target},
                    [](const Record& a, const Record& b) {
                        return a.key < b.key;
                    });
                if (it != current_leaf_->records.begin()) {
                    --it;
                    record_idx_ = it - current_leaf_->records.begin();
                    state_ = State::VALID;
                    advance_to_valid();
                    return;
                }
                move_to_prev_leaf();
            }
        }
        state_ = State::END;
    }

    size_t count_remaining() {
        size_t count = 0;
        while (valid()) {
            count++;
            next();
        }
        return count;
    }

    std::vector<Record*> collect(size_t limit = std::numeric_limits<size_t>::max()) {
        std::vector<Record*> result;
        while (valid() && result.size() < limit) {
            result.push_back(current());
            next();
        }
        return result;
    }

    double sum_dimension(size_t dim_idx, const CompositeKeySchema& schema,
                         size_t limit = std::numeric_limits<size_t>::max()) {
        double total = 0.0;
        size_t n = 0;
        CompositeKeyEncoder encoder(schema);
        while (valid() && n < limit) {
            uint64_t val = encoder.extract_dim(key(), dim_idx);
            total += static_cast<double>(val);
            next();
            n++;
        }
        return total;
    }

private:
    void advance_forward() {
        record_idx_++;
        if (record_idx_ >= current_leaf_->records.size())
            move_to_next_leaf();
    }

    void advance_reverse() {
        if (record_idx_ == 0)
            move_to_prev_leaf();
        else
            record_idx_--;
    }

    void move_to_next_leaf() {
        if (!current_leaf_) { state_ = State::END; return; }
        LeafNode* nxt = current_leaf_->next_leaf_ptr;
        if (!nxt) {
            NodeId next_id = current_leaf_->next_leaf;
            if (next_id == INVALID_NODE || !leaf_map_) {
                state_ = State::END;
                current_leaf_ = nullptr;
                return;
            }
            auto it = leaf_map_->find(next_id);
            if (it == leaf_map_->end()) {
                state_ = State::END;
                current_leaf_ = nullptr;
                return;
            }
            nxt = it->second;
        }
        current_leaf_ = nxt;
        record_idx_ = 0;
        if (current_leaf_->records.empty()) {
            move_to_next_leaf();
        }
    }

    void move_to_prev_leaf() {
        if (!current_leaf_) { state_ = State::END; return; }
        LeafNode* prv = current_leaf_->prev_leaf_ptr;
        if (!prv) {
            NodeId prev_id = current_leaf_->prev_leaf;
            if (prev_id == INVALID_NODE || !leaf_map_) {
                state_ = State::END;
                current_leaf_ = nullptr;
                return;
            }
            auto it = leaf_map_->find(prev_id);
            if (it == leaf_map_->end()) {
                state_ = State::END;
                current_leaf_ = nullptr;
                return;
            }
            prv = it->second;
        }
        current_leaf_ = prv;
        record_idx_ = current_leaf_->records.empty()
                    ? 0 : current_leaf_->records.size() - 1;
        if (current_leaf_->records.empty()) {
            move_to_prev_leaf();
        }
    }

    void advance_to_valid() {
        while (state_ == State::VALID && current_leaf_) {
            if (record_idx_ >= current_leaf_->records.size()) {
                if (direction_ == ScanDirection::FORWARD)
                    move_to_next_leaf();
                else
                    move_to_prev_leaf();
                continue;
            }

            auto& rec = current_leaf_->records[record_idx_];

            if (direction_ == ScanDirection::FORWARD && rec.key > range_.high) {
                state_ = State::END; return;
            }
            if (direction_ == ScanDirection::REVERSE && rec.key < range_.low) {
                state_ = State::END; return;
            }

            if (rec.key < range_.low || rec.key > range_.high) {
                if (direction_ == ScanDirection::FORWARD) advance_forward();
                else advance_reverse();
                continue;
            }

            if (!clean_ && (rec.tombstone || !rec.version.is_visible(reader_txn_))) {
                if (direction_ == ScanDirection::FORWARD) advance_forward();
                else advance_reverse();
                continue;
            }

            if (predicates_ && schema_) {
                if (!predicates_->evaluate_record(
                        rec.key, *schema_, &current_leaf_->null_bitmap, record_idx_)) {
                    if (direction_ == ScanDirection::FORWARD) advance_forward();
                    else advance_reverse();
                    continue;
                }
            }

            return;
        }
        if (!current_leaf_) state_ = State::END;
    }
};

}  // namespace hptree
