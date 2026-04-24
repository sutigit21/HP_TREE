#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <functional>
#include <optional>
#include <unordered_map>
#include <cassert>
#include <stdexcept>
#include <atomic>
#include <mutex>

namespace hptree {

using PageId   = uint64_t;
using TxnId    = uint64_t;
using LSN      = uint64_t;
using NodeId   = uint64_t;
using Epoch    = uint64_t;

static constexpr PageId  INVALID_PAGE  = std::numeric_limits<PageId>::max();
static constexpr TxnId   INVALID_TXN   = 0;
static constexpr TxnId   TXN_COMMITTED = std::numeric_limits<TxnId>::max();
static constexpr LSN     INVALID_LSN   = 0;
static constexpr NodeId  INVALID_NODE  = std::numeric_limits<NodeId>::max();

static constexpr size_t  DEFAULT_PAGE_SIZE       = 16384;
static constexpr size_t  DEFAULT_BRANCHING_FACTOR = 32;
static constexpr size_t  DEFAULT_MAX_LEAF_SIZE    = 32;
static constexpr size_t  DEFAULT_MIN_LEAF_SIZE    = 16;
static constexpr double  DEFAULT_BETA_STRICT      = 1e-9;
static constexpr double  MIN_POSITIVE_VALUE       = 1e-7;
static constexpr size_t  DEFAULT_BUFFER_POOL_SIZE = 1024;
static constexpr size_t  DEFAULT_DELTA_BUFFER_CAP = 4096;
static constexpr size_t  MAX_TREE_DEPTH           = 35;
// Max supported schema dimensions.  Lets us keep dim_stats as a stack-inline
// fixed array and avoid any per-node heap allocation for subtree aggregates.
static constexpr size_t  MAX_DIMS                 = 8;

enum class BetaStrategy : uint8_t {
    FIXED_STRICT    = 0,
    ARITHMETIC_MEAN = 1,
    MEDIAN          = 2,
    STDDEV_2X       = 3,
    STDDEV_6X       = 4,
    ADAPTIVE_LOCAL  = 5,
};

enum class PredicateOp : uint8_t {
    EQ, NEQ, LT, LTE, GT, GTE, BETWEEN, IN, IS_NULL, IS_NOT_NULL, LIKE,
};

enum class WorkloadProfile : uint8_t {
    ANALYTICAL   = 0,
    SCAN_HEAVY   = 1,
    WRITE_HEAVY  = 2,
    BALANCED     = 3,
    CUSTOM       = 4,
};

struct HPTreeConfig {
    size_t       branching_factor   = DEFAULT_BRANCHING_FACTOR;
    size_t       max_leaf_size      = DEFAULT_MAX_LEAF_SIZE;
    size_t       min_leaf_size      = DEFAULT_MIN_LEAF_SIZE;
    size_t       page_size          = DEFAULT_PAGE_SIZE;
    size_t       buffer_pool_pages  = DEFAULT_BUFFER_POOL_SIZE;
    size_t       delta_buffer_cap   = DEFAULT_DELTA_BUFFER_CAP;
    double       beta_strict        = DEFAULT_BETA_STRICT;
    BetaStrategy beta_strategy      = BetaStrategy::ARITHMETIC_MEAN;
    bool         enable_wal         = false;
    bool         enable_mvcc        = false;
    bool         enable_buffer_pool = false;
    bool         enable_delta_buffer= false;
    bool         enable_aggregates  = true;
    bool         single_threaded    = true;
    WorkloadProfile workload_profile  = WorkloadProfile::BALANCED;
    double       bulk_load_fill_factor = -1.0;
    std::string  wal_path           = "hp_tree.wal";
    std::string  data_path          = "hp_tree.dat";
};

struct DimensionDesc {
    std::string name;
    uint8_t     bits;
    enum Encoding { LINEAR, DICTIONARY } encoding = LINEAR;
    int64_t     base_value = 0;
    double      scale      = 1.0;
    std::unordered_map<std::string, uint64_t> dict_encode;
    std::unordered_map<uint64_t, std::string> dict_decode;
    bool        nullable   = true;
    uint64_t    null_sentinel = 0;

    void init_null_sentinel() { null_sentinel = (1ULL << bits) - 1; }

    uint64_t encode(int64_t val) const {
        int64_t shifted = val - base_value;
        if (shifted < 0) shifted = 0;
        uint64_t max_val = (1ULL << bits) - (nullable ? 2 : 1);
        return static_cast<uint64_t>(std::min(static_cast<uint64_t>(shifted), max_val));
    }
    uint64_t encode_float(double val) const {
        return encode(static_cast<int64_t>(val * scale));
    }
    uint64_t encode_string(const std::string& val) const {
        auto it = dict_encode.find(val);
        return it != dict_encode.end() ? it->second : 0;
    }
    int64_t  decode_int(uint64_t c)   const { return static_cast<int64_t>(c) + base_value; }
    double   decode_float(uint64_t c) const { return static_cast<double>(c) / scale; }
    bool     is_null_value(uint64_t c) const { return nullable && c == null_sentinel; }
};

struct CompositeKeySchema {
    std::vector<DimensionDesc> dimensions;
    size_t total_bits = 0;

    // Precomputed per-dim offsets/masks — filled by finalize().  Cached here
    // so the hot paths (PredicateSet::evaluate / to_key_range) don't pay an
    // O(dim_count) linear scan every time they extract a dimension value.
    std::array<uint8_t,  MAX_DIMS> cached_offsets{};
    std::array<uint64_t, MAX_DIMS> cached_masks{};

    void finalize() {
        total_bits = 0;
        for (auto& d : dimensions) {
            if (d.nullable) d.init_null_sentinel();
            total_bits += d.bits;
        }
        // Populate offset/mask caches — offset_of(i) is the bit offset of dim i
        // from the LSB, counting bits of dims [i+1 .. end).
        for (size_t i = 0; i < dimensions.size() && i < MAX_DIMS; ++i) {
            uint8_t off = 0;
            for (size_t j = i + 1; j < dimensions.size(); ++j)
                off += dimensions[j].bits;
            cached_offsets[i] = off;
            cached_masks[i]   = (1ULL << dimensions[i].bits) - 1;
        }
    }
    size_t  dim_count() const { return dimensions.size(); }
    uint8_t offset_of(size_t dim_idx) const { return cached_offsets[dim_idx]; }
    uint64_t mask_of(size_t dim_idx) const  { return cached_masks[dim_idx]; }
};

using CompositeKey = __uint128_t;

static constexpr CompositeKey COMPOSITE_KEY_MIN = 0;
static constexpr CompositeKey COMPOSITE_KEY_MAX = ~static_cast<CompositeKey>(0);

struct CompositeKeyEncoder {
    const CompositeKeySchema& schema;
    explicit CompositeKeyEncoder(const CompositeKeySchema& s) : schema(s) {}

    uint64_t extract_dim(CompositeKey key, size_t dim_idx) const {
        uint8_t off = schema.offset_of(dim_idx);
        uint64_t mask = schema.mask_of(dim_idx);
        return static_cast<uint64_t>((key >> off) & mask);
    }
};

struct BetaComputer {
    static double compute_beta(CompositeKey min_val, CompositeKey max_val) {
        if (min_val == 0 && max_val == 0) return 0.0;
        double mn = static_cast<double>(min_val);
        double mx = static_cast<double>(max_val);
        if (mn <= 0.0 || mx <= 0.0) return std::numeric_limits<double>::infinity();
        double diff = mx - mn;
        if (std::abs(diff) < MIN_POSITIVE_VALUE / 1e6) return 0.0;
        double num = diff * diff;
        double den = 4.0 * std::max(std::abs(mn), MIN_POSITIVE_VALUE)
                         * std::max(std::abs(mx), MIN_POSITIVE_VALUE);
        if (den < MIN_POSITIVE_VALUE / 1e7)
            return (num > MIN_POSITIVE_VALUE / 1e7)
                ? std::numeric_limits<double>::infinity() : 0.0;
        return num / den;
    }

    struct Thresholds { double am, median, stddev_2x, stddev_6x; };

    static Thresholds compute_dynamic_thresholds(
            const std::vector<CompositeKey>& keys, double strict_default) {
        Thresholds t{strict_default, strict_default, strict_default, strict_default};
        if (keys.empty()) return t;
        size_t n = keys.size();
        double sum = 0, sum_sq = 0;
        double mn = static_cast<double>(keys[0]);
        double mx = static_cast<double>(keys[0]);
        std::vector<double> dvals(n);
        for (size_t i = 0; i < n; ++i) {
            double v = static_cast<double>(keys[i]);
            dvals[i] = v; sum += v; sum_sq += v * v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        double mean_val = sum / n;
        double var_val  = (sum_sq / n) - (mean_val * mean_val);
        double std_val  = std::sqrt(std::max(var_val, 0.0));
        std::nth_element(dvals.begin(), dvals.begin() + n / 2, dvals.end());
        double median_val = dvals[n / 2];
        double b_total = compute_beta(
            static_cast<CompositeKey>(mn), static_cast<CompositeKey>(mx));
        t.am = (b_total != std::numeric_limits<double>::infinity()) ? b_total * 0.1 : strict_default;
        t.median    = (median_val > 1.0) ? median_val * 0.001 : strict_default;
        t.stddev_2x = (std_val > 1.0) ? std_val * 2.0 : strict_default;
        t.stddev_6x = (std_val > 1.0) ? std_val * 6.0 : strict_default;
        t.am        = std::max(t.am, MIN_POSITIVE_VALUE);
        t.median    = std::max(t.median, MIN_POSITIVE_VALUE);
        t.stddev_2x = std::max(t.stddev_2x, MIN_POSITIVE_VALUE);
        t.stddev_6x = std::max(t.stddev_6x, MIN_POSITIVE_VALUE);
        return t;
    }

    static double select_threshold(const Thresholds& t, BetaStrategy strategy,
                                   double strict_default) {
        switch (strategy) {
        case BetaStrategy::ARITHMETIC_MEAN: return t.am;
        case BetaStrategy::MEDIAN:          return t.median;
        case BetaStrategy::STDDEV_2X:       return t.stddev_2x;
        case BetaStrategy::STDDEV_6X:       return t.stddev_6x;
        case BetaStrategy::ADAPTIVE_LOCAL:  return 0.0;
        case BetaStrategy::FIXED_STRICT:
        default:                            return strict_default;
        }
    }
};

// Simplified Record: no payload vector, no MVCC, no tombstone.
// Key + single 8-byte value, matching the benchmark dataset exactly.
struct Record {
    CompositeKey key   = 0;
    uint64_t     value = 0;

    bool operator<(const Record& o)  const { return key < o.key; }
    bool operator==(const Record& o) const { return key == o.key; }
};

struct KeyRange {
    CompositeKey low  = COMPOSITE_KEY_MIN;
    CompositeKey high = COMPOSITE_KEY_MAX;

    bool contains(CompositeKey k) const { return k >= low && k <= high; }
    bool overlaps(const KeyRange& o) const { return low <= o.high && high >= o.low; }
    bool fully_contains(const KeyRange& o) const { return low <= o.low && high >= o.high; }
    bool is_empty() const { return low > high; }
};

struct Predicate {
    size_t       dim_idx;
    PredicateOp  op;
    CompositeKey value;
    CompositeKey value_high;
    std::vector<CompositeKey> in_values;

    static Predicate eq(size_t d, CompositeKey v) { return {d, PredicateOp::EQ, v, v, {}}; }
    static Predicate between(size_t d, CompositeKey lo, CompositeKey hi) {
        return {d, PredicateOp::BETWEEN, lo, hi, {}};
    }
    static Predicate in(size_t d, std::vector<CompositeKey> vals) {
        return {d, PredicateOp::IN, 0, 0, std::move(vals)};
    }
    static Predicate lt(size_t d, CompositeKey v)  { return {d, PredicateOp::LT,  v, 0, {}}; }
    static Predicate lte(size_t d, CompositeKey v) { return {d, PredicateOp::LTE, v, 0, {}}; }
    static Predicate gt(size_t d, CompositeKey v)  { return {d, PredicateOp::GT,  v, 0, {}}; }
    static Predicate gte(size_t d, CompositeKey v) { return {d, PredicateOp::GTE, v, 0, {}}; }
    static Predicate neq(size_t d, CompositeKey v) { return {d, PredicateOp::NEQ, v, 0, {}}; }
};

struct PredicateSet {
    std::vector<Predicate> predicates;

    KeyRange to_key_range(const CompositeKeySchema& schema) const {
        KeyRange range;
        for (auto& p : predicates) {
            if (p.dim_idx >= schema.dimensions.size()) continue;
            uint8_t off = schema.offset_of(p.dim_idx);
            CompositeKey mask = static_cast<CompositeKey>(schema.mask_of(p.dim_idx)) << off;
            switch (p.op) {
            case PredicateOp::EQ: {
                CompositeKey val_shifted = static_cast<CompositeKey>(p.value) << off;
                range.low  = (range.low  & ~mask) | val_shifted;
                range.high = (range.high & ~mask) | val_shifted;
                break;
            }
            case PredicateOp::BETWEEN: {
                CompositeKey lo_shifted = static_cast<CompositeKey>(p.value)      << off;
                CompositeKey hi_shifted = static_cast<CompositeKey>(p.value_high) << off;
                range.low  = (range.low  & ~mask) | lo_shifted;
                range.high = (range.high & ~mask) | hi_shifted;
                break;
            }
            case PredicateOp::GTE: {
                CompositeKey lo_shifted = static_cast<CompositeKey>(p.value) << off;
                range.low = (range.low & ~mask) | lo_shifted;
                break;
            }
            case PredicateOp::LTE: {
                CompositeKey hi_shifted = static_cast<CompositeKey>(p.value) << off;
                range.high = (range.high & ~mask) | hi_shifted;
                break;
            }
            default: break;
            }
        }
        return range;
    }

    // Fast per-record predicate evaluation. No null bitmap (no MVCC/null support in core).
    bool evaluate(CompositeKey key, const CompositeKeySchema& schema) const {
        for (auto& p : predicates) {
            uint8_t off = schema.offset_of(p.dim_idx);
            uint64_t mask = schema.mask_of(p.dim_idx);
            uint64_t dim_val = static_cast<uint64_t>((key >> off) & mask);

            switch (p.op) {
            case PredicateOp::EQ:
                if (dim_val != static_cast<uint64_t>(p.value)) return false; break;
            case PredicateOp::NEQ:
                if (dim_val == static_cast<uint64_t>(p.value)) return false; break;
            case PredicateOp::LT:
                if (dim_val >= static_cast<uint64_t>(p.value)) return false; break;
            case PredicateOp::LTE:
                if (dim_val > static_cast<uint64_t>(p.value)) return false; break;
            case PredicateOp::GT:
                if (dim_val <= static_cast<uint64_t>(p.value)) return false; break;
            case PredicateOp::GTE:
                if (dim_val < static_cast<uint64_t>(p.value)) return false; break;
            case PredicateOp::BETWEEN:
                if (dim_val < static_cast<uint64_t>(p.value)
                 || dim_val > static_cast<uint64_t>(p.value_high)) return false;
                break;
            case PredicateOp::IN: {
                bool found = false;
                for (auto& v : p.in_values) {
                    if (dim_val == static_cast<uint64_t>(v)) { found = true; break; }
                }
                if (!found) return false;
                break;
            }
            default: break;
            }
        }
        return true;
    }
};

// Minimal per-dimension stats used at bulk_load to populate subtree aggregates
// that drive beta-based pruning and O(1) aggregation shortcuts.
struct DimStats {
    uint64_t count   = 0;
    double   sum     = 0.0;
    uint64_t min_val = std::numeric_limits<uint64_t>::max();
    uint64_t max_val = 0;

    void add(uint64_t v) {
        count++;
        sum += static_cast<double>(v);
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }
    void merge(const DimStats& o) {
        count += o.count;
        sum   += o.sum;
        if (o.count > 0) {
            if (o.min_val < min_val) min_val = o.min_val;
            if (o.max_val > max_val) max_val = o.max_val;
        }
    }
};

struct CommittedAgg {
    uint64_t count = 0;
    double   sum   = 0.0;

    void add(uint64_t v) { count++; sum += static_cast<double>(v); }
    void sub(uint64_t v) { if (count > 0) { count--; sum -= static_cast<double>(v); } }
    void merge(const CommittedAgg& o) { count += o.count; sum += o.sum; }
};

struct SeqLock {
    std::atomic<uint64_t> seq{0};

    uint64_t read_begin() const {
        uint64_t s;
        do { s = seq.load(std::memory_order_acquire); } while (s & 1);
        return s;
    }
    bool read_validate(uint64_t s) const {
        std::atomic_thread_fence(std::memory_order_acquire);
        return seq.load(std::memory_order_relaxed) == s;
    }
    void write_lock()   { seq.fetch_add(1, std::memory_order_acquire); }
    void write_unlock() { seq.fetch_add(1, std::memory_order_release); }
};

struct TxnWriteEntry {
    void*    inner_node;
    size_t   dim_idx;
    uint64_t dim_val;
    bool     is_insert;
};

struct TxnContext {
    TxnId                       txn_id    = INVALID_TXN;
    Epoch                       read_epoch = 0;
    std::vector<TxnWriteEntry>  write_set;

    void reset() { txn_id = INVALID_TXN; read_epoch = 0; write_set.clear(); }
};

inline CompositeKeySchema make_default_sales_schema() {
    CompositeKeySchema schema;

    DimensionDesc year;
    year.name = "year"; year.bits = 8; year.encoding = DimensionDesc::LINEAR;
    year.base_value = 2000; year.scale = 1.0; year.nullable = true;
    schema.dimensions.push_back(year);

    DimensionDesc month;
    month.name = "month"; month.bits = 4; month.encoding = DimensionDesc::LINEAR;
    month.base_value = 1; month.scale = 1.0; month.nullable = true;
    schema.dimensions.push_back(month);

    DimensionDesc day;
    day.name = "day"; day.bits = 5; day.encoding = DimensionDesc::LINEAR;
    day.base_value = 1; day.scale = 1.0; day.nullable = true;
    schema.dimensions.push_back(day);

    DimensionDesc state;
    state.name = "state"; state.bits = 5; state.encoding = DimensionDesc::DICTIONARY;
    state.nullable = true;
    std::vector<std::string> states = {
        "AZ","CA","FL","GA","IL","MA","MI","NC","NJ","NY","OH","PA","TX","VA","WA"
    };
    for (size_t i = 0; i < states.size(); ++i) {
        state.dict_encode[states[i]] = i;
        state.dict_decode[i] = states[i];
    }
    schema.dimensions.push_back(state);

    DimensionDesc product;
    product.name = "product"; product.bits = 5; product.encoding = DimensionDesc::DICTIONARY;
    product.nullable = true;
    std::vector<std::string> products = {
        "Chair","Desk","Default","Headset","Keyboard","Laptop","Monitor","Mouse","Webcam"
    };
    for (size_t i = 0; i < products.size(); ++i) {
        product.dict_encode[products[i]] = i;
        product.dict_decode[i] = products[i];
    }
    schema.dimensions.push_back(product);

    DimensionDesc price;
    price.name = "price"; price.bits = 19; price.encoding = DimensionDesc::LINEAR;
    price.base_value = 0; price.scale = 100.0; price.nullable = true;
    schema.dimensions.push_back(price);

    DimensionDesc version;
    version.name = "version"; version.bits = 10; version.encoding = DimensionDesc::LINEAR;
    version.base_value = 0; version.scale = 100.0; version.nullable = true;
    schema.dimensions.push_back(version);

    schema.finalize();
    return schema;
}

}  // namespace hptree
