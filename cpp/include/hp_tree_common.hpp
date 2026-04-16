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
#include <variant>
#include <unordered_map>
#include <shared_mutex>
#include <mutex>
#include <atomic>
#include <cassert>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <memory>
#include <chrono>

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
static constexpr size_t  DEFAULT_BRANCHING_FACTOR = 200;
static constexpr size_t  DEFAULT_MAX_LEAF_SIZE    = 50;
static constexpr size_t  DEFAULT_MIN_LEAF_SIZE    = 20;
static constexpr double  DEFAULT_BETA_STRICT      = 1e-9;
static constexpr double  MIN_POSITIVE_VALUE       = 1e-7;
static constexpr size_t  DEFAULT_BUFFER_POOL_SIZE = 1024;
static constexpr size_t  DEFAULT_DELTA_BUFFER_CAP = 4096;
static constexpr double  TOMBSTONE_COMPACT_RATIO  = 0.30;
static constexpr double  DELTA_MERGE_RATIO        = 0.10;
static constexpr size_t  MAX_TREE_DEPTH           = 35;
static constexpr size_t  HISTOGRAM_BUCKETS        = 64;

enum class BetaStrategy : uint8_t {
    FIXED_STRICT    = 0,
    ARITHMETIC_MEAN = 1,
    MEDIAN          = 2,
    STDDEV_2X       = 3,
    STDDEV_6X       = 4,
    ADAPTIVE_LOCAL  = 5,
};

enum class NodeType : uint8_t {
    INTERNAL        = 0,
    LEAF            = 1,
    HOMOGENEOUS_LEAF= 2,
};

enum class WalOpType : uint8_t {
    INSERT          = 0,
    DELETE          = 1,
    UPDATE          = 2,
    SPLIT           = 3,
    MERGE           = 4,
    REBALANCE       = 5,
    CHECKPOINT      = 6,
    TXN_BEGIN       = 7,
    TXN_COMMIT      = 8,
    TXN_ABORT       = 9,
    BULK_LOAD       = 10,
    SCHEMA_CHANGE   = 11,
    BETA_RECALC     = 12,
    DELTA_FLUSH     = 13,
};

enum class LatchMode : uint8_t {
    SHARED    = 0,
    EXCLUSIVE = 1,
};

enum class ScanDirection : uint8_t {
    FORWARD  = 0,
    REVERSE  = 1,
};

enum class PredicateOp : uint8_t {
    EQ,
    NEQ,
    LT,
    LTE,
    GT,
    GTE,
    BETWEEN,
    IN,
    IS_NULL,
    IS_NOT_NULL,
    LIKE,
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
    bool         enable_wal         = true;
    bool         enable_mvcc        = true;
    bool         enable_buffer_pool = false;
    bool         enable_delta_buffer= true;
    bool         enable_aggregates  = true;
    bool         single_threaded    = false;
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

    void init_null_sentinel() {
        null_sentinel = (1ULL << bits) - 1;
    }

    uint64_t encode(int64_t val) const {
        int64_t shifted = val - base_value;
        if (shifted < 0) shifted = 0;
        uint64_t max_val = (1ULL << bits) - (nullable ? 2 : 1);
        return static_cast<uint64_t>(std::min(static_cast<uint64_t>(shifted), max_val));
    }

    uint64_t encode_float(double val) const {
        int64_t scaled = static_cast<int64_t>(val * scale);
        return encode(scaled);
    }

    uint64_t encode_string(const std::string& val) const {
        auto it = dict_encode.find(val);
        if (it != dict_encode.end()) return it->second;
        return 0;
    }

    int64_t decode_int(uint64_t coded) const {
        return static_cast<int64_t>(coded) + base_value;
    }

    double decode_float(uint64_t coded) const {
        return static_cast<double>(coded) / scale;
    }

    std::string decode_string(uint64_t coded) const {
        auto it = dict_decode.find(coded);
        if (it != dict_decode.end()) return it->second;
        return "N/A";
    }

    bool is_null_value(uint64_t coded) const {
        return nullable && coded == null_sentinel;
    }
};

struct CompositeKeySchema {
    std::vector<DimensionDesc> dimensions;
    size_t total_bits = 0;

    void finalize() {
        total_bits = 0;
        for (auto& d : dimensions) {
            if (d.nullable) d.init_null_sentinel();
            total_bits += d.bits;
        }
    }

    size_t dim_count() const { return dimensions.size(); }

    uint8_t offset_of(size_t dim_idx) const {
        uint8_t off = 0;
        for (size_t i = dim_idx + 1; i < dimensions.size(); ++i)
            off += dimensions[i].bits;
        return off;
    }

    uint64_t mask_of(size_t dim_idx) const {
        return (1ULL << dimensions[dim_idx].bits) - 1;
    }
};

using CompositeKey = __uint128_t;

static constexpr CompositeKey COMPOSITE_KEY_MIN = 0;
static constexpr CompositeKey COMPOSITE_KEY_MAX = ~static_cast<CompositeKey>(0);

struct CompositeKeyEncoder {
    const CompositeKeySchema& schema;

    explicit CompositeKeyEncoder(const CompositeKeySchema& s) : schema(s) {}

    CompositeKey encode(const std::vector<int64_t>& int_vals,
                        const std::vector<double>& float_vals,
                        const std::vector<std::string>& str_vals,
                        const std::vector<bool>& null_flags) const {
        CompositeKey key = 0;
        size_t ii = 0, fi = 0, si = 0;

        for (size_t d = 0; d < schema.dimensions.size(); ++d) {
            const auto& dim = schema.dimensions[d];
            uint64_t coded = 0;

            if (d < null_flags.size() && null_flags[d]) {
                coded = dim.null_sentinel;
            } else {
                switch (dim.encoding) {
                case DimensionDesc::LINEAR:
                    if (dim.scale != 1.0 && fi < float_vals.size()) {
                        coded = dim.encode_float(float_vals[fi++]);
                    } else if (ii < int_vals.size()) {
                        coded = dim.encode(int_vals[ii++]);
                    }
                    break;
                case DimensionDesc::DICTIONARY:
                    if (si < str_vals.size())
                        coded = dim.encode_string(str_vals[si++]);
                    break;
                }
            }

            uint8_t off = schema.offset_of(d);
            key |= static_cast<CompositeKey>(coded) << off;
        }
        return key;
    }

    struct Decoded {
        std::vector<int64_t>     int_vals;
        std::vector<double>      float_vals;
        std::vector<std::string> str_vals;
        std::vector<bool>        null_flags;
    };

    Decoded decode(CompositeKey key) const {
        Decoded result;
        result.null_flags.resize(schema.dimensions.size(), false);

        for (size_t d = 0; d < schema.dimensions.size(); ++d) {
            const auto& dim = schema.dimensions[d];
            uint8_t off = schema.offset_of(d);
            uint64_t mask = schema.mask_of(d);
            uint64_t coded = static_cast<uint64_t>((key >> off) & mask);

            if (dim.is_null_value(coded)) {
                result.null_flags[d] = true;
                continue;
            }

            switch (dim.encoding) {
            case DimensionDesc::LINEAR:
                if (dim.scale != 1.0)
                    result.float_vals.push_back(dim.decode_float(coded));
                else
                    result.int_vals.push_back(dim.decode_int(coded));
                break;
            case DimensionDesc::DICTIONARY:
                result.str_vals.push_back(dim.decode_string(coded));
                break;
            }
        }
        return result;
    }

    uint64_t extract_dim(CompositeKey key, size_t dim_idx) const {
        uint8_t off = schema.offset_of(dim_idx);
        uint64_t mask = schema.mask_of(dim_idx);
        return static_cast<uint64_t>((key >> off) & mask);
    }

    CompositeKey set_dim(CompositeKey key, size_t dim_idx, uint64_t val) const {
        uint8_t off = schema.offset_of(dim_idx);
        uint64_t mask = schema.mask_of(dim_idx);
        key &= ~(static_cast<CompositeKey>(mask) << off);
        key |= static_cast<CompositeKey>(val & mask) << off;
        return key;
    }
};

struct NullBitmap {
    std::vector<uint64_t> words;
    size_t                num_records = 0;
    size_t                num_dims    = 0;

    void init(size_t records, size_t dims) {
        num_records = records;
        num_dims = dims;
        size_t total_bits = records * dims;
        words.resize((total_bits + 63) / 64, 0);
    }

    void set_null(size_t record_idx, size_t dim_idx) {
        size_t bit = record_idx * num_dims + dim_idx;
        words[bit / 64] |= (1ULL << (bit % 64));
    }

    void clear_null(size_t record_idx, size_t dim_idx) {
        size_t bit = record_idx * num_dims + dim_idx;
        words[bit / 64] &= ~(1ULL << (bit % 64));
    }

    bool is_null(size_t record_idx, size_t dim_idx) const {
        size_t bit = record_idx * num_dims + dim_idx;
        return (words[bit / 64] >> (bit % 64)) & 1;
    }

    bool any_null_in_record(size_t record_idx) const {
        for (size_t d = 0; d < num_dims; ++d)
            if (is_null(record_idx, d)) return true;
        return false;
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

    struct Thresholds {
        double am;
        double median;
        double stddev_2x;
        double stddev_6x;
    };

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
            dvals[i] = v;
            sum += v;
            sum_sq += v * v;
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

        t.am = (b_total != std::numeric_limits<double>::infinity())
             ? b_total * 0.1 : strict_default;
        t.median = (median_val > 1.0) ? median_val * 0.001 : strict_default;
        t.stddev_2x = (std_val > 1.0) ? std_val * 2.0 : strict_default;
        t.stddev_6x = (std_val > 1.0) ? std_val * 6.0 : strict_default;

        t.am       = std::max(t.am, MIN_POSITIVE_VALUE);
        t.median   = std::max(t.median, MIN_POSITIVE_VALUE);
        t.stddev_2x= std::max(t.stddev_2x, MIN_POSITIVE_VALUE);
        t.stddev_6x= std::max(t.stddev_6x, MIN_POSITIVE_VALUE);
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

    static bool adaptive_should_stop(double child_beta, double parent_beta,
                                     size_t partition_size, size_t num_children) {
        if (child_beta == 0.0) return true;
        if (parent_beta <= 0.0 ||
            parent_beta == std::numeric_limits<double>::infinity())
            return false;
        if (child_beta == std::numeric_limits<double>::infinity())
            return false;

        double ratio = child_beta / parent_beta;
        double fanout_inv = 1.0 / static_cast<double>(std::max(num_children, size_t(2)));
        if (ratio < fanout_inv) return true;

        double n = static_cast<double>(partition_size);
        if (n > 1.0 && child_beta < 1.0 / (n * n)) return true;

        return false;
    }
};

struct VersionInfo {
    TxnId xmin = INVALID_TXN;
    TxnId xmax = TXN_COMMITTED;

    bool is_visible(TxnId reader_txn) const {
        if (xmin == INVALID_TXN) return false;
        if (xmin > reader_txn) return false;
        if (xmax != TXN_COMMITTED && xmax <= reader_txn) return false;
        return true;
    }

    bool is_deleted() const {
        return xmax != TXN_COMMITTED;
    }
};

struct Record {
    CompositeKey key;
    std::vector<uint8_t> payload;
    VersionInfo version;
    bool tombstone = false;

    bool operator<(const Record& o) const { return key < o.key; }
    bool operator==(const Record& o) const { return key == o.key; }
};

struct KeyRange {
    CompositeKey low  = COMPOSITE_KEY_MIN;
    CompositeKey high = COMPOSITE_KEY_MAX;

    bool contains(CompositeKey k) const { return k >= low && k <= high; }
    bool overlaps(const KeyRange& o) const {
        return low <= o.high && high >= o.low;
    }
    bool fully_contains(const KeyRange& o) const {
        return low <= o.low && high >= o.high;
    }
    bool is_empty() const { return low > high; }
};

struct Predicate {
    size_t      dim_idx;
    PredicateOp op;
    CompositeKey value;
    CompositeKey value_high;
    std::vector<CompositeKey> in_values;

    static Predicate eq(size_t d, CompositeKey v) {
        return {d, PredicateOp::EQ, v, v, {}};
    }
    static Predicate between(size_t d, CompositeKey lo, CompositeKey hi) {
        return {d, PredicateOp::BETWEEN, lo, hi, {}};
    }
    static Predicate in(size_t d, std::vector<CompositeKey> vals) {
        return {d, PredicateOp::IN, 0, 0, std::move(vals)};
    }
    static Predicate is_null(size_t d) {
        return {d, PredicateOp::IS_NULL, 0, 0, {}};
    }
    static Predicate is_not_null(size_t d) {
        return {d, PredicateOp::IS_NOT_NULL, 0, 0, {}};
    }
    static Predicate lt(size_t d, CompositeKey v) {
        return {d, PredicateOp::LT, v, 0, {}};
    }
    static Predicate lte(size_t d, CompositeKey v) {
        return {d, PredicateOp::LTE, v, 0, {}};
    }
    static Predicate gt(size_t d, CompositeKey v) {
        return {d, PredicateOp::GT, v, 0, {}};
    }
    static Predicate gte(size_t d, CompositeKey v) {
        return {d, PredicateOp::GTE, v, 0, {}};
    }
    static Predicate neq(size_t d, CompositeKey v) {
        return {d, PredicateOp::NEQ, v, 0, {}};
    }
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
            default:
                break;
            }
        }
        return range;
    }

    bool evaluate_record(CompositeKey key, const CompositeKeySchema& schema,
                         const NullBitmap* nulls, size_t rec_idx) const {
        for (auto& p : predicates) {
            uint8_t off = schema.offset_of(p.dim_idx);
            uint64_t mask = schema.mask_of(p.dim_idx);
            uint64_t dim_val = static_cast<uint64_t>((key >> off) & mask);
            bool is_n = (nulls && rec_idx < nulls->num_records)
                      ? nulls->is_null(rec_idx, p.dim_idx) : false;

            switch (p.op) {
            case PredicateOp::IS_NULL:
                if (!is_n) return false;
                break;
            case PredicateOp::IS_NOT_NULL:
                if (is_n) return false;
                break;
            case PredicateOp::EQ:
                if (is_n || dim_val != static_cast<uint64_t>(p.value))
                    return false;
                break;
            case PredicateOp::NEQ:
                if (is_n || dim_val == static_cast<uint64_t>(p.value))
                    return false;
                break;
            case PredicateOp::LT:
                if (is_n || dim_val >= static_cast<uint64_t>(p.value))
                    return false;
                break;
            case PredicateOp::LTE:
                if (is_n || dim_val > static_cast<uint64_t>(p.value))
                    return false;
                break;
            case PredicateOp::GT:
                if (is_n || dim_val <= static_cast<uint64_t>(p.value))
                    return false;
                break;
            case PredicateOp::GTE:
                if (is_n || dim_val < static_cast<uint64_t>(p.value))
                    return false;
                break;
            case PredicateOp::BETWEEN:
                if (is_n || dim_val < static_cast<uint64_t>(p.value)
                         || dim_val > static_cast<uint64_t>(p.value_high))
                    return false;
                break;
            case PredicateOp::IN: {
                if (is_n) return false;
                bool found = false;
                for (auto& v : p.in_values) {
                    if (dim_val == static_cast<uint64_t>(v)) { found = true; break; }
                }
                if (!found) return false;
                break;
            }
            default:
                break;
            }
        }
        return true;
    }
};

class NodeLatch {
    mutable std::shared_mutex mtx_;
public:
    void lock_shared()   const { mtx_.lock_shared(); }
    void unlock_shared() const { mtx_.unlock_shared(); }
    void lock()                { mtx_.lock(); }
    void unlock()              { mtx_.unlock(); }
    bool try_lock()            { return mtx_.try_lock(); }
    bool try_lock_shared() const { return mtx_.try_lock_shared(); }
};

struct AggregateStats {
    uint64_t count           = 0;
    uint64_t count_non_null  = 0;
    double   sum             = 0.0;
    double   min_val         = std::numeric_limits<double>::max();
    double   max_val         = std::numeric_limits<double>::lowest();
    double   sum_sq          = 0.0;

    void add(double v) {
        count++;
        count_non_null++;
        sum += v;
        sum_sq += v * v;
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }

    void remove(double v) {
        if (count == 0) return;
        count--;
        count_non_null--;
        sum -= v;
        sum_sq -= v * v;
    }

    void add_null() { count++; }

    void merge(const AggregateStats& o) {
        count += o.count;
        count_non_null += o.count_non_null;
        sum += o.sum;
        sum_sq += o.sum_sq;
        if (o.min_val < min_val) min_val = o.min_val;
        if (o.max_val > max_val) max_val = o.max_val;
    }

    double mean() const {
        return count_non_null > 0 ? sum / count_non_null : 0.0;
    }

    double variance() const {
        if (count_non_null < 2) return 0.0;
        double m = mean();
        return (sum_sq / count_non_null) - (m * m);
    }

    double stddev() const { return std::sqrt(std::max(variance(), 0.0)); }
};

struct PerDimAggregates {
    std::vector<AggregateStats> dims;

    void init(size_t num_dims) { dims.resize(num_dims); }

    void add_record(CompositeKey key, const CompositeKeySchema& schema,
                    const NullBitmap* nulls, size_t rec_idx) {
        for (size_t d = 0; d < schema.dimensions.size() && d < dims.size(); ++d) {
            bool is_n = (nulls && rec_idx < nulls->num_records)
                      ? nulls->is_null(rec_idx, d) : false;
            if (is_n) {
                dims[d].add_null();
            } else {
                uint8_t off = schema.offset_of(d);
                uint64_t mask = schema.mask_of(d);
                uint64_t coded = static_cast<uint64_t>((key >> off) & mask);
                dims[d].add(static_cast<double>(coded));
            }
        }
    }

    void remove_record(CompositeKey key, const CompositeKeySchema& schema,
                       const NullBitmap* nulls, size_t rec_idx) {
        for (size_t d = 0; d < schema.dimensions.size() && d < dims.size(); ++d) {
            bool is_n = (nulls && rec_idx < nulls->num_records)
                      ? nulls->is_null(rec_idx, d) : false;
            if (!is_n) {
                uint8_t off = schema.offset_of(d);
                uint64_t mask = schema.mask_of(d);
                uint64_t coded = static_cast<uint64_t>((key >> off) & mask);
                dims[d].remove(static_cast<double>(coded));
            }
        }
    }

    void merge(const PerDimAggregates& o) {
        for (size_t d = 0; d < dims.size() && d < o.dims.size(); ++d)
            dims[d].merge(o.dims[d]);
    }
};

static CompositeKeySchema make_default_sales_schema() {
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
