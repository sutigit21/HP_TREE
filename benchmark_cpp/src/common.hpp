#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "hp_tree_common.hpp"
#include <nlohmann/json.hpp>

namespace bench {

using Clock = std::chrono::steady_clock;

static inline double to_ms(Clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
}

class Timer {
    Clock::time_point start_;
public:
    Timer() : start_(Clock::now()) {}
    void reset() { start_ = Clock::now(); }
    double elapsed_ms() const { return to_ms(Clock::now() - start_); }
};

struct DatasetRecord {
    hptree::CompositeKey key;
    uint64_t             value;
};

static constexpr char DATASET_MAGIC[8] = {'H','P','D','S','0','0','0','1'};

inline std::vector<DatasetRecord> load_dataset(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "ERROR: cannot open dataset " << path << "\n";
        std::exit(2);
    }

    char magic[8];
    if (!in.read(magic, 8)) {
        std::cerr << "ERROR: failed reading magic from " << path << "\n";
        std::exit(2);
    }
    if (std::memcmp(magic, DATASET_MAGIC, 8) != 0) {
        std::cerr << "ERROR: bad magic in " << path << "\n";
        std::exit(2);
    }

    uint64_t n = 0;
    if (!in.read(reinterpret_cast<char*>(&n), sizeof(n))) {
        std::cerr << "ERROR: failed reading record count from " << path << "\n";
        std::exit(2);
    }

    std::vector<DatasetRecord> recs;
    recs.resize(static_cast<size_t>(n));

    for (uint64_t i = 0; i < n; ++i) {
        uint64_t lo = 0, hi = 0, v = 0;
        if (!in.read(reinterpret_cast<char*>(&lo), sizeof(lo)) ||
            !in.read(reinterpret_cast<char*>(&hi), sizeof(hi)) ||
            !in.read(reinterpret_cast<char*>(&v),  sizeof(v))) {
            std::cerr << "ERROR: truncated dataset " << path
                      << " at record " << i << " of " << n << "\n";
            std::exit(2);
        }
        hptree::CompositeKey k = (static_cast<hptree::CompositeKey>(hi) << 64)
                               | static_cast<hptree::CompositeKey>(lo);
        recs[i] = {k, v};
    }
    return recs;
}

struct QuerySpec {
    uint64_t n_records;

    std::vector<hptree::CompositeKey> point_lookup_keys;

    struct Range { hptree::CompositeKey lo, hi; };
    Range narrow_range;
    Range wide_range;
    Range range_agg;
    size_t range_agg_dim;

    struct DimFilter { size_t dim; uint64_t value; };
    DimFilter dim_filter;

    std::vector<DimFilter> multi_dim_filter;

    std::vector<hptree::CompositeKey> insert_keys;
    std::vector<uint64_t>             insert_values;
    std::vector<hptree::CompositeKey> delete_keys;

    struct DimRange { size_t dim; uint64_t lo, hi; };
    std::vector<DimRange> hypercube;

    struct GroupBySpec {
        size_t   filter_dim;
        uint64_t filter_val;
        size_t   group_dim;
        size_t   agg_dim;
    } groupby;

    size_t correlated_group_dim;
    size_t correlated_agg_dim;

    std::vector<Range> moving_windows;
    size_t moving_agg_dim;

    std::vector<std::vector<DimFilter>> adhoc_drills;
};

inline hptree::CompositeKey json_to_u128(const nlohmann::json& j) {
    uint64_t lo = j.at(0).get<uint64_t>();
    uint64_t hi = j.at(1).get<uint64_t>();
    return (static_cast<hptree::CompositeKey>(hi) << 64)
         | static_cast<hptree::CompositeKey>(lo);
}

inline nlohmann::json u128_to_json(hptree::CompositeKey k) {
    uint64_t lo = static_cast<uint64_t>(k);
    uint64_t hi = static_cast<uint64_t>(k >> 64);
    return nlohmann::json::array({lo, hi});
}

inline QuerySpec load_query_spec(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "ERROR: cannot open query spec " << path << "\n";
        std::exit(2);
    }
    nlohmann::json j;
    in >> j;

    QuerySpec qs;
    qs.n_records = j.at("n_records").get<uint64_t>();

    for (auto& k : j.at("point_lookup_keys"))
        qs.point_lookup_keys.push_back(json_to_u128(k));

    qs.narrow_range = {json_to_u128(j.at("narrow_range").at("lo")),
                       json_to_u128(j.at("narrow_range").at("hi"))};
    qs.wide_range = {json_to_u128(j.at("wide_range").at("lo")),
                     json_to_u128(j.at("wide_range").at("hi"))};
    qs.range_agg = {json_to_u128(j.at("range_agg").at("lo")),
                    json_to_u128(j.at("range_agg").at("hi"))};
    qs.range_agg_dim = j.at("range_agg").at("agg_dim").get<size_t>();

    qs.dim_filter = {j.at("dim_filter").at("dim").get<size_t>(),
                     j.at("dim_filter").at("value").get<uint64_t>()};

    for (auto& f : j.at("multi_dim_filter"))
        qs.multi_dim_filter.push_back({f.at("dim").get<size_t>(),
                                       f.at("value").get<uint64_t>()});

    for (auto& k : j.at("insert_keys"))
        qs.insert_keys.push_back(json_to_u128(k));
    for (auto& v : j.at("insert_values"))
        qs.insert_values.push_back(v.get<uint64_t>());
    for (auto& k : j.at("delete_keys"))
        qs.delete_keys.push_back(json_to_u128(k));

    for (auto& h : j.at("hypercube"))
        qs.hypercube.push_back({h.at("dim").get<size_t>(),
                                h.at("lo").get<uint64_t>(),
                                h.at("hi").get<uint64_t>()});

    qs.groupby.filter_dim = j.at("groupby").at("filter_dim").get<size_t>();
    qs.groupby.filter_val = j.at("groupby").at("filter_val").get<uint64_t>();
    qs.groupby.group_dim  = j.at("groupby").at("group_dim").get<size_t>();
    qs.groupby.agg_dim    = j.at("groupby").at("agg_dim").get<size_t>();

    qs.correlated_group_dim = j.at("correlated").at("group_dim").get<size_t>();
    qs.correlated_agg_dim   = j.at("correlated").at("agg_dim").get<size_t>();

    for (auto& w : j.at("moving_windows"))
        qs.moving_windows.push_back({json_to_u128(w.at("lo")),
                                     json_to_u128(w.at("hi"))});
    qs.moving_agg_dim = j.at("moving").at("agg_dim").get<size_t>();

    for (auto& d : j.at("adhoc_drills")) {
        std::vector<QuerySpec::DimFilter> drill;
        for (auto& f : d)
            drill.push_back({f.at("dim").get<size_t>(),
                             f.at("value").get<uint64_t>()});
        qs.adhoc_drills.push_back(std::move(drill));
    }
    return qs;
}

struct QueryResult {
    std::string name;
    double      elapsed_ms = 0.0;
    uint64_t    result_count = 0;
    double      result_sum   = 0.0;
    uint64_t    key_checksum = 0;
    std::string extra;
};

inline uint64_t key_checksum_u128(hptree::CompositeKey k) {
    uint64_t lo = static_cast<uint64_t>(k);
    uint64_t hi = static_cast<uint64_t>(k >> 64);
    return lo ^ (hi * 0x9E3779B97F4A7C15ULL);
}

inline void write_results(const std::string& tree_name,
                          const std::string& dataset_name,
                          const std::vector<QueryResult>& results,
                          const std::string& output_path) {
    nlohmann::json j;
    j["tree"] = tree_name;
    j["dataset"] = dataset_name;
    j["queries"] = nlohmann::json::array();
    for (auto& r : results) {
        nlohmann::json q;
        q["name"]        = r.name;
        q["elapsed_ms"]  = r.elapsed_ms;
        q["count"]       = r.result_count;
        q["sum"]         = r.result_sum;
        q["checksum"]    = r.key_checksum;
        q["extra"]       = r.extra;
        j["queries"].push_back(q);
    }
    std::ofstream out(output_path);
    out << j.dump(2) << "\n";
}

static inline std::string arg_get(int argc, char** argv,
                                  const std::string& flag,
                                  const std::string& defv = "") {
    for (int i = 1; i < argc - 1; ++i) {
        if (argv[i] == flag) return argv[i + 1];
    }
    return defv;
}

}  // namespace bench
