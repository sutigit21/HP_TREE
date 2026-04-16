#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.hpp"
#include "hp_tree.hpp"

namespace {

using Key = hptree::CompositeKey;

inline uint64_t extract_dim_u64(const hptree::CompositeKeySchema& schema,
                                Key key, size_t dim) {
    uint8_t  off  = schema.offset_of(dim);
    uint64_t mask = schema.mask_of(dim);
    return static_cast<uint64_t>((key >> off) & mask);
}

inline hptree::PredicateSet build_eq_predicate(size_t dim, uint64_t value) {
    hptree::PredicateSet ps;
    ps.predicates.push_back(hptree::Predicate::eq(dim, static_cast<Key>(value)));
    return ps;
}

inline hptree::PredicateSet build_multi_eq_predicate(
        const std::vector<bench::QuerySpec::DimFilter>& filters) {
    hptree::PredicateSet ps;
    for (auto& f : filters)
        ps.predicates.push_back(hptree::Predicate::eq(f.dim, static_cast<Key>(f.value)));
    return ps;
}

inline hptree::PredicateSet build_hypercube_predicate(
        const std::vector<bench::QuerySpec::DimRange>& ranges) {
    hptree::PredicateSet ps;
    for (auto& r : ranges)
        ps.predicates.push_back(hptree::Predicate::between(
            r.dim, static_cast<Key>(r.lo), static_cast<Key>(r.hi)));
    return ps;
}

}  // namespace

int main(int argc, char** argv) {
    std::string dataset_path = bench::arg_get(argc, argv, "--dataset");
    std::string spec_path    = bench::arg_get(argc, argv, "--spec");
    std::string output_path  = bench::arg_get(argc, argv, "--output");
    std::string dist_name    = bench::arg_get(argc, argv, "--dist", "unknown");

    if (dataset_path.empty() || spec_path.empty() || output_path.empty()) {
        std::cerr << "Usage: hp_runner --dataset <path> --spec <path> "
                     "--output <path> [--dist <name>]\n";
        return 1;
    }

    std::cerr << "[hp] loading dataset " << dataset_path << "...\n";
    auto dataset = bench::load_dataset(dataset_path);
    std::cerr << "[hp] N=" << dataset.size() << "\n";

    auto qs     = bench::load_query_spec(spec_path);
    auto schema = hptree::make_default_sales_schema();

    hptree::HPTreeConfig cfg;
    cfg.max_leaf_size       = 256;
    cfg.branching_factor    = 32;
    cfg.beta_strategy       = hptree::BetaStrategy::ARITHMETIC_MEAN;
    cfg.enable_delta_buffer = true;
    cfg.delta_buffer_cap    = 8192;
    cfg.enable_wal          = false;
    cfg.enable_mvcc         = true;
    cfg.enable_aggregates   = true;
    cfg.enable_buffer_pool  = false;
    cfg.single_threaded     = true;

    hptree::HPTree tree(cfg, schema);
    std::vector<bench::QueryResult> results;

    // =========================================================================
    //  Q1: Bulk Load
    // =========================================================================
    {
        std::vector<hptree::Record> recs;
        recs.reserve(dataset.size());
        for (auto& r : dataset) {
            hptree::Record rec;
            rec.key   = r.key;
            rec.value = r.value;
            recs.push_back(rec);
        }
        bench::Timer t;
        tree.bulk_load(std::move(recs));
        double ms = t.elapsed_ms();

        bench::QueryResult qr;
        qr.name = "Q1_bulk_load";
        qr.elapsed_ms = ms;
        qr.result_count = tree.size();
        results.push_back(qr);
        std::cerr << "[hp] Q1 bulk_load " << ms << "ms  size=" << tree.size() << "\n";
    }

    // =========================================================================
    //  Q2: Point Lookup (2000 keys)
    // =========================================================================
    {
        bench::Timer t;
        uint64_t hits = 0, chk = 0;
        for (auto k : qs.point_lookup_keys) {
            auto rs = tree.search(k);
            for (auto& r : rs) {
                hits++;
                chk ^= bench::key_checksum_u128(r.key);
            }
        }
        double ms = t.elapsed_ms();
        results.push_back({ "Q2_point_lookup", ms, hits, 0.0, chk, "" });
        std::cerr << "[hp] Q2 point_lookup " << ms << "ms  hits=" << hits << "\n";
    }

    // =========================================================================
    //  Q3: Narrow Range
    // =========================================================================
    {
        bench::Timer t;
        auto rs = tree.range_search(qs.narrow_range.lo, qs.narrow_range.hi);
        uint64_t chk = 0; double sum = 0;
        for (auto& r : rs) {
            chk ^= bench::key_checksum_u128(r.key);
            sum += static_cast<double>(extract_dim_u64(schema, r.key, 5));
        }
        double ms = t.elapsed_ms();
        results.push_back({ "Q3_narrow_range", ms, rs.size(), sum, chk, "" });
        std::cerr << "[hp] Q3 narrow_range " << ms << "ms  n=" << rs.size() << "\n";
    }

    // =========================================================================
    //  Q4: Wide Range
    // =========================================================================
    {
        bench::Timer t;
        auto rs = tree.range_search(qs.wide_range.lo, qs.wide_range.hi);
        uint64_t chk = 0; double sum = 0;
        for (auto& r : rs) {
            chk ^= bench::key_checksum_u128(r.key);
            sum += static_cast<double>(extract_dim_u64(schema, r.key, 5));
        }
        double ms = t.elapsed_ms();
        results.push_back({ "Q4_wide_range", ms, rs.size(), sum, chk, "" });
        std::cerr << "[hp] Q4 wide_range " << ms << "ms  n=" << rs.size() << "\n";
    }

    // =========================================================================
    //  Q5: Dim Filter (predicate_search uses per-leaf dim_min/max pruning)
    // =========================================================================
    {
        auto ps = build_eq_predicate(qs.dim_filter.dim, qs.dim_filter.value);
        bench::Timer t;
        auto rs = tree.predicate_search(ps);
        uint64_t chk = 0;
        for (auto& r : rs) chk ^= bench::key_checksum_u128(r.key);
        double ms = t.elapsed_ms();
        results.push_back({ "Q5_dim_filter", ms, rs.size(), 0.0, chk, "" });
        std::cerr << "[hp] Q5 dim_filter " << ms << "ms  n=" << rs.size() << "\n";
    }

    // =========================================================================
    //  Q6: Multi-Dim Filter
    // =========================================================================
    {
        auto ps = build_multi_eq_predicate(qs.multi_dim_filter);
        bench::Timer t;
        auto rs = tree.predicate_search(ps);
        uint64_t chk = 0;
        for (auto& r : rs) chk ^= bench::key_checksum_u128(r.key);
        double ms = t.elapsed_ms();
        results.push_back({ "Q6_multi_dim_filter", ms, rs.size(), 0.0, chk, "" });
        std::cerr << "[hp] Q6 multi_dim_filter " << ms << "ms  n=" << rs.size() << "\n";
    }

    // =========================================================================
    //  Q7: Range Aggregation SUM(dim)
    // =========================================================================
    {
        bench::Timer t;
        auto agg = tree.aggregate_dim(qs.range_agg_dim,
                                      qs.range_agg.lo, qs.range_agg.hi);
        double ms = t.elapsed_ms();
        results.push_back({ "Q7_range_agg", ms, agg.count, agg.sum, 0, "" });
        std::cerr << "[hp] Q7 range_agg " << ms << "ms  n=" << agg.count
                  << " sum=" << agg.sum << "\n";
    }

    // =========================================================================
    //  Q8: Full Scan (iterator)
    // =========================================================================
    {
        bench::Timer t;
        uint64_t cnt = 0, chk = 0;
        auto it = tree.runner_begin();
        while (it.valid()) {
            cnt++;
            chk ^= bench::key_checksum_u128(it.key());
            it.next();
        }
        double ms = t.elapsed_ms();
        results.push_back({ "Q8_full_scan", ms, cnt, 0.0, chk, "" });
        std::cerr << "[hp] Q8 full_scan " << ms << "ms  n=" << cnt << "\n";
    }

    // =========================================================================
    //  Q9: Single Inserts (delta buffer)
    // =========================================================================
    {
        bench::Timer t;
        for (size_t i = 0; i < qs.insert_keys.size(); ++i) {
            hptree::Record rec;
            rec.key   = qs.insert_keys[i];
            rec.value = qs.insert_values[i];
            tree.insert(rec);
        }
        tree.flush_delta();
        double ms = t.elapsed_ms();
        results.push_back({ "Q9_single_inserts", ms,
                            (uint64_t)qs.insert_keys.size(), 0.0, 0, "" });
        std::cerr << "[hp] Q9 single_inserts " << ms << "ms  n="
                  << qs.insert_keys.size() << "\n";
    }

    // =========================================================================
    //  Q10: Deletes
    // =========================================================================
    {
        bench::Timer t;
        uint64_t removed = 0;
        for (auto k : qs.delete_keys) {
            if (tree.remove(k)) removed++;
        }
        tree.flush_delta();
        double ms = t.elapsed_ms();
        results.push_back({ "Q10_deletes", ms, removed, 0.0, 0, "" });
        std::cerr << "[hp] Q10 deletes " << ms << "ms  removed=" << removed << "\n";
    }

    // =========================================================================
    //  Q11: Hypercube (3-dim bounding box) via BETWEEN predicates
    // =========================================================================
    {
        auto ps = build_hypercube_predicate(qs.hypercube);
        bench::Timer t;
        auto rs = tree.predicate_search(ps);
        uint64_t chk = 0;
        for (auto& r : rs) chk ^= bench::key_checksum_u128(r.key);
        double ms = t.elapsed_ms();
        results.push_back({ "Q11_hypercube", ms, rs.size(), 0.0, chk, "" });
        std::cerr << "[hp] Q11 hypercube " << ms << "ms  n=" << rs.size() << "\n";
    }

    // =========================================================================
    //  Q12: Group-By Agg — filter + group_by + sum
    // =========================================================================
    {
        auto ps = build_eq_predicate(qs.groupby.filter_dim, qs.groupby.filter_val);
        bench::Timer t;
        auto rs = tree.predicate_search(ps);
        std::unordered_map<uint64_t, std::pair<uint64_t,double>> groups;
        for (auto& r : rs) {
            uint64_t g = extract_dim_u64(schema, r.key, qs.groupby.group_dim);
            double v = static_cast<double>(
                extract_dim_u64(schema, r.key, qs.groupby.agg_dim));
            auto& e = groups[g];
            e.first++;
            e.second += v;
        }
        double ms = t.elapsed_ms();
        double total_sum = 0;
        uint64_t chk = 0;
        for (auto& [g, v] : groups) {
            total_sum += v.second;
            chk ^= (g * 0x9E3779B97F4A7C15ULL)
                 ^ static_cast<uint64_t>(v.second);
        }
        results.push_back({ "Q12_group_by", ms, rs.size(), total_sum, chk,
                            std::to_string(groups.size()) + " groups" });
        std::cerr << "[hp] Q12 group_by " << ms << "ms  groups=" << groups.size()
                  << " sum=" << total_sum << "\n";
    }

    // =========================================================================
    //  Q13: Correlated Subquery (2 full scans)
    // =========================================================================
    {
        bench::Timer t;
        std::unordered_map<uint64_t, std::pair<uint64_t,double>> stats;
        auto it = tree.runner_begin();
        while (it.valid()) {
            Key k = it.key();
            uint64_t g = extract_dim_u64(schema, k, qs.correlated_group_dim);
            double v = static_cast<double>(
                extract_dim_u64(schema, k, qs.correlated_agg_dim));
            auto& e = stats[g];
            e.first++;
            e.second += v;
            it.next();
        }
        std::unordered_map<uint64_t, double> means;
        means.reserve(stats.size());
        for (auto& [g, s] : stats)
            means[g] = s.first > 0 ? s.second / static_cast<double>(s.first) : 0.0;

        uint64_t above = 0;
        auto it2 = tree.runner_begin();
        while (it2.valid()) {
            Key k = it2.key();
            uint64_t g = extract_dim_u64(schema, k, qs.correlated_group_dim);
            double v = static_cast<double>(
                extract_dim_u64(schema, k, qs.correlated_agg_dim));
            auto m = means.find(g);
            if (m != means.end() && v > m->second) above++;
            it2.next();
        }
        double ms = t.elapsed_ms();
        results.push_back({ "Q13_correlated", ms, above, 0.0, 0,
                            std::to_string(stats.size()) + " groups" });
        std::cerr << "[hp] Q13 correlated " << ms << "ms  above=" << above << "\n";
    }

    // =========================================================================
    //  Q14: Moving Window (12x aggregate_dim over sliding range)
    // =========================================================================
    {
        bench::Timer t;
        double total_sum = 0;
        uint64_t total_cnt = 0;
        uint64_t chk = 0;
        for (auto& w : qs.moving_windows) {
            auto agg = tree.aggregate_dim(qs.moving_agg_dim, w.lo, w.hi);
            total_sum += agg.sum;
            total_cnt += agg.count;
            chk ^= static_cast<uint64_t>(agg.sum);
        }
        double ms = t.elapsed_ms();
        results.push_back({ "Q14_moving_window", ms, total_cnt, total_sum, chk, "" });
        std::cerr << "[hp] Q14 moving_window " << ms << "ms  total_n="
                  << total_cnt << "\n";
    }

    // =========================================================================
    //  Q15: Ad-Hoc Drill (30x predicate_search)
    // =========================================================================
    {
        bench::Timer t;
        uint64_t total_hits = 0, chk = 0;
        for (auto& drill : qs.adhoc_drills) {
            auto ps = build_multi_eq_predicate(drill);
            auto rs = tree.predicate_search(ps);
            total_hits += rs.size();
            for (auto& r : rs) chk ^= bench::key_checksum_u128(r.key);
        }
        double ms = t.elapsed_ms();
        results.push_back({ "Q15_adhoc_drill", ms, total_hits, 0.0, chk,
                            std::to_string(qs.adhoc_drills.size()) + " drills" });
        std::cerr << "[hp] Q15 adhoc_drill " << ms << "ms  total_hits="
                  << total_hits << "\n";
    }

    bench::write_results("hp", dist_name, results, output_path);
    std::cerr << "[hp] wrote " << output_path << "\n";
    return 0;
}
