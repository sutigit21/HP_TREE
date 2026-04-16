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
        uint64_t cnt = 0, chk = 0; double sum = 0;
        auto it = tree.lower_bound(qs.narrow_range.lo);
        while (it.valid() && it.key() <= qs.narrow_range.hi) {
            chk ^= bench::key_checksum_u128(it.key());
            sum += static_cast<double>(extract_dim_u64(schema, it.key(), 5));
            ++cnt; ++it;
        }
        double ms = t.elapsed_ms();
        results.push_back({ "Q3_narrow_range", ms, cnt, sum, chk, "" });
        std::cerr << "[hp] Q3 narrow_range " << ms << "ms  n=" << cnt << "\n";
    }

    // =========================================================================
    //  Q4: Wide Range
    // =========================================================================
    {
        bench::Timer t;
        uint64_t cnt = 0, chk = 0; double sum = 0;
        auto it = tree.lower_bound(qs.wide_range.lo);
        while (it.valid() && it.key() <= qs.wide_range.hi) {
            chk ^= bench::key_checksum_u128(it.key());
            sum += static_cast<double>(extract_dim_u64(schema, it.key(), 5));
            ++cnt; ++it;
        }
        double ms = t.elapsed_ms();
        results.push_back({ "Q4_wide_range", ms, cnt, sum, chk, "" });
        std::cerr << "[hp] Q4 wide_range " << ms << "ms  n=" << cnt << "\n";
    }

    // =========================================================================
    //  Q5: Dim Filter (predicate_search uses per-leaf dim_min/max pruning)
    // =========================================================================
    {
        auto ps = build_eq_predicate(qs.dim_filter.dim, qs.dim_filter.value);
        bench::Timer t;
        uint64_t cnt = 0, chk = 0;
        tree.predicate_search_cb(ps, [&](Key k, uint64_t){
            ++cnt; chk ^= bench::key_checksum_u128(k);
        });
        double ms = t.elapsed_ms();
        results.push_back({ "Q5_dim_filter", ms, cnt, 0.0, chk, "" });
        std::cerr << "[hp] Q5 dim_filter " << ms << "ms  n=" << cnt << "\n";
    }

    // =========================================================================
    //  Q6: Multi-Dim Filter
    // =========================================================================
    {
        auto ps = build_multi_eq_predicate(qs.multi_dim_filter);
        bench::Timer t;
        uint64_t cnt = 0, chk = 0;
        tree.predicate_search_cb(ps, [&](Key k, uint64_t){
            ++cnt; chk ^= bench::key_checksum_u128(k);
        });
        double ms = t.elapsed_ms();
        results.push_back({ "Q6_multi_dim_filter", ms, cnt, 0.0, chk, "" });
        std::cerr << "[hp] Q6 multi_dim_filter " << ms << "ms  n=" << cnt << "\n";
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
        uint64_t cnt = 0, chk = 0;
        tree.predicate_search_cb(ps, [&](Key k, uint64_t){
            ++cnt; chk ^= bench::key_checksum_u128(k);
        });
        double ms = t.elapsed_ms();
        results.push_back({ "Q11_hypercube", ms, cnt, 0.0, chk, "" });
        std::cerr << "[hp] Q11 hypercube " << ms << "ms  n=" << cnt << "\n";
    }

    // =========================================================================
    //  Q12: Group-By Agg — filter + group_by + sum
    // =========================================================================
    {
        auto ps = build_eq_predicate(qs.groupby.filter_dim, qs.groupby.filter_val);
        bench::Timer t;
        std::unordered_map<uint64_t, std::pair<uint64_t,double>> groups;
        uint64_t total = 0;
        tree.predicate_search_cb(ps, [&](Key k, uint64_t){
            uint64_t g = extract_dim_u64(schema, k, qs.groupby.group_dim);
            double v = static_cast<double>(
                extract_dim_u64(schema, k, qs.groupby.agg_dim));
            auto& e = groups[g];
            e.first++;
            e.second += v;
            ++total;
        });
        double ms = t.elapsed_ms();
        double total_sum = 0;
        uint64_t chk = 0;
        for (auto& [g, v] : groups) {
            total_sum += v.second;
            chk ^= (g * 0x9E3779B97F4A7C15ULL)
                 ^ static_cast<uint64_t>(v.second);
        }
        results.push_back({ "Q12_group_by", ms, total, total_sum, chk,
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
            tree.predicate_search_cb(ps, [&](Key k, uint64_t){
                ++total_hits; chk ^= bench::key_checksum_u128(k);
            });
        }
        double ms = t.elapsed_ms();
        results.push_back({ "Q15_adhoc_drill", ms, total_hits, 0.0, chk,
                            std::to_string(qs.adhoc_drills.size()) + " drills" });
        std::cerr << "[hp] Q15 adhoc_drill " << ms << "ms  total_hits="
                  << total_hits << "\n";
    }

    // Schema dim constants (must match make_default_sales_schema)
    constexpr size_t DIM_YEAR = 0, DIM_MONTH = 1, /*DIM_DAY=2,*/
                     DIM_STATE = 3, DIM_PRODUCT = 4, DIM_PRICE = 5;
    constexpr uint64_t YEAR_2021 = 21, YEAR_2022 = 22;

    // =========================================================================
    //  Q16: Top-K groups —
    //        WHERE year=2022 GROUP BY product SUM(price) ORDER BY sum DESC LIMIT 5
    // =========================================================================
    {
        auto ps = build_eq_predicate(DIM_YEAR, YEAR_2022);
        bench::Timer t;
        std::unordered_map<uint64_t, std::pair<uint64_t,double>> groups;
        tree.predicate_search_cb(ps, [&](Key k, uint64_t){
            uint64_t g = extract_dim_u64(schema, k, DIM_PRODUCT);
            double v = static_cast<double>(extract_dim_u64(schema, k, DIM_PRICE));
            auto& e = groups[g];
            e.first++; e.second += v;
        });
        std::vector<std::pair<uint64_t,double>> ranked;
        ranked.reserve(groups.size());
        for (auto& [g, v] : groups) ranked.emplace_back(g, v.second);
        constexpr size_t K = 5;
        size_t take = std::min(K, ranked.size());
        std::partial_sort(ranked.begin(), ranked.begin() + take, ranked.end(),
                          [](const auto& a, const auto& b){ return a.second > b.second; });
        double ms = t.elapsed_ms();
        double top_sum = 0; uint64_t chk = 0;
        for (size_t i = 0; i < take; ++i) {
            top_sum += ranked[i].second;
            chk ^= (ranked[i].first * 0x9E3779B97F4A7C15ULL)
                 ^ static_cast<uint64_t>(ranked[i].second);
        }
        results.push_back({ "Q16_topk_groups", ms, take, top_sum, chk,
                            std::to_string(groups.size()) + " groups" });
        std::cerr << "[hp] Q16 topk_groups " << ms << "ms  top" << take
                  << "_sum=" << top_sum << "\n";
    }

    // =========================================================================
    //  Q17: HAVING —
    //        GROUP BY state SUM(price) HAVING sum > threshold  (all records)
    // =========================================================================
    {
        constexpr double HAVING_THRESHOLD = 5.0e9;
        bench::Timer t;
        std::unordered_map<uint64_t, double> sums;
        auto it = tree.runner_begin();
        while (it.valid()) {
            uint64_t g = extract_dim_u64(schema, it.key(), DIM_STATE);
            double v = static_cast<double>(extract_dim_u64(schema, it.key(), DIM_PRICE));
            sums[g] += v;
            it.next();
        }
        uint64_t kept = 0; double kept_sum = 0; uint64_t chk = 0;
        for (auto& [g, s] : sums) {
            if (s > HAVING_THRESHOLD) {
                kept++; kept_sum += s;
                chk ^= (g * 0x9E3779B97F4A7C15ULL) ^ static_cast<uint64_t>(s);
            }
        }
        double ms = t.elapsed_ms();
        results.push_back({ "Q17_having_clause", ms, kept, kept_sum, chk,
                            std::to_string(sums.size()) + " groups total" });
        std::cerr << "[hp] Q17 having_clause " << ms << "ms  kept="
                  << kept << "/" << sums.size() << "\n";
    }

    // =========================================================================
    //  Q18: Year/Month Rollup —
    //        year IN [20,23] GROUP BY (year,month) SUM(price) COUNT(*)
    //        + compute per-year totals (rollup)
    // =========================================================================
    {
        bench::Timer t;
        std::unordered_map<uint64_t, std::pair<uint64_t,double>> ym;  // (yr<<4)|mo
        auto it = tree.lower_bound(qs.wide_range.lo);
        while (it.valid() && it.key() <= qs.wide_range.hi) {
            uint64_t y = extract_dim_u64(schema, it.key(), DIM_YEAR);
            uint64_t m = extract_dim_u64(schema, it.key(), DIM_MONTH);
            double v = static_cast<double>(extract_dim_u64(schema, it.key(), DIM_PRICE));
            auto& e = ym[(y << 4) | m];
            e.first++; e.second += v;
            ++it;
        }
        // Rollup: per-year totals.
        std::unordered_map<uint64_t, std::pair<uint64_t,double>> yr_totals;
        for (auto& [k, v] : ym) {
            auto& e = yr_totals[k >> 4];
            e.first  += v.first;
            e.second += v.second;
        }
        double ms = t.elapsed_ms();
        double total_sum = 0; uint64_t total_cnt = 0; uint64_t chk = 0;
        for (auto& [k, v] : ym) {
            total_sum += v.second; total_cnt += v.first;
            chk ^= (k * 0x9E3779B97F4A7C15ULL) ^ static_cast<uint64_t>(v.second);
        }
        results.push_back({ "Q18_ym_rollup", ms, total_cnt, total_sum, chk,
                            std::to_string(ym.size()) + " ym / "
                            + std::to_string(yr_totals.size()) + " yr" });
        std::cerr << "[hp] Q18 ym_rollup " << ms << "ms  ym_groups="
                  << ym.size() << " yr_groups=" << yr_totals.size() << "\n";
    }

    // =========================================================================
    //  Q19: Correlated multi-dim partition —
    //        COUNT(*) WHERE price > AVG(price) PARTITION BY (state,product)
    //        Two full scans; second compares each row to its partition mean.
    // =========================================================================
    {
        bench::Timer t;
        std::unordered_map<uint64_t, std::pair<uint64_t,double>> stats;
        auto it = tree.runner_begin();
        while (it.valid()) {
            uint64_t s = extract_dim_u64(schema, it.key(), DIM_STATE);
            uint64_t p = extract_dim_u64(schema, it.key(), DIM_PRODUCT);
            uint64_t g = (s << 8) | p;
            double v = static_cast<double>(extract_dim_u64(schema, it.key(), DIM_PRICE));
            auto& e = stats[g];
            e.first++; e.second += v;
            it.next();
        }
        std::unordered_map<uint64_t, double> means;
        means.reserve(stats.size());
        for (auto& [g, s] : stats)
            means[g] = s.first > 0 ? s.second / static_cast<double>(s.first) : 0.0;

        uint64_t above = 0;
        auto it2 = tree.runner_begin();
        while (it2.valid()) {
            uint64_t s = extract_dim_u64(schema, it2.key(), DIM_STATE);
            uint64_t p = extract_dim_u64(schema, it2.key(), DIM_PRODUCT);
            uint64_t g = (s << 8) | p;
            double v = static_cast<double>(extract_dim_u64(schema, it2.key(), DIM_PRICE));
            auto m = means.find(g);
            if (m != means.end() && v > m->second) above++;
            it2.next();
        }
        double ms = t.elapsed_ms();
        results.push_back({ "Q19_corr_multi_dim", ms, above, 0.0, 0,
                            std::to_string(stats.size()) + " partitions" });
        std::cerr << "[hp] Q19 corr_multi_dim " << ms << "ms  above=" << above
                  << " partitions=" << stats.size() << "\n";
    }

    // =========================================================================
    //  Q20: Semi-join / YoY growth —
    //        COUNT(product) WHERE SUM(price)[yr=2022] > SUM(price)[yr=2021]
    // =========================================================================
    {
        bench::Timer t;
        auto ps_a = build_eq_predicate(DIM_YEAR, YEAR_2021);
        auto ps_b = build_eq_predicate(DIM_YEAR, YEAR_2022);
        std::unordered_map<uint64_t, double> sum_a, sum_b;
        tree.predicate_search_cb(ps_a, [&](Key k, uint64_t){
            sum_a[extract_dim_u64(schema, k, DIM_PRODUCT)]
                += static_cast<double>(extract_dim_u64(schema, k, DIM_PRICE));
        });
        tree.predicate_search_cb(ps_b, [&](Key k, uint64_t){
            sum_b[extract_dim_u64(schema, k, DIM_PRODUCT)]
                += static_cast<double>(extract_dim_u64(schema, k, DIM_PRICE));
        });
        uint64_t growers = 0; double delta = 0; uint64_t chk = 0;
        for (auto& [p, sb] : sum_b) {
            double sa = sum_a.count(p) ? sum_a[p] : 0.0;
            if (sb > sa) {
                growers++;
                delta += sb - sa;
                chk ^= (p * 0x9E3779B97F4A7C15ULL) ^ static_cast<uint64_t>(sb - sa);
            }
        }
        double ms = t.elapsed_ms();
        results.push_back({ "Q20_yoy_semijoin", ms, growers, delta, chk,
                            std::to_string(sum_a.size()) + "a/"
                            + std::to_string(sum_b.size()) + "b" });
        std::cerr << "[hp] Q20 yoy_semijoin " << ms << "ms  growers="
                  << growers << " delta=" << delta << "\n";
    }

    bench::write_results("hp", dist_name, results, output_path);
    std::cerr << "[hp] wrote " << output_path << "\n";
    return 0;
}
