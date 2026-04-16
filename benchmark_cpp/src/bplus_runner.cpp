#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common.hpp"
#include "hp_tree_common.hpp"

#include <tlx/container/btree_multimap.hpp>

namespace {

using Key   = hptree::CompositeKey;
using Value = uint64_t;

struct U128Less {
    bool operator()(Key a, Key b) const noexcept { return a < b; }
};

using BTree = tlx::btree_multimap<Key, Value, U128Less>;

inline uint64_t extract_dim_u64(const hptree::CompositeKeySchema& schema,
                                Key key, size_t dim) {
    uint8_t  off  = schema.offset_of(dim);
    uint64_t mask = schema.mask_of(dim);
    return static_cast<uint64_t>((key >> off) & mask);
}

}  // namespace

int main(int argc, char** argv) {
    std::string dataset_path = bench::arg_get(argc, argv, "--dataset");
    std::string spec_path    = bench::arg_get(argc, argv, "--spec");
    std::string output_path  = bench::arg_get(argc, argv, "--output");
    std::string dist_name    = bench::arg_get(argc, argv, "--dist", "unknown");

    if (dataset_path.empty() || spec_path.empty() || output_path.empty()) {
        std::cerr << "Usage: bplus_runner --dataset <path> --spec <path> "
                     "--output <path> [--dist <name>]\n";
        return 1;
    }

    std::cerr << "[bplus] loading dataset " << dataset_path << "...\n";
    auto dataset = bench::load_dataset(dataset_path);
    std::cerr << "[bplus] N=" << dataset.size() << "\n";

    auto qs     = bench::load_query_spec(spec_path);
    auto schema = hptree::make_default_sales_schema();
    std::vector<bench::QueryResult> results;

    // =========================================================================
    //  Q1: Bulk Load  (sort + tlx::btree_multimap::bulk_load)
    // =========================================================================
    BTree tree;
    {
        std::vector<std::pair<Key, Value>> pairs;
        pairs.reserve(dataset.size());
        for (auto& r : dataset) pairs.emplace_back(r.key, r.value);

        bench::Timer t;
        std::sort(pairs.begin(), pairs.end(),
                  [](const auto& a, const auto& b){ return a.first < b.first; });
        tree.bulk_load(pairs.begin(), pairs.end());
        double ms = t.elapsed_ms();

        bench::QueryResult qr;
        qr.name = "Q1_bulk_load";
        qr.elapsed_ms = ms;
        qr.result_count = tree.size();
        results.push_back(qr);
        std::cerr << "[bplus] Q1 bulk_load " << ms << "ms  size=" << tree.size() << "\n";
    }

    // =========================================================================
    //  Q2: Point Lookup  (2000 keys)
    // =========================================================================
    {
        bench::Timer t;
        uint64_t hits = 0, chk = 0;
        for (auto k : qs.point_lookup_keys) {
            auto range = tree.equal_range(k);
            for (auto it = range.first; it != range.second; ++it) {
                hits++;
                chk ^= bench::key_checksum_u128(it->first);
            }
        }
        double ms = t.elapsed_ms();
        bench::QueryResult qr;
        qr.name = "Q2_point_lookup";
        qr.elapsed_ms = ms;
        qr.result_count = hits;
        qr.key_checksum = chk;
        results.push_back(qr);
        std::cerr << "[bplus] Q2 point_lookup " << ms << "ms  hits=" << hits << "\n";
    }

    auto range_scan = [&](Key lo, Key hi, uint64_t& count,
                          uint64_t& chk, double& sum_price) {
        count = 0; chk = 0; sum_price = 0;
        auto it = tree.lower_bound(lo);
        auto end = tree.upper_bound(hi);
        for (; it != end; ++it) {
            count++;
            chk ^= bench::key_checksum_u128(it->first);
            sum_price += static_cast<double>(extract_dim_u64(schema, it->first, 5));
        }
    };

    // =========================================================================
    //  Q3: Narrow Range
    // =========================================================================
    {
        bench::Timer t;
        uint64_t cnt = 0, chk = 0; double sum = 0;
        range_scan(qs.narrow_range.lo, qs.narrow_range.hi, cnt, chk, sum);
        double ms = t.elapsed_ms();
        results.push_back({ "Q3_narrow_range", ms, cnt, sum, chk, "" });
        std::cerr << "[bplus] Q3 narrow_range " << ms << "ms  n=" << cnt << "\n";
    }

    // =========================================================================
    //  Q4: Wide Range
    // =========================================================================
    {
        bench::Timer t;
        uint64_t cnt = 0, chk = 0; double sum = 0;
        range_scan(qs.wide_range.lo, qs.wide_range.hi, cnt, chk, sum);
        double ms = t.elapsed_ms();
        results.push_back({ "Q4_wide_range", ms, cnt, sum, chk, "" });
        std::cerr << "[bplus] Q4 wide_range " << ms << "ms  n=" << cnt << "\n";
    }

    // =========================================================================
    //  Q5: Dim Filter (full scan; B+ has no dim metadata)
    // =========================================================================
    {
        bench::Timer t;
        uint64_t cnt = 0, chk = 0;
        for (auto it = tree.begin(); it != tree.end(); ++it) {
            if (extract_dim_u64(schema, it->first, qs.dim_filter.dim)
                    == qs.dim_filter.value) {
                cnt++;
                chk ^= bench::key_checksum_u128(it->first);
            }
        }
        double ms = t.elapsed_ms();
        results.push_back({ "Q5_dim_filter", ms, cnt, 0.0, chk, "" });
        std::cerr << "[bplus] Q5 dim_filter " << ms << "ms  n=" << cnt << "\n";
    }

    // =========================================================================
    //  Q6: Multi-Dim Filter
    // =========================================================================
    {
        bench::Timer t;
        uint64_t cnt = 0, chk = 0;
        for (auto it = tree.begin(); it != tree.end(); ++it) {
            bool ok = true;
            for (auto& f : qs.multi_dim_filter) {
                if (extract_dim_u64(schema, it->first, f.dim) != f.value) {
                    ok = false; break;
                }
            }
            if (ok) { cnt++; chk ^= bench::key_checksum_u128(it->first); }
        }
        double ms = t.elapsed_ms();
        results.push_back({ "Q6_multi_dim_filter", ms, cnt, 0.0, chk, "" });
        std::cerr << "[bplus] Q6 multi_dim_filter " << ms << "ms  n=" << cnt << "\n";
    }

    // =========================================================================
    //  Q7: Range Aggregation SUM(dim)
    // =========================================================================
    {
        bench::Timer t;
        uint64_t cnt = 0;
        double sum = 0.0;
        auto it  = tree.lower_bound(qs.range_agg.lo);
        auto end = tree.upper_bound(qs.range_agg.hi);
        for (; it != end; ++it) {
            sum += static_cast<double>(extract_dim_u64(schema, it->first,
                                                       qs.range_agg_dim));
            cnt++;
        }
        double ms = t.elapsed_ms();
        results.push_back({ "Q7_range_agg", ms, cnt, sum, 0, "" });
        std::cerr << "[bplus] Q7 range_agg " << ms << "ms  n=" << cnt
                  << " sum=" << sum << "\n";
    }

    // =========================================================================
    //  Q8: Full Scan
    // =========================================================================
    {
        bench::Timer t;
        uint64_t cnt = 0, chk = 0;
        for (auto it = tree.begin(); it != tree.end(); ++it) {
            cnt++;
            chk ^= bench::key_checksum_u128(it->first);
        }
        double ms = t.elapsed_ms();
        results.push_back({ "Q8_full_scan", ms, cnt, 0.0, chk, "" });
        std::cerr << "[bplus] Q8 full_scan " << ms << "ms  n=" << cnt << "\n";
    }

    // =========================================================================
    //  Q9: Single Inserts
    // =========================================================================
    {
        bench::Timer t;
        for (size_t i = 0; i < qs.insert_keys.size(); ++i) {
            tree.insert(std::make_pair(qs.insert_keys[i], qs.insert_values[i]));
        }
        double ms = t.elapsed_ms();
        results.push_back({ "Q9_single_inserts", ms,
                            (uint64_t)qs.insert_keys.size(), 0.0, 0, "" });
        std::cerr << "[bplus] Q9 single_inserts " << ms << "ms  n="
                  << qs.insert_keys.size() << "\n";
    }

    // =========================================================================
    //  Q10: Deletes
    // =========================================================================
    {
        bench::Timer t;
        uint64_t removed = 0;
        for (auto k : qs.delete_keys) {
            auto it = tree.find(k);
            if (it != tree.end()) { tree.erase(it); removed++; }
        }
        double ms = t.elapsed_ms();
        results.push_back({ "Q10_deletes", ms, removed, 0.0, 0, "" });
        std::cerr << "[bplus] Q10 deletes " << ms << "ms  removed=" << removed << "\n";
    }

    // =========================================================================
    //  Q11: Hypercube (3-dim bounding box) — full scan with 3 range preds
    // =========================================================================
    {
        bench::Timer t;
        uint64_t cnt = 0, chk = 0;
        for (auto it = tree.begin(); it != tree.end(); ++it) {
            bool ok = true;
            for (auto& dr : qs.hypercube) {
                uint64_t v = extract_dim_u64(schema, it->first, dr.dim);
                if (v < dr.lo || v > dr.hi) { ok = false; break; }
            }
            if (ok) { cnt++; chk ^= bench::key_checksum_u128(it->first); }
        }
        double ms = t.elapsed_ms();
        results.push_back({ "Q11_hypercube", ms, cnt, 0.0, chk, "" });
        std::cerr << "[bplus] Q11 hypercube " << ms << "ms  n=" << cnt << "\n";
    }

    // =========================================================================
    //  Q12: Group-By Agg  — filter + group-by + sum
    // =========================================================================
    {
        bench::Timer t;
        std::unordered_map<uint64_t, std::pair<uint64_t,double>> groups;
        uint64_t cnt = 0;
        for (auto it = tree.begin(); it != tree.end(); ++it) {
            if (extract_dim_u64(schema, it->first, qs.groupby.filter_dim)
                    != qs.groupby.filter_val) continue;
            uint64_t g = extract_dim_u64(schema, it->first, qs.groupby.group_dim);
            double v = static_cast<double>(
                extract_dim_u64(schema, it->first, qs.groupby.agg_dim));
            auto& e = groups[g];
            e.first++;
            e.second += v;
            cnt++;
        }
        double ms = t.elapsed_ms();
        double total_sum = 0;
        uint64_t chk = 0;
        for (auto& [g, v] : groups) {
            total_sum += v.second;
            chk ^= (g * 0x9E3779B97F4A7C15ULL)
                 ^ static_cast<uint64_t>(v.second);
        }
        results.push_back({ "Q12_group_by", ms, cnt, total_sum, chk,
                            std::to_string(groups.size()) + " groups" });
        std::cerr << "[bplus] Q12 group_by " << ms << "ms  groups="
                  << groups.size() << " sum=" << total_sum << "\n";
    }

    // =========================================================================
    //  Q13: Correlated Subquery — per-group avg, then count records above avg
    // =========================================================================
    {
        bench::Timer t;
        std::unordered_map<uint64_t, std::pair<uint64_t,double>> stats;
        for (auto it = tree.begin(); it != tree.end(); ++it) {
            uint64_t g = extract_dim_u64(schema, it->first, qs.correlated_group_dim);
            double v = static_cast<double>(
                extract_dim_u64(schema, it->first, qs.correlated_agg_dim));
            auto& e = stats[g];
            e.first++;
            e.second += v;
        }
        std::unordered_map<uint64_t, double> means;
        means.reserve(stats.size());
        for (auto& [g, s] : stats)
            means[g] = s.first > 0 ? s.second / static_cast<double>(s.first) : 0.0;

        uint64_t above = 0;
        for (auto it = tree.begin(); it != tree.end(); ++it) {
            uint64_t g = extract_dim_u64(schema, it->first, qs.correlated_group_dim);
            double v = static_cast<double>(
                extract_dim_u64(schema, it->first, qs.correlated_agg_dim));
            auto m = means.find(g);
            if (m != means.end() && v > m->second) above++;
        }
        double ms = t.elapsed_ms();
        results.push_back({ "Q13_correlated", ms, above, 0.0, 0,
                            std::to_string(stats.size()) + " groups" });
        std::cerr << "[bplus] Q13 correlated " << ms << "ms  above=" << above << "\n";
    }

    // =========================================================================
    //  Q14: Moving Window (12x range + SUM)
    // =========================================================================
    {
        bench::Timer t;
        double total_sum = 0;
        uint64_t total_cnt = 0;
        uint64_t chk = 0;
        for (auto& w : qs.moving_windows) {
            double win_sum = 0;
            uint64_t win_cnt = 0;
            auto it  = tree.lower_bound(w.lo);
            auto end = tree.upper_bound(w.hi);
            for (; it != end; ++it) {
                win_sum += static_cast<double>(
                    extract_dim_u64(schema, it->first, qs.moving_agg_dim));
                win_cnt++;
            }
            total_sum += win_sum;
            total_cnt += win_cnt;
            chk ^= static_cast<uint64_t>(win_sum);
        }
        double ms = t.elapsed_ms();
        results.push_back({ "Q14_moving_window", ms, total_cnt, total_sum, chk, "" });
        std::cerr << "[bplus] Q14 moving_window " << ms << "ms  total_n="
                  << total_cnt << "\n";
    }

    // =========================================================================
    //  Q15: Ad-Hoc Drill (30x full scan + multi-dim filter)
    // =========================================================================
    {
        bench::Timer t;
        uint64_t total_hits = 0, chk = 0;
        for (auto& drill : qs.adhoc_drills) {
            for (auto it = tree.begin(); it != tree.end(); ++it) {
                bool ok = true;
                for (auto& f : drill) {
                    if (extract_dim_u64(schema, it->first, f.dim) != f.value) {
                        ok = false; break;
                    }
                }
                if (ok) { total_hits++; chk ^= bench::key_checksum_u128(it->first); }
            }
        }
        double ms = t.elapsed_ms();
        results.push_back({ "Q15_adhoc_drill", ms, total_hits, 0.0, chk,
                            std::to_string(qs.adhoc_drills.size()) + " drills" });
        std::cerr << "[bplus] Q15 adhoc_drill " << ms << "ms  total_hits="
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
        bench::Timer t;
        std::unordered_map<uint64_t, std::pair<uint64_t,double>> groups;
        for (auto it = tree.begin(); it != tree.end(); ++it) {
            if (extract_dim_u64(schema, it->first, DIM_YEAR) != YEAR_2022) continue;
            uint64_t g = extract_dim_u64(schema, it->first, DIM_PRODUCT);
            double v = static_cast<double>(extract_dim_u64(schema, it->first, DIM_PRICE));
            auto& e = groups[g];
            e.first++; e.second += v;
        }
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
        std::cerr << "[bplus] Q16 topk_groups " << ms << "ms  top" << take
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
        for (auto it = tree.begin(); it != tree.end(); ++it) {
            uint64_t g = extract_dim_u64(schema, it->first, DIM_STATE);
            double v = static_cast<double>(extract_dim_u64(schema, it->first, DIM_PRICE));
            sums[g] += v;
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
        std::cerr << "[bplus] Q17 having_clause " << ms << "ms  kept="
                  << kept << "/" << sums.size() << "\n";
    }

    // =========================================================================
    //  Q18: Year/Month Rollup —
    //        year IN [20,23] GROUP BY (year,month) SUM(price) COUNT(*)
    //        + compute per-year totals (rollup)
    // =========================================================================
    {
        bench::Timer t;
        std::unordered_map<uint64_t, std::pair<uint64_t,double>> ym;
        auto it  = tree.lower_bound(qs.wide_range.lo);
        auto end = tree.upper_bound(qs.wide_range.hi);
        for (; it != end; ++it) {
            uint64_t y = extract_dim_u64(schema, it->first, DIM_YEAR);
            uint64_t m = extract_dim_u64(schema, it->first, DIM_MONTH);
            double v = static_cast<double>(extract_dim_u64(schema, it->first, DIM_PRICE));
            auto& e = ym[(y << 4) | m];
            e.first++; e.second += v;
        }
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
        std::cerr << "[bplus] Q18 ym_rollup " << ms << "ms  ym_groups="
                  << ym.size() << " yr_groups=" << yr_totals.size() << "\n";
    }

    // =========================================================================
    //  Q19: Correlated multi-dim partition —
    //        COUNT(*) WHERE price > AVG(price) PARTITION BY (state,product)
    // =========================================================================
    {
        bench::Timer t;
        std::unordered_map<uint64_t, std::pair<uint64_t,double>> stats;
        for (auto it = tree.begin(); it != tree.end(); ++it) {
            uint64_t s = extract_dim_u64(schema, it->first, DIM_STATE);
            uint64_t p = extract_dim_u64(schema, it->first, DIM_PRODUCT);
            uint64_t g = (s << 8) | p;
            double v = static_cast<double>(extract_dim_u64(schema, it->first, DIM_PRICE));
            auto& e = stats[g];
            e.first++; e.second += v;
        }
        std::unordered_map<uint64_t, double> means;
        means.reserve(stats.size());
        for (auto& [g, s] : stats)
            means[g] = s.first > 0 ? s.second / static_cast<double>(s.first) : 0.0;

        uint64_t above = 0;
        for (auto it = tree.begin(); it != tree.end(); ++it) {
            uint64_t s = extract_dim_u64(schema, it->first, DIM_STATE);
            uint64_t p = extract_dim_u64(schema, it->first, DIM_PRODUCT);
            uint64_t g = (s << 8) | p;
            double v = static_cast<double>(extract_dim_u64(schema, it->first, DIM_PRICE));
            auto m = means.find(g);
            if (m != means.end() && v > m->second) above++;
        }
        double ms = t.elapsed_ms();
        results.push_back({ "Q19_corr_multi_dim", ms, above, 0.0, 0,
                            std::to_string(stats.size()) + " partitions" });
        std::cerr << "[bplus] Q19 corr_multi_dim " << ms << "ms  above=" << above
                  << " partitions=" << stats.size() << "\n";
    }

    // =========================================================================
    //  Q20: Semi-join / YoY growth —
    //        COUNT(product) WHERE SUM(price)[yr=2022] > SUM(price)[yr=2021]
    // =========================================================================
    {
        bench::Timer t;
        std::unordered_map<uint64_t, double> sum_a, sum_b;
        for (auto it = tree.begin(); it != tree.end(); ++it) {
            uint64_t y = extract_dim_u64(schema, it->first, DIM_YEAR);
            if (y != YEAR_2021 && y != YEAR_2022) continue;
            uint64_t p = extract_dim_u64(schema, it->first, DIM_PRODUCT);
            double v = static_cast<double>(extract_dim_u64(schema, it->first, DIM_PRICE));
            if (y == YEAR_2021) sum_a[p] += v; else sum_b[p] += v;
        }
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
        std::cerr << "[bplus] Q20 yoy_semijoin " << ms << "ms  growers="
                  << growers << " delta=" << delta << "\n";
    }

    // =========================================================================
    //  Q21: Complex OR-Bitmap -- full scan + OR logic
    // =========================================================================
    {
        constexpr uint64_t YEAR_2020 = 20, YEAR_2023 = 23;
        constexpr uint64_t STATE_CA  = 1,  STATE_NY  = 9;
        constexpr uint64_t PROD_LAPTOP = 5;
        bench::Timer t;
        uint64_t cnt = 0, chk = 0; double sum = 0.0;
        for (auto it = tree.begin(); it != tree.end(); ++it) {
            uint64_t y = extract_dim_u64(schema, it->first, DIM_YEAR);
            uint64_t s = extract_dim_u64(schema, it->first, DIM_STATE);
            uint64_t p = extract_dim_u64(schema, it->first, DIM_PRODUCT);
            bool hit = (y == YEAR_2022 && s == STATE_CA)
                    || (y == YEAR_2020 && s == STATE_NY)
                    || (y == YEAR_2023 && p == PROD_LAPTOP);
            if (hit) {
                ++cnt;
                chk ^= bench::key_checksum_u128(it->first);
                sum += static_cast<double>(extract_dim_u64(schema, it->first, DIM_PRICE));
            }
        }
        double ms = t.elapsed_ms();
        results.push_back({ "Q21_or_bitmap", ms, cnt, sum, chk, "3 groups" });
        std::cerr << "[bplus] Q21 or_bitmap " << ms << "ms  n=" << cnt << "\n";
    }

    // =========================================================================
    //  Q22: Windowed Top-3 per Month -- single full scan into 12-bucket map,
    //        then partial_sort for top-3 per month.
    // =========================================================================
    {
        bench::Timer t;
        std::unordered_map<uint64_t, double> buckets;
        uint64_t total_rows = 0;
        for (auto it = tree.begin(); it != tree.end(); ++it) {
            uint64_t y = extract_dim_u64(schema, it->first, DIM_YEAR);
            if (y != YEAR_2022) continue;
            uint64_t m = extract_dim_u64(schema, it->first, DIM_MONTH);
            uint64_t s = extract_dim_u64(schema, it->first, DIM_STATE);
            buckets[(m << 8) | s] +=
                static_cast<double>(extract_dim_u64(schema, it->first, DIM_PRICE));
            ++total_rows;
        }
        double total_top3_sum = 0.0; uint64_t chk = 0;
        for (uint64_t m = 1; m <= 12; ++m) {
            std::vector<std::pair<uint64_t,double>> ranked;
            for (auto& [mk, v] : buckets) {
                if ((mk >> 8) == m) ranked.emplace_back(mk & 0xFF, v);
            }
            size_t take = std::min<size_t>(3, ranked.size());
            if (take > 0) {
                std::partial_sort(ranked.begin(), ranked.begin() + take, ranked.end(),
                    [](const auto& a, const auto& b){ return a.second > b.second; });
                for (size_t i = 0; i < take; ++i) {
                    total_top3_sum += ranked[i].second;
                    chk ^= (m * 0x100000001B3ULL)
                         ^ (ranked[i].first * 0x9E3779B97F4A7C15ULL)
                         ^ static_cast<uint64_t>(ranked[i].second);
                }
            }
        }
        double ms = t.elapsed_ms();
        results.push_back({ "Q22_window_top3_month", ms, total_rows,
                            total_top3_sum, chk, "12 months" });
        std::cerr << "[bplus] Q22 window_top3_month " << ms << "ms  top3_sum="
                  << total_top3_sum << "\n";
    }

    // =========================================================================
    //  Q23: Multi-Stage CTE -- two full scans: build avg21, then count above.
    // =========================================================================
    {
        bench::Timer t;
        std::unordered_map<uint64_t, std::pair<uint64_t,double>> s21;
        for (auto it = tree.begin(); it != tree.end(); ++it) {
            if (extract_dim_u64(schema, it->first, DIM_YEAR) != YEAR_2021) continue;
            uint64_t s = extract_dim_u64(schema, it->first, DIM_STATE);
            uint64_t p = extract_dim_u64(schema, it->first, DIM_PRODUCT);
            uint64_t g = (s << 8) | p;
            double v = static_cast<double>(extract_dim_u64(schema, it->first, DIM_PRICE));
            auto& e = s21[g];
            e.first++; e.second += v;
        }
        std::unordered_map<uint64_t, double> avg21;
        avg21.reserve(s21.size());
        for (auto& [g, rec] : s21)
            avg21[g] = rec.first ? rec.second / static_cast<double>(rec.first) : 0.0;

        uint64_t above = 0; double above_sum = 0; uint64_t chk = 0;
        for (auto it = tree.begin(); it != tree.end(); ++it) {
            if (extract_dim_u64(schema, it->first, DIM_YEAR) != YEAR_2022) continue;
            uint64_t s = extract_dim_u64(schema, it->first, DIM_STATE);
            uint64_t p = extract_dim_u64(schema, it->first, DIM_PRODUCT);
            uint64_t g = (s << 8) | p;
            double v = static_cast<double>(extract_dim_u64(schema, it->first, DIM_PRICE));
            auto ait = avg21.find(g);
            if (ait != avg21.end() && v > ait->second) {
                above++; above_sum += v;
                chk ^= bench::key_checksum_u128(it->first);
            }
        }
        double ms = t.elapsed_ms();
        results.push_back({ "Q23_cte_correlated", ms, above, above_sum, chk,
                            std::to_string(avg21.size()) + " partitions" });
        std::cerr << "[bplus] Q23 cte_correlated " << ms << "ms  above="
                  << above << "\n";
    }

    // =========================================================================
    //  Q24: YoY Self-Join -- single full scan into (year,state,product) buckets.
    // =========================================================================
    {
        bench::Timer t;
        std::unordered_map<uint64_t, double> sum_21, sum_22;
        for (auto it = tree.begin(); it != tree.end(); ++it) {
            uint64_t y = extract_dim_u64(schema, it->first, DIM_YEAR);
            if (y != YEAR_2021 && y != YEAR_2022) continue;
            uint64_t g = (extract_dim_u64(schema, it->first, DIM_STATE) << 8)
                       |  extract_dim_u64(schema, it->first, DIM_PRODUCT);
            double v = static_cast<double>(extract_dim_u64(schema, it->first, DIM_PRICE));
            if (y == YEAR_2021) sum_21[g] += v;
            else                sum_22[g] += v;
        }
        uint64_t growers = 0; double delta_sum = 0; uint64_t chk = 0;
        for (auto& [g, s22] : sum_22) {
            auto it = sum_21.find(g);
            double s21 = (it != sum_21.end()) ? it->second : 0.0;
            if (s22 > s21 * 1.2) {
                growers++;
                delta_sum += (s22 - s21);
                chk ^= (g * 0x9E3779B97F4A7C15ULL)
                     ^ static_cast<uint64_t>(s22 - s21);
            }
        }
        double ms = t.elapsed_ms();
        results.push_back({ "Q24_yoy_selfjoin", ms, growers, delta_sum, chk,
                            std::to_string(sum_21.size()) + "/"
                            + std::to_string(sum_22.size()) + " partitions" });
        std::cerr << "[bplus] Q24 yoy_selfjoin " << ms << "ms  growers="
                  << growers << "\n";
    }

    // =========================================================================
    //  Q25: Dense Hyperbox -- full scan with 4 per-record BETWEEN predicates.
    // =========================================================================
    {
        constexpr uint64_t Y_LO = 21, Y_HI = 23;
        constexpr uint64_t S_LO = 1,  S_HI = 5;
        constexpr uint64_t P_LO = 5,  P_HI = 7;
        constexpr uint64_t PR_LO = 30000, PR_HI = 180000;
        bench::Timer t;
        uint64_t cnt = 0, chk = 0; double sum = 0.0;
        for (auto it = tree.begin(); it != tree.end(); ++it) {
            uint64_t y = extract_dim_u64(schema, it->first, DIM_YEAR);
            if (y < Y_LO || y > Y_HI) continue;
            uint64_t s = extract_dim_u64(schema, it->first, DIM_STATE);
            if (s < S_LO || s > S_HI) continue;
            uint64_t p = extract_dim_u64(schema, it->first, DIM_PRODUCT);
            if (p < P_LO || p > P_HI) continue;
            uint64_t pr = extract_dim_u64(schema, it->first, DIM_PRICE);
            if (pr < PR_LO || pr > PR_HI) continue;
            ++cnt;
            chk ^= bench::key_checksum_u128(it->first);
            sum += static_cast<double>(pr);
        }
        double ms = t.elapsed_ms();
        results.push_back({ "Q25_dense_hyperbox", ms, cnt, sum, chk, "4-dim box" });
        std::cerr << "[bplus] Q25 dense_hyperbox " << ms << "ms  n=" << cnt << "\n";
    }

    bench::write_results("bplus", dist_name, results, output_path);
    std::cerr << "[bplus] wrote " << output_path << "\n";
    return 0;
}
