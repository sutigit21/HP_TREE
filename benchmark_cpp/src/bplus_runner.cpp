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

    bench::write_results("bplus", dist_name, results, output_path);
    std::cerr << "[bplus] wrote " << output_path << "\n";
    return 0;
}
