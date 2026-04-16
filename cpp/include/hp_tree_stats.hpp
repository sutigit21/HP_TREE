#pragma once

#include "hp_tree_node.hpp"

namespace hptree {

struct HistogramBucket {
    CompositeKey low;
    CompositeKey high;
    uint64_t     count       = 0;
    uint64_t     distinct    = 0;
    double       density     = 0.0;
};

struct DimensionHistogram {
    size_t dim_idx = 0;
    std::string dim_name;
    std::vector<HistogramBucket> buckets;
    uint64_t total_count    = 0;
    uint64_t null_count     = 0;
    uint64_t distinct_count = 0;
    double   min_val        = 0.0;
    double   max_val        = 0.0;
    double   avg_val        = 0.0;

    double estimate_selectivity(uint64_t low, uint64_t high) const {
        if (total_count == 0 || buckets.empty()) return 1.0;
        uint64_t matching = 0;
        for (auto& b : buckets) {
            uint64_t b_lo = static_cast<uint64_t>(b.low);
            uint64_t b_hi = static_cast<uint64_t>(b.high);
            if (high < b_lo || low > b_hi) continue;
            uint64_t overlap_lo = std::max(low, b_lo);
            uint64_t overlap_hi = std::min(high, b_hi);
            double bucket_range = static_cast<double>(b_hi - b_lo + 1);
            double overlap_range = static_cast<double>(overlap_hi - overlap_lo + 1);
            double fraction = (bucket_range > 0) ? overlap_range / bucket_range : 1.0;
            matching += static_cast<uint64_t>(b.count * fraction);
        }
        return static_cast<double>(matching) / total_count;
    }
};

struct TreeStatistics {
    uint64_t total_records      = 0;
    uint64_t total_leaves       = 0;
    uint64_t total_internal     = 0;
    uint64_t total_homogeneous  = 0;
    uint32_t tree_depth         = 0;
    size_t   memory_bytes       = 0;
    double   avg_leaf_fill      = 0.0;
    double   avg_beta           = 0.0;
    double   min_beta           = std::numeric_limits<double>::max();
    double   max_beta           = 0.0;

    struct BetaDistribution {
        double p1  = 0, p5  = 0, p10 = 0, p25 = 0;
        double p50 = 0, p75 = 0, p90 = 0, p95 = 0, p99 = 0;
    } beta_dist;

    BetaComputer::Thresholds current_thresholds;
    std::vector<DimensionHistogram> dim_histograms;
    uint64_t delta_buffer_size   = 0;
    uint64_t delta_total_flushes = 0;
    uint64_t total_splits        = 0;
    uint64_t total_merges        = 0;
    uint64_t total_rebalances    = 0;
};

struct QueryCost {
    double estimated_io_cost     = 0.0;
    double estimated_cpu_cost    = 0.0;
    double total_cost            = 0.0;
    uint64_t estimated_rows      = 0;
    uint64_t estimated_partitions= 0;
    bool   recommend_seq_scan    = false;
    std::string explanation;
};

class StatisticsCollector {
    const CompositeKeySchema* schema_ = nullptr;

public:
    explicit StatisticsCollector(const CompositeKeySchema* schema = nullptr)
        : schema_(schema) {}

    void set_schema(const CompositeKeySchema* schema) { schema_ = schema; }

    TreeStatistics collect(HPNode* root, size_t max_leaf_size) const {
        TreeStatistics stats;
        if (!root) return stats;

        std::vector<double> beta_values;
        std::vector<double> fill_ratios;

        collect_recursive(root, stats, beta_values, fill_ratios, 0);

        stats.tree_depth = compute_depth(root);
        if (root) stats.memory_bytes = root->memory_usage();

        if (!fill_ratios.empty()) {
            stats.avg_leaf_fill = std::accumulate(
                fill_ratios.begin(), fill_ratios.end(), 0.0) / fill_ratios.size();
        }

        if (!beta_values.empty()) {
            stats.avg_beta = std::accumulate(
                beta_values.begin(), beta_values.end(), 0.0) / beta_values.size();
            std::sort(beta_values.begin(), beta_values.end());
            auto pct = [&](double p) -> double {
                size_t idx = static_cast<size_t>(p * (beta_values.size() - 1));
                return beta_values[std::min(idx, beta_values.size() - 1)];
            };
            stats.beta_dist.p1  = pct(0.01);
            stats.beta_dist.p5  = pct(0.05);
            stats.beta_dist.p10 = pct(0.10);
            stats.beta_dist.p25 = pct(0.25);
            stats.beta_dist.p50 = pct(0.50);
            stats.beta_dist.p75 = pct(0.75);
            stats.beta_dist.p90 = pct(0.90);
            stats.beta_dist.p95 = pct(0.95);
            stats.beta_dist.p99 = pct(0.99);
        }

        if (schema_) build_histograms(root, stats);

        return stats;
    }

    DimensionHistogram build_histogram_for_dim(
            HPNode* root, size_t dim_idx, size_t num_buckets = HISTOGRAM_BUCKETS) const {
        DimensionHistogram hist;
        if (!schema_ || dim_idx >= schema_->dimensions.size()) return hist;

        hist.dim_idx = dim_idx;
        hist.dim_name = schema_->dimensions[dim_idx].name;

        std::vector<uint64_t> values;
        collect_dim_values(root, dim_idx, values);
        if (values.empty()) return hist;

        std::sort(values.begin(), values.end());
        hist.total_count = values.size();
        hist.min_val = static_cast<double>(values.front());
        hist.max_val = static_cast<double>(values.back());

        double sum = 0;
        for (auto v : values) sum += static_cast<double>(v);
        hist.avg_val = sum / values.size();

        {
            uint64_t prev = values[0];
            uint64_t distinct = 1;
            for (size_t i = 1; i < values.size(); ++i) {
                if (values[i] != prev) { distinct++; prev = values[i]; }
            }
            hist.distinct_count = distinct;
        }

        size_t per_bucket = std::max<size_t>(1, values.size() / num_buckets);
        for (size_t i = 0; i < values.size(); i += per_bucket) {
            size_t end = std::min(i + per_bucket, values.size());
            HistogramBucket b;
            b.low = static_cast<CompositeKey>(values[i]);
            b.high = static_cast<CompositeKey>(values[end - 1]);
            b.count = end - i;

            uint64_t prev_v = values[i]; uint64_t dc = 1;
            for (size_t j = i + 1; j < end; ++j) {
                if (values[j] != prev_v) { dc++; prev_v = values[j]; }
            }
            b.distinct = dc;

            double range = static_cast<double>(b.high) - static_cast<double>(b.low);
            b.density = (range > 0) ? b.count / range : b.count;

            hist.buckets.push_back(b);
        }

        return hist;
    }

    QueryCost estimate_cost(HPNode* root, const PredicateSet& preds,
                            const TreeStatistics& stats) const {
        QueryCost cost;
        if (!root || !schema_ || stats.total_records == 0) return cost;

        double combined_selectivity = 1.0;
        for (auto& p : preds.predicates) {
            if (p.dim_idx < stats.dim_histograms.size()) {
                auto& hist = stats.dim_histograms[p.dim_idx];
                double sel = 1.0;
                switch (p.op) {
                case PredicateOp::EQ:
                    sel = (hist.distinct_count > 0)
                        ? 1.0 / hist.distinct_count : 1.0;
                    break;
                case PredicateOp::BETWEEN:
                    sel = hist.estimate_selectivity(
                        static_cast<uint64_t>(p.value),
                        static_cast<uint64_t>(p.value_high));
                    break;
                case PredicateOp::LT:
                case PredicateOp::LTE:
                    sel = hist.estimate_selectivity(0,
                        static_cast<uint64_t>(p.value));
                    break;
                case PredicateOp::GT:
                case PredicateOp::GTE:
                    sel = hist.estimate_selectivity(
                        static_cast<uint64_t>(p.value),
                        static_cast<uint64_t>(hist.max_val));
                    break;
                case PredicateOp::IN:
                    sel = static_cast<double>(p.in_values.size())
                        / std::max<uint64_t>(hist.distinct_count, 1);
                    break;
                case PredicateOp::IS_NULL:
                    sel = (hist.total_count > 0)
                        ? static_cast<double>(hist.null_count) / hist.total_count
                        : 0.0;
                    break;
                default:
                    sel = 0.5;
                    break;
                }
                combined_selectivity *= std::max(sel, 0.0001);
            }
        }

        cost.estimated_rows = static_cast<uint64_t>(
            stats.total_records * combined_selectivity);
        cost.estimated_partitions = static_cast<uint64_t>(
            stats.total_leaves * std::min(combined_selectivity * 3.0, 1.0));

        double page_cost = 1.0;
        double cpu_tuple_cost = 0.01;
        double cpu_index_cost = 0.005;

        cost.estimated_io_cost =
            std::log2(std::max<double>(stats.total_leaves, 1.0)) * page_cost
            + cost.estimated_partitions * page_cost;
        cost.estimated_cpu_cost =
            std::log2(std::max<double>(stats.total_leaves, 1.0)) * cpu_index_cost
            + cost.estimated_rows * cpu_tuple_cost;

        cost.total_cost = cost.estimated_io_cost + cost.estimated_cpu_cost;

        double seq_scan_cost = stats.total_records * cpu_tuple_cost
                             + (stats.total_records / 100.0) * page_cost;

        cost.recommend_seq_scan = (combined_selectivity > 0.3)
                               || (cost.total_cost > seq_scan_cost);

        std::ostringstream oss;
        oss << "Selectivity: " << combined_selectivity
            << ", Est.Rows: " << cost.estimated_rows
            << ", IndexCost: " << cost.total_cost
            << ", SeqScanCost: " << seq_scan_cost
            << (cost.recommend_seq_scan ? " -> SeqScan" : " -> IndexScan");
        cost.explanation = oss.str();

        return cost;
    }

private:
    void collect_recursive(HPNode* node, TreeStatistics& stats,
                           std::vector<double>& betas,
                           std::vector<double>& fills,
                           uint32_t depth) const {
        if (!node) return;

        if (node->is_leaf()) {
            auto* leaf = static_cast<LeafNode*>(node);
            stats.total_leaves++;
            stats.total_records += leaf->meta.record_count;

            if (leaf->is_homogeneous()) stats.total_homogeneous++;

            double beta = leaf->meta.beta_value;
            if (std::isfinite(beta)) {
                betas.push_back(beta);
                if (beta < stats.min_beta) stats.min_beta = beta;
                if (beta > stats.max_beta) stats.max_beta = beta;
            }

            double fill = (DEFAULT_MAX_LEAF_SIZE > 0)
                ? static_cast<double>(leaf->records.size()) / DEFAULT_MAX_LEAF_SIZE
                : 0.0;
            fills.push_back(fill);
        } else {
            auto* internal = static_cast<InternalNode*>(node);
            stats.total_internal++;
            for (auto& c : internal->children) {
                collect_recursive(c.get(), stats, betas, fills, depth + 1);
            }
        }
    }

    uint32_t compute_depth(HPNode* node) const {
        if (!node) return 0;
        if (node->is_leaf()) return 1;
        auto* internal = static_cast<InternalNode*>(node);
        uint32_t max_d = 0;
        for (auto& c : internal->children) {
            max_d = std::max(max_d, compute_depth(c.get()));
        }
        return max_d + 1;
    }

    void build_histograms(HPNode* root, TreeStatistics& stats) const {
        if (!schema_) return;
        stats.dim_histograms.resize(schema_->dimensions.size());
        for (size_t d = 0; d < schema_->dimensions.size(); ++d) {
            stats.dim_histograms[d] = build_histogram_for_dim(root, d);
        }
    }

    void collect_dim_values(HPNode* node, size_t dim_idx,
                            std::vector<uint64_t>& values) const {
        if (!node || !schema_) return;
        if (node->is_leaf()) {
            auto* leaf = static_cast<LeafNode*>(node);
            CompositeKeyEncoder encoder(*schema_);
            for (auto& r : leaf->records) {
                if (!r.tombstone) {
                    values.push_back(encoder.extract_dim(r.key, dim_idx));
                }
            }
        } else {
            auto* internal = static_cast<InternalNode*>(node);
            for (auto& c : internal->children)
                collect_dim_values(c.get(), dim_idx, values);
        }
    }
};

}  // namespace hptree
