#!/usr/bin/env python3
"""
ML-based Optimizer for HP-Tree Beta Threshold Power (k)
=======================================================
Finds optimal k in the stopping criterion: beta < 1/N^k

Score = Σ ln(bp_ms / hp_ms) across all query types.
  - Each HP-Tree win contributes a positive term (larger margin → larger term)
  - Each HP-Tree loss contributes a negative term (penalizes regressions)
  - Maximizing score = maximizing both win count AND win margins

Model: GradientBoostingRegressor (sklearn) with fallback to numpy polyfit
Features: [log(N), k, log(beta_total), coeff_of_variation, skewness]
"""

import math
import time
import random
import statistics
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hp_vs_bplus_benchmark import (
    BPlusTree, HPTree,
    compute_beta, encode_key, extract_dim,
    BITS, STATES, PRODUCTS, STATE_ENC, PROD_ENC,
    gen_uniform, gen_clustered, gen_skewed, gen_sequential,
    bench,
)

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

ORDER = 50
BRANCHING = 20

QUERY_TYPES = [
    "Bulk Load",
    "Point Lookup",
    "Narrow Range",
    "Wide Range",
    "Dim Filter",
    "Multi-Dim Filter",
    "Aggregation",
    "Full Scan",
    "Single Inserts",
    "Deletes",
]

QUERY_WEIGHTS = {
    "Point Lookup":    1.5,
    "Narrow Range":    1.2,
    "Wide Range":      1.2,
    "Aggregation":     1.3,
    "Single Inserts":  1.3,
    "Dim Filter":      1.0,
    "Multi-Dim Filter":1.0,
    "Full Scan":       0.8,
    "Bulk Load":       0.7,
    "Deletes":         0.8,
}


def compute_score(bp_results, hp_results):
    score = 0.0
    wins = 0
    for qt in QUERY_TYPES:
        bp_ms = bp_results[qt]["avg_ms"]
        hp_ms = hp_results[qt]["avg_ms"]
        speedup = bp_ms / max(hp_ms, 1e-9)
        w = QUERY_WEIGHTS.get(qt, 1.0)
        score += w * math.log(max(speedup, 1e-9))
        if speedup > 1.0:
            wins += 1
    return score, wins


def extract_features(pairs, k):
    keys = [p[0] for p in pairs]
    n = len(keys)
    mean_k = sum(keys) / n
    var_k = sum((x - mean_k) ** 2 for x in keys) / n
    std_k = math.sqrt(var_k)
    cv = std_k / mean_k if mean_k != 0 else 0

    if std_k > 0 and n > 2:
        skew = sum((x - mean_k) ** 3 for x in keys) / (n * std_k ** 3)
    else:
        skew = 0.0

    beta_total = compute_beta(min(keys), max(keys))

    return [
        math.log(n),
        k,
        math.log(beta_total + 1e-15),
        cv,
        skew,
    ]


def run_bp_queries(bp_tree, pairs, keys):
    results = {}
    rng = random.Random(99)
    sample = [rng.choice(keys) for _ in range(500)]

    def bp_bulk():
        t = BPlusTree(order=ORDER)
        t.bulk_load(list(pairs))
        return t
    results["Bulk Load"] = bench(bp_bulk, iterations=1)

    def point():
        found = 0
        for k in sample:
            if bp_tree.search(k) is not None:
                found += 1
        return found
    results["Point Lookup"] = bench(point, iterations=3)

    lo_n = encode_key(2022, 6, 1, "AZ", "Chair", 0.0, 0.0)
    hi_n = encode_key(2022, 6, 28, "WA", "Webcam", 5000.0, 10.0)
    results["Narrow Range"] = bench(lambda: len(bp_tree.range_search(lo_n, hi_n)), iterations=3)

    lo_w = encode_key(2020, 1, 1, "AZ", "Chair", 0.0, 0.0)
    hi_w = encode_key(2023, 12, 28, "WA", "Webcam", 5000.0, 10.0)
    results["Wide Range"] = bench(lambda: len(bp_tree.range_search(lo_w, hi_w)), iterations=3)

    year_enc = 2022 - 2000
    results["Dim Filter"] = bench(lambda: len(bp_tree.dim_filter(0, year_enc)), iterations=2)

    state_enc = STATE_ENC["CA"]
    results["Multi-Dim Filter"] = bench(
        lambda: len(bp_tree.multi_dim_filter([(0, year_enc), (3, state_enc)])), iterations=2)

    lo_a = encode_key(2021, 1, 1, "AZ", "Chair", 0.0, 0.0)
    hi_a = encode_key(2023, 12, 28, "WA", "Webcam", 5000.0, 10.0)
    results["Aggregation"] = bench(lambda: bp_tree.aggregate_range(lo_a, hi_a, 5), iterations=3)

    results["Full Scan"] = bench(lambda: len(bp_tree.scan_all()), iterations=2)

    rng2 = random.Random(55)
    extra = []
    for _ in range(500):
        year = rng2.randint(2018, 2025)
        month = rng2.randint(1, 12)
        day = rng2.randint(1, 28)
        state = rng2.choice(STATES)
        product = rng2.choice(PRODUCTS)
        price = round(rng2.uniform(5, 3000), 2)
        version = round(rng2.uniform(0.5, 10), 2)
        extra.append((encode_key(year, month, day, state, product, price, version), 0))

    def bp_insert():
        t = BPlusTree(order=ORDER)
        t.bulk_load(list(pairs[:5000]))
        t0 = time.perf_counter_ns()
        for ek, ev in extra:
            t.insert(ek, ev)
        return (time.perf_counter_ns() - t0) / 1e6
    r = bench(bp_insert, iterations=1)
    r["avg_ms"] = r["result"]
    results["Single Inserts"] = r

    bp_del = BPlusTree(order=ORDER)
    bp_del.bulk_load(list(pairs))
    del_sample = random.Random(77).sample(keys, min(300, len(keys)))
    def bp_delete():
        c = 0
        for dk in del_sample:
            if bp_del.delete(dk):
                c += 1
        return c
    results["Deletes"] = bench(bp_delete, iterations=1)

    return results


def run_hp_queries(hp_tree, pairs, keys):
    results = {}
    rng = random.Random(99)
    sample = [rng.choice(keys) for _ in range(500)]
    k_power = hp_tree.split_power

    def hp_bulk():
        t = HPTree(max_leaf=ORDER, branching=BRANCHING, split_power=k_power)
        t.bulk_load(list(pairs))
        return t
    results["Bulk Load"] = bench(hp_bulk, iterations=1)

    def point():
        found = 0
        for k in sample:
            if hp_tree.search(k) is not None:
                found += 1
        return found
    results["Point Lookup"] = bench(point, iterations=3)

    lo_n = encode_key(2022, 6, 1, "AZ", "Chair", 0.0, 0.0)
    hi_n = encode_key(2022, 6, 28, "WA", "Webcam", 5000.0, 10.0)
    results["Narrow Range"] = bench(lambda: len(hp_tree.range_search(lo_n, hi_n)), iterations=3)

    lo_w = encode_key(2020, 1, 1, "AZ", "Chair", 0.0, 0.0)
    hi_w = encode_key(2023, 12, 28, "WA", "Webcam", 5000.0, 10.0)
    results["Wide Range"] = bench(lambda: len(hp_tree.range_search(lo_w, hi_w)), iterations=3)

    year_enc = 2022 - 2000
    results["Dim Filter"] = bench(lambda: len(hp_tree.dim_filter(0, year_enc)), iterations=2)

    state_enc = STATE_ENC["CA"]
    results["Multi-Dim Filter"] = bench(
        lambda: len(hp_tree.multi_dim_filter([(0, year_enc), (3, state_enc)])), iterations=2)

    lo_a = encode_key(2021, 1, 1, "AZ", "Chair", 0.0, 0.0)
    hi_a = encode_key(2023, 12, 28, "WA", "Webcam", 5000.0, 10.0)
    results["Aggregation"] = bench(lambda: hp_tree.aggregate_range(lo_a, hi_a, 5), iterations=3)

    results["Full Scan"] = bench(lambda: len(hp_tree.scan_all()), iterations=2)

    rng2 = random.Random(55)
    extra = []
    for _ in range(500):
        year = rng2.randint(2018, 2025)
        month = rng2.randint(1, 12)
        day = rng2.randint(1, 28)
        state = rng2.choice(STATES)
        product = rng2.choice(PRODUCTS)
        price = round(rng2.uniform(5, 3000), 2)
        version = round(rng2.uniform(0.5, 10), 2)
        extra.append((encode_key(year, month, day, state, product, price, version), 0))

    def hp_insert():
        t = HPTree(max_leaf=ORDER, branching=BRANCHING, split_power=k_power)
        t.bulk_load(list(pairs[:5000]))
        t0 = time.perf_counter_ns()
        for ek, ev in extra:
            t.insert(ek, ev)
        t._flush_delta()
        return (time.perf_counter_ns() - t0) / 1e6
    r = bench(hp_insert, iterations=1)
    r["avg_ms"] = r["result"]
    results["Single Inserts"] = r

    hp_del = HPTree(max_leaf=ORDER, branching=BRANCHING, split_power=k_power)
    hp_del.bulk_load(list(pairs))
    del_sample = random.Random(77).sample(keys, min(300, len(keys)))
    def hp_delete():
        c = 0
        for dk in del_sample:
            if hp_del.delete(dk):
                c += 1
        return c
    results["Deletes"] = bench(hp_delete, iterations=1)

    return results


def collect_training_data(N_values, k_values, distributions):
    X = []
    y = []
    metadata = []

    total = len(N_values) * len(distributions) * len(k_values)
    done = 0

    for N in N_values:
        for dist_name, gen_fn in distributions:
            pairs = gen_fn(N)
            keys = [p[0] for p in pairs]

            print(f"\n  [{dist_name}, N={N:,}] Building B+ Tree...", end="", flush=True)
            bp_tree = BPlusTree(order=ORDER)
            bp_tree.bulk_load(list(pairs))
            bp_results = run_bp_queries(bp_tree, pairs, keys)
            print(" cached.", flush=True)

            for k in k_values:
                done += 1
                print(f"    k={k:.2f} ({done}/{total})...", end="", flush=True)

                hp_tree = HPTree(max_leaf=ORDER, branching=BRANCHING, split_power=k)
                hp_tree.bulk_load(list(pairs))
                hp_results = run_hp_queries(hp_tree, pairs, keys)

                score, wins = compute_score(bp_results, hp_results)
                feats = extract_features(pairs, k)

                X.append(feats)
                y.append(score)

                hp_stats = hp_tree.stats()
                metadata.append({
                    "N": N,
                    "dist": dist_name,
                    "k": k,
                    "score": score,
                    "wins": wins,
                    "leaves": hp_stats["leaves"],
                })
                print(f" score={score:.2f}, wins={wins}/10, leaves={hp_stats['leaves']}")

    return np.array(X), np.array(y), metadata


class PolyFallbackModel:
    def __init__(self, degree=3):
        self.degree = degree
        self.coeffs = None
        self.feature_means = None
        self.feature_stds = None

    def fit(self, X, y):
        self.feature_means = X.mean(axis=0)
        self.feature_stds = X.std(axis=0) + 1e-10
        X_norm = (X - self.feature_means) / self.feature_stds
        n_features = X_norm.shape[1]
        polys = [np.ones(len(X))]
        for i in range(n_features):
            for d in range(1, self.degree + 1):
                polys.append(X_norm[:, i] ** d)
        for i in range(n_features):
            for j in range(i + 1, n_features):
                polys.append(X_norm[:, i] * X_norm[:, j])
        A = np.column_stack(polys)
        self.coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

    def predict(self, X):
        X_norm = (X - self.feature_means) / self.feature_stds
        n_features = X_norm.shape[1]
        polys = [np.ones(len(X))]
        for i in range(n_features):
            for d in range(1, self.degree + 1):
                polys.append(X_norm[:, i] ** d)
        for i in range(n_features):
            for j in range(i + 1, n_features):
                polys.append(X_norm[:, i] * X_norm[:, j])
        A = np.column_stack(polys)
        return A @ self.coeffs


def train_model(X, y):
    if HAS_SKLEARN:
        print("\n  Training GradientBoostingRegressor...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_samples_leaf=2, random_state=42)
        model.fit(X_scaled, y)

        preds = model.predict(X_scaled)
        residuals = y - preds
        rmse = math.sqrt(np.mean(residuals ** 2))
        r2 = 1.0 - np.sum(residuals ** 2) / np.sum((y - np.mean(y)) ** 2)

        feature_names = ["log(N)", "k", "log(beta)", "cv", "skew"]
        importances = model.feature_importances_
        print(f"  RMSE: {rmse:.3f}  |  R²: {r2:.3f}")
        print(f"  Feature Importances:")
        for fn, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
            print(f"    {fn:<12} {imp:.3f}  {'█' * int(imp * 40)}")

        return model, scaler
    else:
        print("\n  Training polynomial fallback model (sklearn not found)...")
        model = PolyFallbackModel(degree=3)
        model.fit(X, y)

        preds = model.predict(X)
        residuals = y - preds
        rmse = math.sqrt(np.mean(residuals ** 2))
        r2 = 1.0 - np.sum(residuals ** 2) / np.sum((y - np.mean(y)) ** 2)
        print(f"  RMSE: {rmse:.3f}  |  R²: {r2:.3f}")

        return model, None


def predict_optimal_k(model, scaler, pairs, k_range=np.arange(1.0, 3.51, 0.05)):
    best_k = 2.0
    best_score = -float('inf')
    scores = []

    for k in k_range:
        feats = np.array([extract_features(pairs, k)])
        if scaler is not None:
            feats = scaler.transform(feats)
        pred = model.predict(feats)[0]
        scores.append((k, pred))
        if pred > best_score:
            best_score = pred
            best_k = k

    return best_k, best_score, scores


def print_score_curve(scores, optimal_k):
    print(f"\n  Score vs k (optimal k = {optimal_k:.2f}):")
    print(f"  {'k':>6} {'Score':>10}  Curve")
    print(f"  {'-'*50}")
    max_s = max(s for _, s in scores)
    min_s = min(s for _, s in scores)
    rng = max_s - min_s if max_s != min_s else 1.0
    for k, s in scores:
        bar_len = int(30 * (s - min_s) / rng)
        marker = " ◀ OPTIMAL" if abs(k - optimal_k) < 0.02 else ""
        print(f"  {k:>6.2f} {s:>10.3f}  {'█' * bar_len}{marker}")


if __name__ == "__main__":
    print("=" * 78)
    print("  HP-Tree Beta Power (k) Optimizer — ML-Based")
    print("=" * 78)
    print(f"  Score = Σ wᵢ·ln(bp_msᵢ / hp_msᵢ) across 10 query types")
    print(f"  Higher score = more HP-Tree wins with bigger margins")
    print(f"  Model: {'GradientBoostingRegressor' if HAS_SKLEARN else 'Polynomial Regression (fallback)'}")
    print()

    N_VALUES = [50000, 100000, 250000]
    K_VALUES = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    DISTRIBUTIONS = [
        ("Uniform", gen_uniform),
        ("Clustered", gen_clustered),
        ("Skewed", gen_skewed),
        ("Sequential", gen_sequential),
    ]

    print(f"  Training grid: {len(N_VALUES)} sizes × {len(DISTRIBUTIONS)} distributions × {len(K_VALUES)} k values")
    print(f"  = {len(N_VALUES) * len(DISTRIBUTIONS) * len(K_VALUES)} benchmark runs")
    print()

    X, y, metadata = collect_training_data(N_VALUES, K_VALUES, DISTRIBUTIONS)

    model, scaler = train_model(X, y)

    print(f"\n\n{'='*78}")
    print("  TRAINING DATA SUMMARY")
    print(f"{'='*78}")
    print(f"\n  {'Distribution':<14} {'N':>8} {'k':>6} {'Score':>8} {'Wins':>6} {'Leaves':>8}")
    print(f"  {'-'*58}")
    for m in metadata:
        print(f"  {m['dist']:<14} {m['N']:>8,} {m['k']:>6.2f} {m['score']:>8.2f} {m['wins']:>5}/10 {m['leaves']:>8,}")

    print(f"\n\n{'='*78}")
    print("  OPTIMAL k PREDICTIONS")
    print(f"{'='*78}")

    test_configs = [
        ("Uniform 50K", gen_uniform, 50000),
        ("Uniform 100K", gen_uniform, 100000),
        ("Uniform 500K", gen_uniform, 500000),
        ("Clustered 50K", gen_clustered, 50000),
        ("Clustered 100K", gen_clustered, 100000),
        ("Clustered 500K", gen_clustered, 500000),
        ("Skewed 50K", gen_skewed, 50000),
        ("Skewed 100K", gen_skewed, 100000),
        ("Skewed 500K", gen_skewed, 500000),
        ("Sequential 50K", gen_sequential, 50000),
        ("Sequential 100K", gen_sequential, 100000),
        ("Sequential 500K", gen_sequential, 500000),
    ]

    print(f"\n  {'Dataset':<22} {'Optimal k':>10} {'Pred Score':>12}")
    print(f"  {'-'*48}")
    for name, gen_fn, n in test_configs:
        pairs = gen_fn(n)
        opt_k, opt_score, scores = predict_optimal_k(model, scaler, pairs)
        print(f"  {name:<22} {opt_k:>10.2f} {opt_score:>12.2f}")

    print(f"\n\n{'='*78}")
    print("  SCORE CURVES (per distribution, N=100K)")
    print(f"{'='*78}")
    for dist_name, gen_fn in DISTRIBUTIONS:
        pairs = gen_fn(100000)
        opt_k, opt_score, scores = predict_optimal_k(model, scaler, pairs)
        print(f"\n  --- {dist_name} (optimal k={opt_k:.2f}) ---")
        print_score_curve(scores, opt_k)

    print()
