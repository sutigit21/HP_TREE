#!/usr/bin/env python3
"""
Orchestrator for the HP-Tree (C++) vs tlx::btree_multimap (C++) benchmark.

Invokes the two C++ runners on each dataset, parses their JSON output,
verifies correctness, and prints a comparison table.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DISTRIBUTIONS = ["uniform", "clustered", "skewed", "sequential"]

QUERIES = [
    ("Q1_bulk_load",       "Bulk Load"),
    ("Q2_point_lookup",    "Point Lookup"),
    ("Q3_narrow_range",    "Narrow Range"),
    ("Q4_wide_range",      "Wide Range"),
    ("Q5_dim_filter",      "Dim Filter"),
    ("Q6_multi_dim_filter","Multi-Dim Filter"),
    ("Q7_range_agg",       "Range Aggregation"),
    ("Q8_full_scan",       "Full Scan"),
    ("Q9_single_inserts",  "Single Inserts"),
    ("Q10_deletes",        "Deletes"),
    ("Q11_hypercube",      "Hypercube 3-dim"),
    ("Q12_group_by",       "Group-By Agg"),
    ("Q13_correlated",     "Correlated Sub"),
    ("Q14_moving_window",  "Moving Window"),
    ("Q15_adhoc_drill",    "Ad-Hoc Drill"),
]


def run_runner(runner, dataset, spec, output, dist):
    cmd = [str(runner),
           "--dataset", str(dataset),
           "--spec",    str(spec),
           "--output",  str(output),
           "--dist",    dist]
    print(f"  > {runner.name} {dist}")
    t0 = time.time()
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        print(f"ERROR: {runner.name} failed with code {res.returncode}")
        sys.exit(1)
    print(f"    ({time.time() - t0:.1f}s)")
    with open(output) as f:
        return json.load(f)


def fmt_ms(ms):
    if ms is None:    return "     --"
    if ms < 1:        return f"{ms:8.3f}"
    if ms < 100:      return f"{ms:8.2f}"
    return f"{ms:8.1f}"


def fmt_speedup(bp, hp):
    if bp is None or hp is None or bp == 0 or hp == 0:
        return "     --"
    r = bp / hp
    if r >= 1.0:
        return f"HP {r:6.1f}x"
    return f"B+ {1/r:6.1f}x"


def results_to_map(rec):
    return {q["name"]: q for q in rec["queries"]}


def correctness_match(bp, hp):
    # Q1: both should report same count.
    # Q9/Q10: deterministic counts.
    # Others: compare count + checksum + sum where present.
    if bp["count"] != hp["count"]:
        return False
    # Allow small floating point diff for sums.
    if abs(bp["sum"] - hp["sum"]) > max(1.0, abs(bp["sum"]) * 1e-6):
        return False
    # Checksums only match when both runners deterministically visit the same
    # records in the same logical set. Insert/delete counts already covered.
    # For queries that XOR over a set, XOR is order-independent.
    # Checksums XOR over the matched set and are order-independent; a mismatch
    # always means a real correctness divergence.
    if bp["checksum"] != hp["checksum"]:
        return False
    return True


def print_table(all_results):
    print()
    print("=" * 96)
    print(f"{'HP-Tree (C++) vs tlx::btree_multimap (C++)':^96}")
    print("=" * 96)

    for dist in DISTRIBUTIONS:
        if dist not in all_results: continue
        r = all_results[dist]
        bp = results_to_map(r["bplus"])
        hp = results_to_map(r["hp"])

        print(f"\n{dist.upper()} DISTRIBUTION")
        print(f"  {'Query':<24}{'tlx::btree':>12}{'HP-Tree':>12}{'Speedup':>14}{'Correct':>10}")
        print("  " + "-" * 72)
        for qname, label in QUERIES:
            bq = bp.get(qname)
            hq = hp.get(qname)
            if bq is None or hq is None:
                print(f"  {label:<24}{'--':>12}{'--':>12}")
                continue
            bp_ms, hp_ms = bq["elapsed_ms"], hq["elapsed_ms"]
            ok = correctness_match(bq, hq)
            print(f"  {label:<24}{fmt_ms(bp_ms):>12}{fmt_ms(hp_ms):>12}"
                  f"{fmt_speedup(bp_ms, hp_ms):>14}"
                  f"{'YES' if ok else 'NO':>10}")

    # Cross-distribution summary: wins/losses
    print("\n" + "=" * 96)
    print(f"{'CROSS-DISTRIBUTION SUMMARY':^96}")
    print("=" * 96)
    print(f"  {'Query':<24}", end="")
    for dist in DISTRIBUTIONS:
        if dist in all_results:
            print(f"{dist:>14}", end="")
    print()
    print("  " + "-" * (24 + 14 * len([d for d in DISTRIBUTIONS if d in all_results])))

    hp_wins = 0; bp_wins = 0; total = 0; correct = 0
    for qname, label in QUERIES:
        print(f"  {label:<24}", end="")
        for dist in DISTRIBUTIONS:
            if dist not in all_results: continue
            bp = results_to_map(all_results[dist]["bplus"]).get(qname)
            hp = results_to_map(all_results[dist]["hp"]).get(qname)
            if bp is None or hp is None:
                print(f"{'--':>14}", end="")
                continue
            total += 1
            if correctness_match(bp, hp): correct += 1
            if bp["elapsed_ms"] == 0 or hp["elapsed_ms"] == 0:
                print(f"{'--':>14}", end=""); continue
            r = bp["elapsed_ms"] / hp["elapsed_ms"]
            if r >= 1.0:
                hp_wins += 1
                print(f"{f'HP {r:5.1f}x':>14}", end="")
            else:
                bp_wins += 1
                print(f"{f'B+ {1/r:5.1f}x':>14}", end="")
        print()

    print("\n  Wins:         HP-Tree={:>3}   tlx::btree={:>3}   of {}".format(
        hp_wins, bp_wins, hp_wins + bp_wins))
    print(f"  Correctness:  {correct}/{total} cells matched")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", type=Path,
                    default=Path(__file__).parent.parent / "build")
    ap.add_argument("--datasets-dir", type=Path,
                    default=Path(__file__).parent.parent / "datasets")
    ap.add_argument("--results-dir", type=Path,
                    default=Path(__file__).parent.parent / "results")
    ap.add_argument("--distributions", nargs="+", default=DISTRIBUTIONS)
    args = ap.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    hp_runner    = args.build_dir / "hp_runner"
    bplus_runner = args.build_dir / "bplus_runner"
    for b in (hp_runner, bplus_runner):
        if not b.exists():
            print(f"ERROR: runner binary not found: {b}")
            print("       Build with: cmake -B build && cmake --build build")
            sys.exit(1)

    all_results = {}
    for dist in args.distributions:
        dataset = args.datasets_dir / f"{dist}.bin"
        if not dataset.exists():
            print(f"WARN: missing {dataset}, skipping")
            continue
        # Prefer per-distribution spec; fall back to shared spec for old layouts.
        spec = args.datasets_dir / f"query_spec_{dist}.json"
        if not spec.exists():
            spec = args.datasets_dir / "query_spec.json"
        if not spec.exists():
            print(f"ERROR: query spec not found for {dist}: {spec}")
            print("       Regenerate with: python generate_datasets.py")
            sys.exit(1)
        print(f"\n--- {dist} --- (spec: {spec.name})")
        bp_json = run_runner(bplus_runner, dataset, spec,
                             args.results_dir / f"bplus_{dist}.json", dist)
        hp_json = run_runner(hp_runner, dataset, spec,
                             args.results_dir / f"hp_{dist}.json", dist)
        all_results[dist] = {"bplus": bp_json, "hp": hp_json}

    print_table(all_results)


if __name__ == "__main__":
    main()
