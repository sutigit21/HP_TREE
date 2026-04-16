#!/usr/bin/env python3
"""
Dataset + query-spec generator for the HP-Tree vs tlx::btree C++ benchmark.

Emits:
  datasets/<distribution>.bin    (binary key/value records)
  datasets/query_spec.json       (identical queries for both runners)

Schema (matches cpp/include/hp_tree_common.hpp make_default_sales_schema):
  year    8  bits  linear  base=2000  scale=1
  month   4  bits  linear  base=1     scale=1
  day     5  bits  linear  base=1     scale=1
  state   5  bits  dict    [AZ,CA,FL,GA,IL,MA,MI,NC,NJ,NY,OH,PA,TX,VA,WA]
  product 5  bits  dict    [Chair,Desk,Default,Headset,Keyboard,Laptop,Monitor,Mouse,Webcam]
  price  19  bits  linear  base=0     scale=100
  version 10 bits  linear  base=0     scale=100
"""

import argparse
import json
import os
import random
import struct
from pathlib import Path

# ---------------------------------------------------------------------------
#  Schema constants (must mirror C++ make_default_sales_schema)
# ---------------------------------------------------------------------------
STATES   = ["AZ","CA","FL","GA","IL","MA","MI","NC","NJ","NY","OH","PA","TX","VA","WA"]
PRODUCTS = ["Chair","Desk","Default","Headset","Keyboard","Laptop","Monitor","Mouse","Webcam"]

BITS   = [8, 4, 5, 5, 5, 19, 10]
BASES  = [2000, 1, 1, 0, 0, 0, 0]
SCALES = [1, 1, 1, 1, 1, 100, 100]

TOTAL_BITS = sum(BITS)  # 56

OFFSETS = []
acc = 0
for i in range(len(BITS) - 1, -1, -1):
    OFFSETS.insert(0, acc)
    acc += BITS[i]
# OFFSETS[i] = bit offset (from LSB) of dimension i

DIM_YEAR, DIM_MONTH, DIM_DAY, DIM_STATE, DIM_PRODUCT, DIM_PRICE, DIM_VERSION = range(7)


def encode_key(year, month, day, state_idx, product_idx, price_x100, version_x100):
    vals = [
        year - BASES[DIM_YEAR],
        month - BASES[DIM_MONTH],
        day - BASES[DIM_DAY],
        state_idx,
        product_idx,
        price_x100,
        version_x100,
    ]
    key = 0
    for i, v in enumerate(vals):
        max_val = (1 << BITS[i]) - 2   # reserve top value for null sentinel
        v = max(0, min(v, max_val))
        key |= (v & ((1 << BITS[i]) - 1)) << OFFSETS[i]
    return key


def dim_value_mask(dim):
    return (1 << BITS[dim]) - 1


def u128_to_pair(k):
    lo = k & 0xFFFFFFFFFFFFFFFF
    hi = (k >> 64) & 0xFFFFFFFFFFFFFFFF
    return [lo, hi]


# ---------------------------------------------------------------------------
#  Distributions
# ---------------------------------------------------------------------------
def gen_uniform(n, rng):
    for _ in range(n):
        yield (
            rng.randint(2020, 2024),
            rng.randint(1, 12),
            rng.randint(1, 28),
            rng.randint(0, len(STATES) - 1),
            rng.randint(0, len(PRODUCTS) - 1),
            rng.randint(5_00, 2000_00),        # $5.00..$2000.00
            rng.randint(100, 500),             # version 1.00..5.00
        )


def gen_clustered(n, rng):
    # 3 centres: (CA,Laptop), (NY,Mouse), (TX,Keyboard)
    centres = [
        ("CA", "Laptop",   (2020, 2022), (1200_00, 400_00)),   # mean, stddev_x100
        ("NY", "Mouse",    (2021, 2023), (35_00,   10_00)),
        ("TX", "Keyboard", (2022, 2024), (80_00,   25_00)),
    ]
    for _ in range(n):
        s_name, p_name, yrng, (mean, stdv) = rng.choice(centres)
        price = max(1_00, int(rng.gauss(mean, stdv)))
        yield (
            rng.randint(*yrng),
            rng.randint(1, 12),
            rng.randint(1, 28),
            STATES.index(s_name),
            PRODUCTS.index(p_name),
            price,
            rng.randint(100, 500),
        )


def gen_skewed(n, rng):
    # 80% in narrow band (CA, Laptop, June 2022); 20% uniform
    hot = int(n * 0.8)
    for i in range(n):
        if i < hot:
            yield (
                2022,
                6,
                rng.randint(1, 28),
                STATES.index("CA"),
                PRODUCTS.index("Laptop"),
                rng.randint(900_00, 1500_00),
                rng.randint(100, 500),
            )
        else:
            yield (
                rng.randint(2020, 2024),
                rng.randint(1, 12),
                rng.randint(1, 28),
                rng.randint(0, len(STATES) - 1),
                rng.randint(0, len(PRODUCTS) - 1),
                rng.randint(5_00, 2000_00),
                rng.randint(100, 500),
            )


def gen_sequential(n, rng):
    # Monotonically increasing across all dimensions (time-series pattern).
    # Step through (year, month, day) sequentially; state/product/price rotate.
    yr, mo, da = 2020, 1, 1
    for i in range(n):
        state_idx   = (i // 100) % len(STATES)
        product_idx = (i //  50) % len(PRODUCTS)
        price = 500_00 + (i % 100_000)                  # gentle price drift
        ver   = 100 + (i % 400)
        yield (yr, mo, da, state_idx, product_idx, price, ver)

        da += 1
        if da > 28:
            da = 1; mo += 1
            if mo > 12:
                mo = 1; yr += 1
                if yr > 2254: yr = 2020


DISTRIBUTIONS = {
    "uniform":    gen_uniform,
    "clustered":  gen_clustered,
    "skewed":     gen_skewed,
    "sequential": gen_sequential,
}


# ---------------------------------------------------------------------------
#  Binary write
# ---------------------------------------------------------------------------
def write_dataset(path, records):
    with open(path, "wb") as f:
        f.write(b"HPDS0001")
        f.write(struct.pack("<Q", len(records)))
        for k, v in records:
            lo = k & 0xFFFFFFFFFFFFFFFF
            hi = (k >> 64) & 0xFFFFFFFFFFFFFFFF
            f.write(struct.pack("<QQQ", lo, hi, v))


# ---------------------------------------------------------------------------
#  Query spec construction
# ---------------------------------------------------------------------------
def build_query_spec(n, rng, sample_keys, extra_insert_keys):
    # Year=2022 is dimension 0 value 22 (encoded = 2022-2000)
    YEAR_2022 = 22

    # Narrow range: year=22, month=6  (June 2022)
    narrow_lo = (YEAR_2022 << OFFSETS[DIM_YEAR]) | (6 << OFFSETS[DIM_MONTH])
    narrow_hi = narrow_lo | ((1 << OFFSETS[DIM_MONTH]) - 1)

    # Wide range: year in [20, 23]  (2020..2023)
    wide_lo = 20 << OFFSETS[DIM_YEAR]
    wide_hi = (23 << OFFSETS[DIM_YEAR]) | ((1 << OFFSETS[DIM_YEAR]) - 1)

    # Range aggregation: same as wide range, aggregate price (dim 5)
    range_agg = {
        "lo": u128_to_pair(wide_lo),
        "hi": u128_to_pair(wide_hi),
        "agg_dim": DIM_PRICE,
    }

    # Dim filter: year = 2022
    dim_filter = {"dim": DIM_YEAR, "value": YEAR_2022}

    # Multi-dim filter: year=2022 AND state=CA
    multi_dim_filter = [
        {"dim": DIM_YEAR,  "value": YEAR_2022},
        {"dim": DIM_STATE, "value": STATES.index("CA")},
    ]

    # Hypercube: year in [20,23], state in [CA..GA idx 1..3], product = Laptop
    hypercube = [
        {"dim": DIM_YEAR,    "lo": 20, "hi": 23},
        {"dim": DIM_STATE,   "lo": STATES.index("CA"), "hi": STATES.index("GA")},
        {"dim": DIM_PRODUCT, "lo": PRODUCTS.index("Laptop"),
                             "hi": PRODUCTS.index("Laptop")},
    ]

    # Group-by: year=2022 GROUP BY state SUM(price)
    groupby = {
        "filter_dim": DIM_YEAR,
        "filter_val": YEAR_2022,
        "group_dim":  DIM_STATE,
        "agg_dim":    DIM_PRICE,
    }

    # Correlated: per-product avg price, count records above avg
    correlated = {
        "group_dim": DIM_PRODUCT,
        "agg_dim":   DIM_PRICE,
    }

    # Moving window: 12 monthly windows within year 2022
    moving_windows = []
    for m in range(1, 13):
        lo = (YEAR_2022 << OFFSETS[DIM_YEAR]) | (m << OFFSETS[DIM_MONTH])
        hi = lo | ((1 << OFFSETS[DIM_MONTH]) - 1)
        moving_windows.append({
            "lo": u128_to_pair(lo),
            "hi": u128_to_pair(hi),
        })

    # Ad-hoc drills: 30 random (year, state) pairs
    adhoc = []
    for _ in range(30):
        adhoc.append([
            {"dim": DIM_YEAR,  "value": rng.randint(20, 24)},
            {"dim": DIM_STATE, "value": rng.randint(0, len(STATES) - 1)},
        ])

    # Point lookup: 2000 keys sampled from the dataset
    pl_keys = rng.sample(sample_keys, min(2000, len(sample_keys)))

    # Insert keys: 1000 new synthetic records
    insert_keys = [
        encode_key(
            rng.randint(2020, 2024), rng.randint(1,12), rng.randint(1,28),
            rng.randint(0,len(STATES)-1), rng.randint(0,len(PRODUCTS)-1),
            rng.randint(1_00, 3000_00), rng.randint(100, 600))
        for _ in range(1000)
    ]
    insert_values = list(range(n, n + 1000))

    # Delete keys: 500 existing keys
    delete_keys = rng.sample(sample_keys, min(500, len(sample_keys)))

    return {
        "n_records": n,
        "point_lookup_keys": [u128_to_pair(k) for k in pl_keys],
        "narrow_range": {"lo": u128_to_pair(narrow_lo),
                         "hi": u128_to_pair(narrow_hi)},
        "wide_range":   {"lo": u128_to_pair(wide_lo),
                         "hi": u128_to_pair(wide_hi)},
        "range_agg":    range_agg,
        "dim_filter":   dim_filter,
        "multi_dim_filter": multi_dim_filter,
        "insert_keys":   [u128_to_pair(k) for k in insert_keys],
        "insert_values": insert_values,
        "delete_keys":   [u128_to_pair(k) for k in delete_keys],
        "hypercube":     hypercube,
        "groupby":       groupby,
        "correlated":    correlated,
        "moving_windows": moving_windows,
        "moving":         {"agg_dim": DIM_PRICE},
        "adhoc_drills":   adhoc,
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000,
                    help="records per distribution")
    ap.add_argument("--outdir", type=Path, default=Path(__file__).parent.parent / "datasets",
                    help="output directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--distributions", nargs="+", default=list(DISTRIBUTIONS.keys()))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    dist_order = ["uniform", "clustered", "skewed", "sequential"]
    for dist in args.distributions:
        if dist not in DISTRIBUTIONS:
            print(f"skipping unknown distribution {dist}")
            continue
        print(f"[gen] {dist}  N={args.n:,}")
        rng = random.Random(args.seed + dist_order.index(dist) * 1000)
        records = []
        for i, tpl in enumerate(DISTRIBUTIONS[dist](args.n, rng)):
            records.append((encode_key(*tpl), i))
        path = args.outdir / f"{dist}.bin"
        write_dataset(path, records)
        print(f"       wrote {path}  ({path.stat().st_size / 1e6:.1f} MB)")

        # Per-distribution query spec — point-lookup and delete keys must be
        # drawn from THIS distribution's keys to actually hit records.
        sample_keys = [k for k, _ in records]
        spec_rng = random.Random(args.seed + 1 + dist_order.index(dist) * 1000)
        spec = build_query_spec(args.n, spec_rng, sample_keys, None)
        spec_path = args.outdir / f"query_spec_{dist}.json"
        with open(spec_path, "w") as f:
            json.dump(spec, f)
        print(f"       wrote {spec_path}")

    # Back-compat default spec (uniform) for tooling that expects query_spec.json.
    default_src = args.outdir / "query_spec_uniform.json"
    if default_src.exists():
        default_dst = args.outdir / "query_spec.json"
        with open(default_src) as f:
            default = f.read()
        with open(default_dst, "w") as f:
            f.write(default)
        print(f"[gen] wrote {default_dst} (= query_spec_uniform.json)")


if __name__ == "__main__":
    main()
