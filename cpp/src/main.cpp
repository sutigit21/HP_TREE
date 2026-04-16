#include "hp_tree.hpp"
#include <random>
#include <iomanip>
#include <thread>

using namespace hptree;

static CompositeKeySchema g_schema;
static CompositeKeyEncoder* g_encoder = nullptr;

static CompositeKey make_key(int year, int month, int day,
                             const std::string& state,
                             const std::string& product,
                             double price, double version) {
    std::vector<int64_t> ints = {
        static_cast<int64_t>(year),
        static_cast<int64_t>(month),
        static_cast<int64_t>(day)
    };
    std::vector<double> floats = {price, version};
    std::vector<std::string> strs = {state, product};
    std::vector<bool> nulls(g_schema.dim_count(), false);
    return g_encoder->encode(ints, floats, strs, nulls);
}

static Record make_record(int year, int month, int day,
                           const std::string& state,
                           const std::string& product,
                           double price, double version) {
    Record r;
    r.key = make_key(year, month, day, state, product, price, version);
    r.tombstone = false;
    r.version.xmin = 1;
    r.version.xmax = TXN_COMMITTED;
    return r;
}

struct Timer {
    std::chrono::high_resolution_clock::time_point start;
    Timer() : start(std::chrono::high_resolution_clock::now()) {}
    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start).count();
    }
};

static void print_sep(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << "\n"
              << "  " << title << "\n"
              << std::string(70, '=') << "\n";
}

static void print_result(const std::string& name, bool passed) {
    std::cout << "  " << std::setw(50) << std::left << name
              << (passed ? "[PASS]" : "[FAIL]") << "\n";
}

// =========================================================================
//  TEST 1: Basic CRUD Operations
// =========================================================================
static bool test_basic_crud() {
    print_sep("TEST 1: Basic CRUD Operations");
    HPTreeConfig cfg;
    cfg.enable_wal = false;
    cfg.enable_delta_buffer = false;
    cfg.max_leaf_size = 10;
    cfg.branching_factor = 4;
    cfg.beta_strategy = BetaStrategy::ARITHMETIC_MEAN;
    HPTree tree(cfg, g_schema);

    bool all_pass = true;

    auto r1 = make_record(2022, 10, 15, "CA", "Laptop", 1200.00, 1.0);
    auto r2 = make_record(2022, 10, 16, "NY", "Monitor", 350.50, 1.1);
    auto r3 = make_record(2023, 1, 1, "TX", "Keyboard", 75.00, 2.0);

    tree.insert(r1); tree.insert(r2); tree.insert(r3);

    {
        bool p = tree.size() == 3;
        print_result("Insert 3 records -> size==3", p);
        all_pass &= p;
    }
    {
        auto res = tree.search(r1.key);
        bool p = !res.empty() && res[0]->key == r1.key;
        print_result("Point search finds inserted record", p);
        all_pass &= p;
    }
    {
        tree.remove(r2.key);
        bool p = tree.search(r2.key).empty();
        print_result("Delete removes record from search", p);
        all_pass &= p;
    }
    {
        Record r1_updated = r1;
        r1_updated.payload = {0x42, 0x43};
        tree.update(r1.key, r1_updated);
        auto res = tree.search(r1.key);
        bool p = !res.empty() && res[0]->payload.size() == 2;
        print_result("In-place update modifies payload", p);
        all_pass &= p;
    }
    {
        auto r4 = make_record(2021, 5, 20, "FL", "Mouse", 25.00, 3.0);
        Record r4_new = make_record(2024, 1, 1, "GA", "Chair", 500.00, 4.0);
        tree.insert(r4);
        tree.update(r4.key, r4_new);
        auto old_res = tree.search(r4.key);
        auto new_res = tree.search(r4_new.key);
        bool p = !new_res.empty();
        print_result("Key-changing update works", p);
        all_pass &= p;
    }

    return all_pass;
}

// =========================================================================
//  TEST 2: Bulk Load + Range Queries
// =========================================================================
static bool test_bulk_load_and_range() {
    print_sep("TEST 2: Bulk Load + Range Queries");
    HPTreeConfig cfg;
    cfg.enable_wal = false;
    cfg.enable_delta_buffer = false;
    cfg.max_leaf_size = 50;
    cfg.branching_factor = 20;
    cfg.beta_strategy = BetaStrategy::ARITHMETIC_MEAN;
    HPTree tree(cfg, g_schema);

    bool all_pass = true;
    const size_t N = 10000;

    std::mt19937 rng(42);
    std::vector<std::string> states = {"CA","TX","FL","NY","PA","IL","OH","GA","NC","MI"};
    std::vector<std::string> products = {"Laptop","Mouse","Keyboard","Monitor","Webcam","Headset","Desk","Chair"};

    std::vector<Record> records;
    records.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        int year = 2020 + static_cast<int>(rng() % 4);
        int month = 1 + static_cast<int>(rng() % 12);
        int day = 1 + static_cast<int>(rng() % 28);
        double price = 10.0 + (rng() % 300000) / 100.0;
        double version = 1.0 + (rng() % 500) / 100.0;
        auto& st = states[rng() % states.size()];
        auto& pr = products[rng() % products.size()];
        records.push_back(make_record(year, month, day, st, pr, price, version));
    }

    {
        Timer t;
        tree.bulk_load(records);
        double ms = t.elapsed_ms();
        bool p = tree.size() == N;
        print_result("Bulk load " + std::to_string(N) + " records ("
                     + std::to_string(ms) + "ms)", p);
        all_pass &= p;
    }
    {
        CompositeKey lo = make_key(2022, 1, 1, "AZ", "Chair", 0.0, 0.0);
        CompositeKey hi = make_key(2022, 12, 28, "WA", "Webcam", 5000.0, 10.0);
        Timer t;
        auto res = tree.range_search(lo, hi);
        double ms = t.elapsed_ms();
        bool p = !res.empty();
        print_result("Range search year=2022 -> " + std::to_string(res.size())
                     + " results (" + std::to_string(ms) + "ms)", p);
        all_pass &= p;
    }
    {
        CompositeKey lo = make_key(2022, 10, 15, "CA", "Laptop", 1000.0, 0.0);
        CompositeKey hi = make_key(2022, 10, 15, "CA", "Laptop", 2000.0, 10.0);
        auto res = tree.range_search(lo, hi);
        print_result("Narrow range search -> " + std::to_string(res.size()) + " results", true);
        all_pass &= true;
    }
    {
        CompositeKey lo = make_key(2000, 1, 1, "AZ", "Chair", 0.0, 0.0);
        CompositeKey hi = make_key(2000, 12, 28, "WA", "Webcam", 5000.0, 10.0);
        auto res = tree.range_search(lo, hi);
        bool p = res.empty();
        print_result("Miss query (year 2000) -> 0 results", p);
        all_pass &= p;
    }

    return all_pass;
}

// =========================================================================
//  TEST 3: Predicate Search (complex predicates)
// =========================================================================
static bool test_predicate_search() {
    print_sep("TEST 3: Predicate Search");
    HPTreeConfig cfg;
    cfg.enable_wal = false;
    cfg.enable_delta_buffer = false;
    cfg.max_leaf_size = 50;
    cfg.branching_factor = 20;
    HPTree tree(cfg, g_schema);

    bool all_pass = true;
    const size_t N = 5000;

    std::mt19937 rng(123);
    std::vector<std::string> states = {"CA","TX","FL","NY","PA"};
    std::vector<std::string> products = {"Laptop","Mouse","Keyboard","Monitor","Webcam"};

    std::vector<Record> records;
    for (size_t i = 0; i < N; ++i) {
        int year = 2020 + static_cast<int>(rng() % 5);
        int month = 1 + static_cast<int>(rng() % 12);
        int day = 1 + static_cast<int>(rng() % 28);
        double price = 10.0 + (rng() % 200000) / 100.0;
        double version = 1.0 + (rng() % 500) / 100.0;
        records.push_back(make_record(year, month, day,
            states[rng() % states.size()],
            products[rng() % products.size()],
            price, version));
    }
    tree.bulk_load(records);

    {
        size_t year_dim = 0;
        uint64_t year_2022_encoded = g_schema.dimensions[0].encode(2022);
        PredicateSet ps;
        ps.predicates.push_back(Predicate::eq(year_dim, year_2022_encoded));
        auto res = tree.predicate_search(ps);
        bool p = true;
        CompositeKeyEncoder enc(g_schema);
        for (auto* r : res) {
            int64_t y = g_schema.dimensions[0].decode_int(enc.extract_dim(r->key, 0));
            if (y != 2022) { p = false; break; }
        }
        print_result("EQ predicate (year=2022) -> " + std::to_string(res.size())
                     + " results, all correct=" + (p?"yes":"no"), p);
        all_pass &= p;
    }
    {
        size_t year_dim = 0;
        uint64_t lo = g_schema.dimensions[0].encode(2021);
        uint64_t hi = g_schema.dimensions[0].encode(2023);
        PredicateSet ps;
        ps.predicates.push_back(Predicate::between(year_dim, lo, hi));
        auto res = tree.predicate_search(ps);
        bool p = !res.empty();
        print_result("BETWEEN predicate (year 2021-2023) -> "
                     + std::to_string(res.size()), p);
        all_pass &= p;
    }
    {
        size_t state_dim = 3;
        uint64_t ca = g_schema.dimensions[state_dim].encode_string("CA");
        uint64_t ny = g_schema.dimensions[state_dim].encode_string("NY");
        uint64_t tx = g_schema.dimensions[state_dim].encode_string("TX");
        PredicateSet ps;
        ps.predicates.push_back(Predicate::in(state_dim, {ca, ny, tx}));
        auto res = tree.predicate_search(ps);
        bool p = !res.empty();
        print_result("IN predicate (state in CA,NY,TX) -> "
                     + std::to_string(res.size()), p);
        all_pass &= p;
    }

    return all_pass;
}

// =========================================================================
//  TEST 4: Iterator (forward + reverse scan)
// =========================================================================
static bool test_iterator() {
    print_sep("TEST 4: Iterator (Forward + Reverse)");
    HPTreeConfig cfg;
    cfg.enable_wal = false;
    cfg.enable_delta_buffer = false;
    cfg.max_leaf_size = 10;
    cfg.branching_factor = 4;
    HPTree tree(cfg, g_schema);

    bool all_pass = true;

    std::vector<Record> records;
    for (int y = 2020; y <= 2023; ++y) {
        for (int m = 1; m <= 12; ++m) {
            records.push_back(make_record(y, m, 15, "CA", "Laptop", 100.0, 1.0));
        }
    }
    tree.bulk_load(records);

    {
        auto it = tree.begin();
        size_t count = 0;
        CompositeKey prev = 0;
        bool ordered = true;
        while (it.valid()) {
            if (it.key() < prev) ordered = false;
            prev = it.key();
            count++;
            it.next();
        }
        bool p = count == records.size() && ordered;
        print_result("Forward scan: " + std::to_string(count)
                     + " records, ordered=" + (ordered?"yes":"no"), p);
        all_pass &= p;
    }
    {
        auto it = tree.rbegin();
        size_t count = 0;
        CompositeKey prev = COMPOSITE_KEY_MAX;
        bool ordered = true;
        while (it.valid()) {
            if (it.key() > prev) ordered = false;
            prev = it.key();
            count++;
            it.next();
        }
        bool p = count == records.size() && ordered;
        print_result("Reverse scan: " + std::to_string(count)
                     + " records, ordered=" + (ordered?"yes":"no"), p);
        all_pass &= p;
    }
    {
        CompositeKey lo = make_key(2022, 1, 1, "AZ", "Chair", 0.0, 0.0);
        CompositeKey hi = make_key(2022, 12, 28, "WA", "Webcam", 5000.0, 10.0);
        auto it = tree.begin(lo, hi);
        auto collected = it.collect();
        bool p = !collected.empty();
        print_result("Range-bounded iterator -> " + std::to_string(collected.size())
                     + " results", p);
        all_pass &= p;
    }

    return all_pass;
}

// =========================================================================
//  TEST 5: Delta Buffer (LSM-style batched writes)
// =========================================================================
static bool test_delta_buffer() {
    print_sep("TEST 5: Delta Buffer (LSM-style writes)");
    HPTreeConfig cfg;
    cfg.enable_wal = false;
    cfg.enable_delta_buffer = true;
    cfg.delta_buffer_cap = 100;
    cfg.max_leaf_size = 20;
    cfg.branching_factor = 10;
    HPTree tree(cfg, g_schema);

    bool all_pass = true;

    std::vector<Record> seed;
    for (int i = 0; i < 200; ++i) {
        seed.push_back(make_record(2020, 1 + i % 12, 1 + i % 28,
                                   "CA", "Laptop", 100.0 + i, 1.0));
    }
    tree.bulk_load(seed);

    for (int i = 0; i < 50; ++i) {
        auto rec = make_record(2024, 1 + i % 12, 1 + i % 28,
                               "NY", "Mouse", 50.0 + i, 2.0);
        tree.insert(rec);
    }

    {
        auto r = make_record(2024, 1, 1, "NY", "Mouse", 50.0, 2.0);
        auto res = tree.search(r.key);
        bool p = !res.empty();
        print_result("Search finds record in delta buffer", p);
        all_pass &= p;
    }
    {
        tree.flush_delta();
        auto r = make_record(2024, 1, 1, "NY", "Mouse", 50.0, 2.0);
        auto res = tree.search(r.key);
        bool p = !res.empty();
        print_result("After flush, record still findable in tree", p);
        all_pass &= p;
    }

    return all_pass;
}

// =========================================================================
//  TEST 6: MVCC Visibility
// =========================================================================
static bool test_mvcc() {
    print_sep("TEST 6: MVCC Visibility");
    HPTreeConfig cfg;
    cfg.enable_wal = false;
    cfg.enable_delta_buffer = false;
    cfg.enable_mvcc = true;
    cfg.max_leaf_size = 50;
    HPTree tree(cfg, g_schema);

    bool all_pass = true;

    auto r1 = make_record(2022, 6, 15, "CA", "Laptop", 999.0, 1.0);
    TxnId t1 = tree.begin_transaction();
    tree.insert(r1, t1);
    tree.commit_transaction(t1);

    {
        TxnId t2 = tree.begin_transaction();
        auto res = tree.search(r1.key, t2);
        bool p = !res.empty();
        print_result("Committed record visible to later txn", p);
        all_pass &= p;
    }
    {
        TxnId t_old = 0;
        auto res = tree.search(r1.key, t_old);
        bool p = res.empty();
        print_result("Committed record invisible to older txn", p);
        all_pass &= p;
    }

    return all_pass;
}

// =========================================================================
//  TEST 7: Beta Threshold Strategies Comparison
// =========================================================================
static bool test_beta_strategies() {
    print_sep("TEST 7: Beta Threshold Strategies");
    bool all_pass = true;
    const size_t N = 5000;

    std::mt19937 rng(42);
    std::vector<std::string> states = {"CA","TX","FL","NY","PA"};
    std::vector<std::string> products = {"Laptop","Mouse","Keyboard","Monitor","Webcam"};

    std::vector<Record> records;
    for (size_t i = 0; i < N; ++i) {
        int year = 2020 + static_cast<int>(rng() % 4);
        int month = 1 + static_cast<int>(rng() % 12);
        int day = 1 + static_cast<int>(rng() % 28);
        double price = 10.0 + (rng() % 300000) / 100.0;
        double version = 1.0 + (rng() % 500) / 100.0;
        records.push_back(make_record(year, month, day,
            states[rng() % states.size()],
            products[rng() % products.size()],
            price, version));
    }

    struct StrategyResult {
        std::string name;
        BetaStrategy strat;
        uint64_t leaves = 0;
        uint64_t homo = 0;
        uint32_t depth = 0;
        double build_ms = 0;
        double query_ms = 0;
        size_t query_results = 0;
    };

    std::vector<StrategyResult> results;

    BetaStrategy strats[] = {
        BetaStrategy::FIXED_STRICT,
        BetaStrategy::ARITHMETIC_MEAN,
        BetaStrategy::MEDIAN,
        BetaStrategy::STDDEV_2X,
        BetaStrategy::STDDEV_6X,
        BetaStrategy::ADAPTIVE_LOCAL,
    };
    const char* names[] = {
        "FIXED_STRICT", "ARITHMETIC_MEAN", "MEDIAN", "STDDEV_2X", "STDDEV_6X",
        "ADAPTIVE_LOCAL"
    };

    CompositeKey q_lo = make_key(2022, 1, 1, "AZ", "Chair", 0.0, 0.0);
    CompositeKey q_hi = make_key(2022, 12, 28, "WA", "Webcam", 5000.0, 10.0);

    for (int s = 0; s < 6; ++s) {
        HPTreeConfig cfg;
        cfg.enable_wal = false;
        cfg.enable_delta_buffer = false;
        cfg.max_leaf_size = 50;
        cfg.branching_factor = 20;
        cfg.beta_strategy = strats[s];
        HPTree tree(cfg, g_schema);

        auto recs_copy = records;
        Timer bt;
        tree.bulk_load(recs_copy);
        double build_ms = bt.elapsed_ms();

        Timer qt;
        auto qres = tree.range_search(q_lo, q_hi);
        double query_ms = qt.elapsed_ms();

        auto stats = tree.statistics();
        results.push_back({names[s], strats[s], stats.total_leaves,
                          stats.total_homogeneous, stats.tree_depth,
                          build_ms, query_ms, qres.size()});
    }

    std::cout << "\n  " << std::setw(20) << "Strategy"
              << std::setw(10) << "Leaves"
              << std::setw(10) << "Homo"
              << std::setw(8) << "Depth"
              << std::setw(12) << "Build(ms)"
              << std::setw(12) << "Query(ms)"
              << std::setw(10) << "Results" << "\n";
    std::cout << "  " << std::string(82, '-') << "\n";

    for (auto& r : results) {
        std::cout << "  " << std::setw(20) << r.name
                  << std::setw(10) << r.leaves
                  << std::setw(10) << r.homo
                  << std::setw(8) << r.depth
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.build_ms
                  << std::setw(12) << r.query_ms
                  << std::setw(10) << r.query_results << "\n";
    }

    all_pass = true;
    for (auto& r : results) {
        bool p = r.leaves > 0 && r.depth > 0;
        print_result(r.name + " produces valid tree", p);
        all_pass &= p;
    }

    return all_pass;
}

// =========================================================================
//  TEST 8: Statistics & Cost Model
// =========================================================================
static bool test_statistics_and_cost() {
    print_sep("TEST 8: Statistics & Cost Model");
    HPTreeConfig cfg;
    cfg.enable_wal = false;
    cfg.enable_delta_buffer = false;
    cfg.max_leaf_size = 50;
    cfg.branching_factor = 20;
    HPTree tree(cfg, g_schema);

    bool all_pass = true;
    const size_t N = 5000;

    std::mt19937 rng(42);
    std::vector<std::string> states = {"CA","TX","FL","NY","PA"};
    std::vector<std::string> products = {"Laptop","Mouse","Keyboard"};

    std::vector<Record> records;
    for (size_t i = 0; i < N; ++i) {
        records.push_back(make_record(
            2020 + static_cast<int>(rng() % 4),
            1 + static_cast<int>(rng() % 12),
            1 + static_cast<int>(rng() % 28),
            states[rng() % states.size()],
            products[rng() % products.size()],
            10.0 + (rng() % 300000) / 100.0,
            1.0 + (rng() % 500) / 100.0));
    }
    tree.bulk_load(records);

    {
        auto stats = tree.statistics();
        bool p = stats.total_records == N && stats.total_leaves > 0
              && stats.tree_depth > 0;
        print_result("Statistics: records=" + std::to_string(stats.total_records)
                     + " leaves=" + std::to_string(stats.total_leaves)
                     + " depth=" + std::to_string(stats.tree_depth), p);
        all_pass &= p;
    }
    {
        auto stats = tree.statistics();
        bool p = stats.dim_histograms.size() == g_schema.dim_count();
        print_result("Histograms built for all "
                     + std::to_string(g_schema.dim_count()) + " dimensions", p);
        all_pass &= p;
        if (p && !stats.dim_histograms.empty()) {
            auto& h = stats.dim_histograms[0];
            std::cout << "    -> dim[0] '" << h.dim_name
                      << "': distinct=" << h.distinct_count
                      << " buckets=" << h.buckets.size() << "\n";
        }
    }
    {
        PredicateSet ps;
        uint64_t y = g_schema.dimensions[0].encode(2022);
        ps.predicates.push_back(Predicate::eq(0, y));
        auto cost = tree.estimate_query_cost(ps);
        bool p = cost.estimated_rows > 0;
        print_result("Cost estimate: rows=" + std::to_string(cost.estimated_rows)
                     + " cost=" + std::to_string(cost.total_cost)
                     + (cost.recommend_seq_scan ? " (SeqScan)" : " (IndexScan)"), p);
        all_pass &= p;
    }

    return all_pass;
}

// =========================================================================
//  TEST 9: Aggregates (COUNT, SUM, GROUP BY)
// =========================================================================
static bool test_aggregates() {
    print_sep("TEST 9: Aggregates (COUNT, SUM, GROUP BY)");
    HPTreeConfig cfg;
    cfg.enable_wal = false;
    cfg.enable_delta_buffer = false;
    cfg.enable_aggregates = true;
    cfg.max_leaf_size = 50;
    cfg.branching_factor = 20;
    HPTree tree(cfg, g_schema);

    bool all_pass = true;

    std::vector<Record> records;
    for (int y = 2020; y <= 2023; ++y) {
        for (int m = 1; m <= 12; ++m) {
            for (int d = 1; d <= 28; d += 7) {
                records.push_back(
                    make_record(y, m, d, "CA", "Laptop", 100.0 * m, 1.0));
            }
        }
    }
    tree.bulk_load(records);

    {
        uint64_t c = tree.count();
        bool p = c == records.size();
        print_result("COUNT(*) = " + std::to_string(c), p);
        all_pass &= p;
    }
    {
        PredicateSet ps;
        uint64_t y = g_schema.dimensions[0].encode(2022);
        ps.predicates.push_back(Predicate::eq(0, y));
        uint64_t c = tree.count_predicate(ps);
        bool p = c > 0;
        print_result("COUNT(*) WHERE year=2022 -> " + std::to_string(c), p);
        all_pass &= p;
    }
    {
        auto agg = tree.aggregate_dim(5);
        bool p = agg.count > 0 && agg.sum > 0;
        print_result("SUM(price): count=" + std::to_string(agg.count)
                     + " sum=" + std::to_string(agg.sum)
                     + " avg=" + std::to_string(agg.avg), p);
        all_pass &= p;
    }
    {
        auto groups = tree.group_by_count(0);
        bool p = groups.size() == 4;
        std::string detail;
        for (auto& [k, v] : groups) {
            int64_t year = g_schema.dimensions[0].decode_int(k);
            detail += std::to_string(year) + ":" + std::to_string(v) + " ";
        }
        print_result("GROUP BY year -> " + std::to_string(groups.size())
                     + " groups [" + detail + "]", p);
        all_pass &= p;
    }

    return all_pass;
}

// =========================================================================
//  TEST 10: Stress Test (high-volume insert + delete + query)
// =========================================================================
static bool test_stress() {
    print_sep("TEST 10: Stress Test (100K ops)");
    HPTreeConfig cfg;
    cfg.enable_wal = false;
    cfg.enable_delta_buffer = true;
    cfg.delta_buffer_cap = 2000;
    cfg.max_leaf_size = 50;
    cfg.branching_factor = 50;
    cfg.beta_strategy = BetaStrategy::ARITHMETIC_MEAN;
    HPTree tree(cfg, g_schema);

    bool all_pass = true;
    const size_t N = 100000;

    std::mt19937 rng(999);
    std::vector<std::string> states = {"CA","TX","FL","NY","PA","IL","OH","GA"};
    std::vector<std::string> products = {"Laptop","Mouse","Keyboard","Monitor",
                                         "Webcam","Headset","Desk","Chair"};

    std::vector<Record> records;
    records.reserve(N);
    {
        Timer t;
        for (size_t i = 0; i < N; ++i) {
            records.push_back(make_record(
                2018 + static_cast<int>(rng() % 8),
                1 + static_cast<int>(rng() % 12),
                1 + static_cast<int>(rng() % 28),
                states[rng() % states.size()],
                products[rng() % products.size()],
                1.0 + (rng() % 500000) / 100.0,
                0.5 + (rng() % 1000) / 100.0));
        }
        tree.bulk_load(records);
        double ms = t.elapsed_ms();
        bool p = tree.size() == N;
        print_result("Bulk load " + std::to_string(N) + " records in "
                     + std::to_string(ms) + "ms", p);
        all_pass &= p;
    }
    {
        Timer t;
        size_t found = 0;
        for (int i = 0; i < 1000; ++i) {
            auto& r = records[rng() % records.size()];
            auto res = tree.search(r.key);
            if (!res.empty()) found++;
        }
        double ms = t.elapsed_ms();
        bool p = found > 500;
        print_result("1000 point searches in " + std::to_string(ms) + "ms, found="
                     + std::to_string(found), p);
        all_pass &= p;
    }
    {
        Timer t;
        CompositeKey lo = make_key(2022, 1, 1, "AZ", "Chair", 0.0, 0.0);
        CompositeKey hi = make_key(2022, 12, 28, "WA", "Webcam", 5000.0, 10.0);
        auto res = tree.range_search(lo, hi);
        double ms = t.elapsed_ms();
        bool p = !res.empty();
        print_result("Range query over " + std::to_string(N) + " records -> "
                     + std::to_string(res.size()) + " results in "
                     + std::to_string(ms) + "ms", p);
        all_pass &= p;
    }
    {
        Timer t;
        for (int i = 0; i < 1000; ++i) {
            tree.insert(make_record(
                2025, 1 + static_cast<int>(rng() % 12),
                1 + static_cast<int>(rng() % 28),
                states[rng() % states.size()],
                products[rng() % products.size()],
                100.0 + (rng() % 100000) / 100.0,
                1.0 + (rng() % 500) / 100.0));
        }
        tree.flush_delta();
        double ms = t.elapsed_ms();
        print_result("1000 incremental inserts + flush in "
                     + std::to_string(ms) + "ms", true);
    }
    {
        tree.print_stats(std::cout);
        all_pass &= true;
    }

    return all_pass;
}

// =========================================================================
//  TEST 11: WAL (Write-Ahead Log)
// =========================================================================
static bool test_wal() {
    print_sep("TEST 11: Write-Ahead Log");
    bool all_pass = true;

    std::string wal_path = "/tmp/hp_tree_test.wal";

    {
        HPTreeConfig cfg;
        cfg.enable_wal = true;
        cfg.wal_path = wal_path;
        cfg.enable_delta_buffer = false;
        HPTree tree(cfg, g_schema);

        TxnId t1 = tree.begin_transaction();
        tree.insert(make_record(2022, 6, 15, "CA", "Laptop", 1000.0, 1.0), t1);
        tree.insert(make_record(2023, 1, 1, "NY", "Mouse", 25.0, 2.0), t1);
        tree.commit_transaction(t1);

        tree.checkpoint();
        print_result("WAL checkpoint written", true);
    }
    {
        WalManager wal_reader;
        auto recovery = wal_reader.recover(wal_path);
        bool p = !recovery.redo_records.empty()
              && !recovery.committed_txns.empty();
        print_result("WAL recovery: " + std::to_string(recovery.redo_records.size())
                     + " records, " + std::to_string(recovery.committed_txns.size())
                     + " committed txns", p);
        all_pass &= p;
    }

    std::remove(wal_path.c_str());
    return all_pass;
}

// =========================================================================
//  TEST 12: Composite Key Encode/Decode Roundtrip
// =========================================================================
static bool test_composite_key_roundtrip() {
    print_sep("TEST 12: Composite Key Encode/Decode");
    bool all_pass = true;
    CompositeKeyEncoder encoder(g_schema);

    struct TestCase {
        int year, month, day;
        std::string state, product;
        double price, version;
    };

    std::vector<TestCase> cases = {
        {2022, 10, 15, "CA", "Laptop", 1200.00, 1.0},
        {2020, 1, 1, "NY", "Mouse", 25.00, 5.5},
        {2024, 12, 28, "TX", "Keyboard", 75.50, 2.5},
        {2018, 6, 15, "FL", "Monitor", 350.00, 1.1},
    };

    for (auto& tc : cases) {
        CompositeKey key = make_key(tc.year, tc.month, tc.day,
                                    tc.state, tc.product, tc.price, tc.version);
        auto decoded = encoder.decode(key);

        bool p = true;
        if (!decoded.int_vals.empty()) {
            int64_t dec_year = g_schema.dimensions[0].decode_int(
                encoder.extract_dim(key, 0));
            int64_t dec_month = g_schema.dimensions[1].decode_int(
                encoder.extract_dim(key, 1));
            int64_t dec_day = g_schema.dimensions[2].decode_int(
                encoder.extract_dim(key, 2));
            if (dec_year != tc.year) p = false;
            if (dec_month != tc.month) p = false;
            if (dec_day != tc.day) p = false;
        }

        std::string dec_state = g_schema.dimensions[3].decode_string(
            encoder.extract_dim(key, 3));
        std::string dec_product = g_schema.dimensions[4].decode_string(
            encoder.extract_dim(key, 4));
        if (dec_state != tc.state) p = false;
        if (dec_product != tc.product) p = false;

        std::string label = std::to_string(tc.year) + "-"
            + std::to_string(tc.month) + "-" + std::to_string(tc.day)
            + " " + tc.state + " " + tc.product;
        print_result("Roundtrip: " + label, p);
        all_pass &= p;
    }

    return all_pass;
}

// =========================================================================
//  TEST 13: Split and Merge under sequential inserts/deletes
// =========================================================================
static bool test_split_merge() {
    print_sep("TEST 13: Split/Merge under sequential ops");
    HPTreeConfig cfg;
    cfg.enable_wal = false;
    cfg.enable_delta_buffer = false;
    cfg.max_leaf_size = 5;
    cfg.min_leaf_size = 2;
    cfg.branching_factor = 3;
    HPTree tree(cfg, g_schema);

    bool all_pass = true;

    std::vector<Record> inserted;
    for (int i = 0; i < 50; ++i) {
        auto r = make_record(2020 + i % 5, 1 + i % 12, 1 + i % 28,
                             "CA", "Laptop", 100.0 + i, 1.0);
        tree.insert(r);
        inserted.push_back(r);
    }

    {
        auto stats = tree.statistics();
        bool p = stats.total_splits > 0;
        print_result("Splits occurred: " + std::to_string(stats.total_splits), p);
        all_pass &= p;
    }
    {
        bool p = tree.size() == 50;
        print_result("All 50 records present", p);
        all_pass &= p;
    }

    for (int i = 0; i < 40; ++i) {
        tree.remove(inserted[i].key);
    }
    tree.compact();

    {
        bool found_all = true;
        for (int i = 40; i < 50; ++i) {
            auto res = tree.search(inserted[i].key);
            if (res.empty()) { found_all = false; break; }
        }
        print_result("Remaining 10 records all findable", found_all);
        all_pass &= found_all;
    }

    return all_pass;
}

// =========================================================================
//  TEST 14: Rebuild (online index rebuild)
// =========================================================================
static bool test_rebuild() {
    print_sep("TEST 14: Online Index Rebuild");
    HPTreeConfig cfg;
    cfg.enable_wal = false;
    cfg.enable_delta_buffer = false;
    cfg.max_leaf_size = 10;
    cfg.branching_factor = 4;
    HPTree tree(cfg, g_schema);

    bool all_pass = true;

    std::vector<Record> records;
    for (int i = 0; i < 100; ++i) {
        records.push_back(make_record(2020 + i % 4, 1 + i % 12, 1 + i % 28,
                                      "CA", "Laptop", 100.0 + i, 1.0));
    }
    tree.bulk_load(records);

    for (int i = 0; i < 50; ++i) {
        tree.remove(records[i].key);
    }

    auto before = tree.statistics();
    tree.rebuild();
    auto after = tree.statistics();

    {
        bool p = after.total_records <= before.total_records
              && after.total_records == 50;
        print_result("Rebuild compacts: before=" + std::to_string(before.total_records)
                     + " after=" + std::to_string(after.total_records), p);
        all_pass &= p;
    }
    {
        bool found_all = true;
        for (int i = 50; i < 100; ++i) {
            auto res = tree.search(records[i].key);
            if (res.empty()) { found_all = false; break; }
        }
        print_result("All surviving records findable after rebuild", found_all);
        all_pass &= found_all;
    }

    return all_pass;
}

// =========================================================================
//  TEST 15: Adaptive Local Beta Threshold
// =========================================================================
static bool test_adaptive_beta() {
    print_sep("TEST 15: Adaptive Local Beta (zero-cost data-driven threshold)");
    bool all_pass = true;

    std::mt19937 rng(777);
    std::vector<std::string> states = {"CA","TX","FL","NY","PA"};
    std::vector<std::string> products = {"Laptop","Mouse","Keyboard","Monitor","Webcam"};

    auto gen_uniform = [&](size_t n) {
        std::vector<Record> recs;
        for (size_t i = 0; i < n; ++i) {
            recs.push_back(make_record(
                2020 + static_cast<int>(rng() % 5),
                1 + static_cast<int>(rng() % 12),
                1 + static_cast<int>(rng() % 28),
                states[rng() % states.size()],
                products[rng() % products.size()],
                10.0 + (rng() % 300000) / 100.0,
                1.0 + (rng() % 500) / 100.0));
        }
        return recs;
    };

    auto gen_clustered = [&](size_t n) {
        std::vector<Record> recs;
        for (size_t i = 0; i < n; ++i) {
            int cluster = static_cast<int>(i / (n / 3));
            int year = 2020 + cluster;
            recs.push_back(make_record(
                year, 6, 15, "CA", "Laptop",
                100.0 + (rng() % 100) / 100.0, 1.0));
        }
        return recs;
    };

    auto gen_skewed = [&](size_t n) {
        std::vector<Record> recs;
        for (size_t i = 0; i < n; ++i) {
            bool hot = (rng() % 100) < 80;
            if (hot) {
                recs.push_back(make_record(
                    2022, 6, 15, "CA", "Laptop",
                    500.0 + (rng() % 1000) / 100.0, 1.0));
            } else {
                recs.push_back(make_record(
                    2020 + static_cast<int>(rng() % 5),
                    1 + static_cast<int>(rng() % 12),
                    1 + static_cast<int>(rng() % 28),
                    states[rng() % states.size()],
                    products[rng() % products.size()],
                    10.0 + (rng() % 300000) / 100.0,
                    1.0 + (rng() % 500) / 100.0));
            }
        }
        return recs;
    };

    struct DistTest {
        std::string name;
        std::function<std::vector<Record>(size_t)> gen;
    };
    std::vector<DistTest> dists = {
        {"Uniform",   gen_uniform},
        {"Clustered", gen_clustered},
        {"Skewed",    gen_skewed},
    };

    const size_t N = 10000;
    CompositeKey q_lo = make_key(2022, 1, 1, "AZ", "Chair", 0.0, 0.0);
    CompositeKey q_hi = make_key(2022, 12, 28, "WA", "Webcam", 5000.0, 10.0);

    std::cout << "\n  " << std::setw(12) << "Distrib"
              << std::setw(10) << "Leaves"
              << std::setw(10) << "Homo"
              << std::setw(8)  << "Depth"
              << std::setw(12) << "Build(ms)"
              << std::setw(12) << "Query(ms)"
              << std::setw(10) << "Results" << "\n";
    std::cout << "  " << std::string(74, '-') << "\n";

    for (auto& dt : dists) {
        auto recs = dt.gen(N);

        HPTreeConfig cfg;
        cfg.enable_wal = false;
        cfg.enable_delta_buffer = false;
        cfg.max_leaf_size = 50;
        cfg.branching_factor = 20;
        cfg.beta_strategy = BetaStrategy::ADAPTIVE_LOCAL;
        HPTree tree(cfg, g_schema);

        auto recs_copy = recs;
        Timer bt;
        tree.bulk_load(recs_copy);
        double build_ms = bt.elapsed_ms();

        Timer qt;
        auto qres = tree.range_search(q_lo, q_hi);
        double query_ms = qt.elapsed_ms();

        auto stats = tree.statistics();

        std::cout << "  " << std::setw(12) << dt.name
                  << std::setw(10) << stats.total_leaves
                  << std::setw(10) << stats.total_homogeneous
                  << std::setw(8)  << stats.tree_depth
                  << std::setw(12) << std::fixed << std::setprecision(2) << build_ms
                  << std::setw(12) << query_ms
                  << std::setw(10) << qres.size() << "\n";

        bool p = stats.total_leaves > 0 && stats.tree_depth > 0;
        print_result("ADAPTIVE_LOCAL on " + dt.name + ": valid tree", p);
        all_pass &= p;

        HPTreeConfig cfg_am;
        cfg_am.enable_wal = false;
        cfg_am.enable_delta_buffer = false;
        cfg_am.max_leaf_size = 50;
        cfg_am.branching_factor = 20;
        cfg_am.beta_strategy = BetaStrategy::ARITHMETIC_MEAN;
        HPTree tree_am(cfg_am, g_schema);
        auto recs_am = recs;
        tree_am.bulk_load(recs_am);
        auto qres_am = tree_am.range_search(q_lo, q_hi);
        bool correct = qres.size() == qres_am.size();
        print_result("  Result count matches AM (" + std::to_string(qres.size())
                     + " vs " + std::to_string(qres_am.size()) + ")", correct);
        all_pass &= correct;
    }

    return all_pass;
}

// =========================================================================
//  MAIN
// =========================================================================
int main() {
    g_schema = make_default_sales_schema();
    CompositeKeyEncoder encoder(g_schema);
    g_encoder = &encoder;

    std::cout << "\n"
              << std::string(70, '#') << "\n"
              << "  HP-TREE C++ Implementation - Comprehensive Test Suite\n"
              << "  Schema: " << g_schema.dim_count() << " dimensions, "
              << g_schema.total_bits << " total bits\n"
              << std::string(70, '#') << "\n";

    int total = 0, passed = 0;

    auto run = [&](bool (*test_fn)()) {
        total++;
        if (test_fn()) passed++;
    };

    run(test_composite_key_roundtrip);
    run(test_basic_crud);
    run(test_bulk_load_and_range);
    run(test_predicate_search);
    run(test_iterator);
    run(test_delta_buffer);
    run(test_mvcc);
    run(test_beta_strategies);
    run(test_statistics_and_cost);
    run(test_aggregates);
    run(test_stress);
    run(test_wal);
    run(test_split_merge);
    run(test_rebuild);
    run(test_adaptive_beta);

    std::cout << "\n" << std::string(70, '#') << "\n"
              << "  RESULTS: " << passed << " / " << total << " tests passed\n"
              << std::string(70, '#') << "\n\n";

    return (passed == total) ? 0 : 1;
}
