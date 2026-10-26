/*
 * count_build -- build COUNT's read-only n-gram continuation tables (Eq. 1-2 of the paper).
 *
 * For every history order n = 1..max_order it counts how often each continuation token w
 * follows each n-token history h in the corpus, then keeps:
 *
 *     rows:        histories h with C(h) >= history_min_count[n]
 *     candidates:  the top k_store continuations of h by count, dropping C(h,w) < pair_min_count[n]
 *     scores:      log10( C(h,w) / C(h) ) quantized to 8 bits per order, spanning
 *                  [min retained log10, 0]   -- exactly the paper's quantization
 *
 * Stupid backoff (Eq. 2) is NOT baked into the stored rows: a row holds observed continuations
 * only, and the reader assembles backed-off candidates from the suffix chain at lookup time.
 * That is a deliberate deviation from the paper's storage layout (their rows store merged
 * top-K): the tables end up smaller, and K, J and the Sec. 5.1 table-mixing weight lambda stay
 * inference-time knobs instead of build-time commitments.
 *
 * Two knobs mirror the paper's Table 4 / Appendix A.2:
 *   --history-min-count  (their c_min, 265 for the general Dolma tables)
 *   --pair-min-count     (their "n-gram floor": 7 for orders 4-6, 0 for orders 1-3)
 *
 * Memory: the counting map holds one 12-byte slot per distinct (history, continuation) pair
 * (48-bit history hash + 16-bit continuation = 64-bit key, plus a uint32 count). Slots are
 * compacted in place into the record array, so peak RSS is the map capacity, not map + records.
 * On this 62 GB box (an LLM service already holds most of it) keep the estimate under ~4 GB:
 * 100M tokens at order 6 needs ~1.6 GB.
 *
 * Build:  see count_native/build.sh
 * Usage:  count_build --corpus data/fineweb-edu/train.bin --vocab 50304 \
 *             --tokens 100000000 --max-order 6 --k-store 32 \
 *             --history-min-count 20 --pair-min-count 0:0,1:0,2:0,3:2,4:2,5:2,6:2 \
 *             --out-dir count_tables/fw100m_o6
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cinttypes>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "count_common.h"

/* A slot and a record are the same thing: (pair key, count). key == 0 marks an empty slot. */
struct __attribute__((packed)) Entry {
    uint64_t key;
    uint32_t cnt;
};
static_assert(sizeof(Entry) == 12, "Entry must stay 12 bytes: it sets the memory ceiling");

struct OrderSetting {
    int order;
    uint32_t history_min_count;
    uint32_t pair_min_count;
};

struct Options {
    std::string corpus = "data/fineweb_edu/train.bin";
    std::string out_dir = "count_tables/count";
    uint64_t tokens = 0;          // 0 = whole file
    uint64_t offset = 0;          // start token (lets a domain table use a disjoint slice)
    uint32_t vocab = 50304;
    int max_order = 6;
    int k_store = 32;
    double backoff = 0.4;         // Eq. 2's discount, recorded for the reader
    int threads = 8;
    double memory_gib = 4.0;      // ceiling for the counting map
    bool dry_run = false;
    std::vector<OrderSetting> settings;   // settings[order], order >= 1
};

static void die(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "count_build: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(1);
}

static uint64_t parse_size(const char* s) {
    char* end = nullptr;
    double v = strtod(s, &end);
    if (end == s) die("bad number '%s'", s);
    if (end && *end) {
        char c = *end;
        if (c == 'k' || c == 'K') v *= 1e3;
        else if (c == 'm' || c == 'M') v *= 1e6;
        else if (c == 'g' || c == 'G') v *= 1e9;
        else die("bad suffix in '%s'", s);
    }
    return (uint64_t)(v + 0.5);
}

/* "5" or "1:10,2:15,3:20" -> per-order values (the bare form applies to every order). */
static std::vector<std::pair<int, uint32_t>> parse_per_order(const char* spec, int max_order) {
    std::vector<std::pair<int, uint32_t>> out;
    const char* p = spec;
    while (*p) {
        char* end = nullptr;
        long first = strtol(p, &end, 10);
        if (end == p) die("bad threshold spec '%s'", spec);
        p = end;
        if (*p == ':') {
            ++p;
            long val = strtol(p, &end, 10);
            if (end == p) die("bad threshold value in '%s'", spec);
            p = end;
            if (first < 1 || first > max_order) die("order %ld out of range 1..%d", first, max_order);
            out.push_back({(int)first, (uint32_t)val});
        } else {
            for (int o = 1; o <= max_order; ++o) out.push_back({o, (uint32_t)first});
        }
        if (*p == ',') ++p;
        else if (*p) die("trailing garbage in '%s'", spec);
    }
    return out;
}

static void apply_settings(Options& opt, const char* hist_spec, const char* pair_spec) {
    opt.settings.assign(opt.max_order + 1, OrderSetting{0, 0, 0});
    auto h = parse_per_order(hist_spec, opt.max_order);
    auto c = parse_per_order(pair_spec, opt.max_order);
    if ((int)h.size() != opt.max_order || (int)c.size() != opt.max_order)
        die("threshold spec must cover every order 1..%d", opt.max_order);
    for (auto& kv : h) opt.settings[kv.first].history_min_count = kv.second;
    for (auto& kv : c) opt.settings[kv.first].pair_min_count = kv.second;
    for (int o = 1; o <= opt.max_order; ++o) {
        opt.settings[o].order = o;
        if (opt.settings[o].history_min_count < 1)
            die("history-min-count for order %d must be >= 1", o);
    }
}

static void parse_args(int argc, char** argv, Options& opt) {
    std::string hist = "20", pair = "0";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto take = [&](const char* /*what*/) -> const char* {
            if (i + 1 >= argc) die("%s needs a value", a.c_str());
            return argv[++i];
        };
        if (a == "--corpus") opt.corpus = take(a.c_str());
        else if (a == "--out-dir") opt.out_dir = take(a.c_str());
        else if (a == "--tokens") opt.tokens = parse_size(take(a.c_str()));
        else if (a == "--offset") opt.offset = parse_size(take(a.c_str()));
        else if (a == "--vocab") opt.vocab = (uint32_t)parse_size(take(a.c_str()));
        else if (a == "--max-order") opt.max_order = (int)parse_size(take(a.c_str()));
        else if (a == "--k-store") opt.k_store = (int)parse_size(take(a.c_str()));
        else if (a == "--backoff") opt.backoff = strtod(take(a.c_str()), nullptr);
        else if (a == "--threads") opt.threads = (int)parse_size(take(a.c_str()));
        else if (a == "--memory-gib") opt.memory_gib = strtod(take(a.c_str()), nullptr);
        else if (a == "--history-min-count") hist = take(a.c_str());
        else if (a == "--pair-min-count") pair = take(a.c_str());
        else if (a == "--dry-run") opt.dry_run = true;
        else if (a == "--help" || a == "-h") {
            printf(
                "count_build --corpus FILE [--offset N] [--tokens N] --vocab V [--max-order 6]\n"
                "  [--k-store 32] [--backoff 0.4] [--history-min-count C | o:C,o:C,...]\n"
                "  [--pair-min-count F | o:F,o:F,...] [--threads T] [--memory-gib G] [--out-dir D]\n"
                "  [--dry-run]\n");
            exit(0);
        } else die("unknown argument '%s'", a.c_str());
    }
    if (opt.max_order < 1 || opt.max_order > COUNT_MAX_ORDER_LIMIT)
        die("max-order must be 1..%d", COUNT_MAX_ORDER_LIMIT);
    if (opt.k_store < 1 || opt.k_store > 4096) die("k-store must be 1..4096");
    if (opt.vocab < 2 || opt.vocab > 65536) die("vocab must be 2..65536 (16-bit corpus)");
    if (opt.threads < 1) opt.threads = 1;
    apply_settings(opt, hist.c_str(), pair.c_str());
}

/* ------------------------------------------------------------------ corpus */

struct Corpus {
    const uint16_t* tok = nullptr;
    uint64_t n = 0;
    int fd = -1;

    void open(const std::string& path, uint64_t offset, uint64_t tokens) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) die("cannot open corpus '%s'", path.c_str());
        struct stat st;
        if (fstat(fd, &st) != 0) die("cannot stat '%s'", path.c_str());
        if (st.st_size % 2) die("corpus '%s' is not a uint16 stream (odd size)", path.c_str());
        uint64_t total = (uint64_t)st.st_size / 2;
        if (offset >= total) die("offset %" PRIu64 " beyond corpus of %" PRIu64 " tokens", offset, total);
        uint64_t avail = total - offset;
        n = tokens ? std::min<uint64_t>(tokens, avail) : avail;
        /* 2 MiB alignment so the mapping is page-friendly on both sides of the mmap. */
        uint64_t off_bytes = offset * 2;
        uint64_t page = 4096;
        uint64_t adjust = off_bytes % page;
        void* m = mmap(nullptr, (size_t)(n * 2 + adjust), PROT_READ, MAP_PRIVATE, fd,
                       (off_t)(off_bytes - adjust));
        if (m == MAP_FAILED) die("mmap of '%s' failed", path.c_str());
        tok = (const uint16_t*)m + adjust;
    }
    uint32_t at(uint64_t i) const { return tok[i]; }
};

/* ------------------------------------------------------------------ map */

struct CountMap {
    std::vector<Entry> slot;   // key == 0 -> empty
    size_t mask = 0, size = 0, limit = 0;
    bool overflowed = false;

    void init(size_t capacity) {
        size_t cap = 1;
        while (cap < capacity) cap <<= 1;
        slot.assign(cap, Entry{0, 0});
        mask = cap - 1;
        limit = (size_t)(cap * 0.70);
    }
    double gib() const { return (double)slot.size() * sizeof(Entry) / (1024.0 * 1024.0 * 1024.0); }

    bool grow() {
        size_t new_cap = slot.size() * 2;
        if (new_cap * sizeof(Entry) > 8.0 * 1024 * 1024 * 1024) return false;  // hard ceiling
        std::vector<Entry> old;
        old.swap(slot);
        slot.assign(new_cap, Entry{0, 0});
        mask = new_cap - 1;
        limit = (size_t)(new_cap * 0.70);
        for (auto& e : old) {
            if (!e.key) continue;
            size_t i = count_slot(e.key, mask);
            while (slot[i].key) i = (i + 1) & mask;
            slot[i] = e;
        }
        return true;
    }

    inline void add(uint64_t key, uint32_t delta) {
        if (key == 0) key = 1;   // 0 is the empty marker
        size_t i = count_slot(key, mask);
        while (true) {
            Entry& e = slot[i];
            if (e.key == key) { e.cnt += delta; return; }
            if (e.key == 0) {
                e.key = key; e.cnt = delta; ++size;
                if (size > limit) overflowed = true;
                return;
            }
            i = (i + 1) & mask;
        }
    }
};

/* ------------------------------------------------------------------ parallel sort */

static void parallel_sort(std::vector<Entry>& v, int threads) {
    const size_t n = v.size();
    if (n < 2) return;
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    size_t chunk = (n + threads - 1) / threads;
    if (chunk < 4096) chunk = 4096;
    std::vector<size_t> bounds;
    for (size_t b = 0; b < n; b += chunk) bounds.push_back(std::min(b + chunk, n));
#pragma omp parallel for schedule(dynamic)
    for (size_t b = 0; b < bounds.size(); ++b) {
        size_t lo = b ? bounds[b - 1] : 0;
        std::sort(v.begin() + lo, v.begin() + bounds[b],
                  [](const Entry& a, const Entry& b) { return a.key < b.key; });
    }
    /* bottom-up merge into a scratch buffer, ping-ponging so no stage is O(n log n) */
    std::vector<Entry> scratch(n);
    std::vector<Entry>* src = &v;
    std::vector<Entry>* dst = &scratch;
    for (size_t width = chunk; width < n; width *= 2) {
#pragma omp parallel for schedule(dynamic)
        for (size_t lo = 0; lo < n; lo += 2 * width) {
            size_t mid = std::min(lo + width, n);
            size_t hi = std::min(lo + 2 * width, n);
            Entry* out = dst->data() + lo;
            std::merge(src->data() + lo, src->data() + mid, src->data() + mid, src->data() + hi, out,
                       [](const Entry& a, const Entry& b) { return a.key < b.key; });
        }
        std::swap(src, dst);
    }
    if (src != &v) std::copy(src->begin(), src->begin() + n, v.begin());
}

/* ------------------------------------------------------------------ one order */

struct OrderStats {
    uint64_t rows = 0;
    uint64_t distinct_pairs = 0;
    uint64_t dropped_hist = 0;
    uint64_t dropped_pair_floor = 0;
    uint64_t candidates = 0;
    uint64_t candidate_slots = 0;
    double log10_min = 0.0;
    double gib_map = 0.0;
    double seconds = 0.0;
};

static double log10_u32(uint32_t x) { return std::log10((double)x); }

/* Walk the sorted records grouped by history. select=true also writes the order's files.
   Both walks run the same deterministic top-k_store selection so that the quantization range
   (which needs the global minimum score) can be known before any byte is written. */
template <bool Write>
static OrderStats walk_order(const std::vector<Entry>& rec, const OrderSetting& st, int k_store,
                             double lo_fixed, const char* path_keys, const char* path_cands,
                             const char* path_codes, const char* path_hcnt) {
    OrderStats s;
    std::vector<uint32_t> cand_tok, cand_cnt;
    cand_tok.reserve(64);
    cand_cnt.reserve(64);
    std::vector<int32_t> out_cands;
    std::vector<uint8_t> out_codes;

    FILE* f_keys = nullptr;
    FILE* f_cands = nullptr;
    FILE* f_codes = nullptr;
    FILE* f_hcnt = nullptr;
    if (Write) {
        f_keys = fopen(path_keys, "wb");
        f_cands = fopen(path_cands, "wb");
        f_codes = fopen(path_codes, "wb");
        f_hcnt = fopen(path_hcnt, "wb");
        if (!f_keys || !f_cands || !f_codes || !f_hcnt) die("cannot write table files in out-dir");
    }

    uint64_t i = 0;
    const uint64_t n = rec.size();
    /* Walk 1 finds the global minimum retained log10 score; walk 2 writes with that range, so
       every row of the order shares one quantization range (the paper quantizes per n-gram size). */
    bool first_log = !Write;
    double lo = Write ? lo_fixed : 0.0;

    while (i < n) {
        const uint64_t hist = count_pair_hist(rec[i].key);
        uint64_t j = i;
        uint64_t hist_count = 0;
        cand_tok.clear();
        cand_cnt.clear();
        while (j < n && count_pair_hist(rec[j].key) == hist) {
            uint32_t cnt = rec[j].cnt;
            hist_count += cnt;
            ++s.distinct_pairs;
            if (cnt >= st.pair_min_count) {
                cand_tok.push_back(count_pair_tok(rec[j].key));
                cand_cnt.push_back(cnt);
            } else {
                ++s.dropped_pair_floor;
            }
            ++j;
        }
        i = j;

        if (hist_count < st.history_min_count) { ++s.dropped_hist; continue; }
        if (cand_tok.empty()) continue;

        size_t k = cand_tok.size();
        if (k > (size_t)k_store) {
            /* Same denominator per row, so the top-k_store by score is the top-k_store by count.
               Tie-break on token id so walk 1 and walk 2 pick the identical set. */
            std::vector<uint32_t> idx(k);
            for (size_t t = 0; t < k; ++t) idx[t] = t;
            std::nth_element(idx.begin(), idx.begin() + k_store, idx.end(), [&](uint32_t a, uint32_t b) {
                if (cand_cnt[a] != cand_cnt[b]) return cand_cnt[a] > cand_cnt[b];
                return cand_tok[a] < cand_tok[b];
            });
            idx.resize(k_store);
            std::sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) {
                if (cand_cnt[a] != cand_cnt[b]) return cand_cnt[a] > cand_cnt[b];
                return cand_tok[a] < cand_tok[b];
            });
            std::vector<uint32_t> t2(k_store), c2(k_store);
            for (int t = 0; t < k_store; ++t) { t2[t] = cand_tok[idx[t]]; c2[t] = cand_cnt[idx[t]]; }
            cand_tok.swap(t2);
            cand_cnt.swap(c2);
            k = k_store;
        }

        if (!Write) {
            double min_log = log10_u32(cand_cnt[k - 1]) - log10_u32((uint32_t)hist_count);
            if (first_log || min_log < lo) { lo = min_log; first_log = false; }
        }

        ++s.rows;
        s.candidates += k;
        s.candidate_slots += k_store;

        if (Write) {
            uint64_t key = hist;
            fwrite(&key, sizeof(key), 1, f_keys);
            uint32_t hc = (uint32_t)(hist_count > 0xFFFFFFFFull ? 0xFFFFFFFFull : hist_count);
            fwrite(&hc, sizeof(hc), 1, f_hcnt);
            double range = -lo;   /* 0 - lo */
            for (size_t t = 0; t < k; ++t) {
                double l10 = log10_u32(cand_cnt[t]) - log10_u32(hc);
                int code = 255;
                if (range > 0.0) {
                    double q = (l10 - lo) * 255.0 / range;
                    code = (int)std::lround(q);
                    if (code < 0) code = 0;
                    if (code > 255) code = 255;
                }
                out_codes.push_back((uint8_t)code);
                out_cands.push_back((int32_t)cand_tok[t]);
            }
            for (size_t t = k; t < (size_t)k_store; ++t) {
                out_cands.push_back(COUNT_INVALID_TOKEN);
                out_codes.push_back(0);
            }
            fwrite(out_cands.data(), sizeof(int32_t), out_cands.size(), f_cands);
            fwrite(out_codes.data(), sizeof(uint8_t), out_codes.size(), f_codes);
            out_cands.clear();
            out_codes.clear();
        }
    }

    s.log10_min = first_log ? 0.0 : lo;

    if (Write) {
        fclose(f_keys);
        fclose(f_cands);
        fclose(f_codes);
        fclose(f_hcnt);
    }
    return s;
}

static void count_order(const Corpus& corpus, int order, CountMap& map) {
    const uint64_t n = corpus.n;
    if (n <= (uint64_t)order) return;
    for (uint64_t t = 0; t + 1 < n; ++t) {
        if (t + 1 < (uint64_t)order) continue;      /* not enough left context for this order */
        uint64_t h = count_hist1(corpus.at(t));
        for (int j = 1; j < order; ++j) h = count_hist_extend(h, corpus.at(t - j));
        map.add(count_pair_key(h, corpus.at(t + 1)), 1);
        if (map.overflowed) {
            map.overflowed = false;
            if (!map.grow()) die("counting map for order %d exceeded its ceiling; lower --tokens, "
                                 "raise --memory-gib, or lower --max-order", order);
        }
    }
}

/* ------------------------------------------------------------------ unigram row (S(w | empty)) */

struct Unigram {
    std::vector<uint32_t> ids;
    std::vector<uint8_t> codes;
    double log10_min = 0.0;
};

/* ------------------------------------------------------------------ main */

int main(int argc, char** argv) {
    Options opt;
    parse_args(argc, argv, opt);

    Corpus corpus;
    corpus.open(opt.corpus, opt.offset, opt.tokens);
    fprintf(stderr, "count_build: corpus %s tokens=%" PRIu64 " offset=%" PRIu64 " vocab=%u max_order=%d k_store=%d\n",
            opt.corpus.c_str(), corpus.n, opt.offset, opt.vocab, opt.max_order, opt.k_store);

    /* Unigram counts: the backoff floor of Eq. 2 (S(w | empty)) and the sanity check on --vocab. */
    std::vector<uint64_t> uni(opt.vocab, 0);
    uint32_t max_tok = 0;
    uint64_t total = 0;
    for (uint64_t t = 0; t < corpus.n; ++t) {
        uint32_t x = corpus.at(t);
        if (x >= opt.vocab) die("corpus contains token %u >= vocab %u", x, opt.vocab);
        ++uni[x];
        ++total;
        if (x > max_tok) max_tok = x;
    }
    fprintf(stderr, "count_build: unigrams over %" PRIu64 " tokens, max token id %u\n", total, max_tok);

    /* mkdir -p: a table directory is usually count_tables/<name> */
    {
        std::string path;
        for (char c : opt.out_dir) {
            path.push_back(c);
            if (c == '/' && path != "/") mkdir(path.c_str(), 0775);
        }
        mkdir(opt.out_dir.c_str(), 0775);
    }
    if (access(opt.out_dir.c_str(), W_OK) != 0) die("out-dir '%s' is not writable", opt.out_dir.c_str());

    std::vector<OrderStats> stats(opt.max_order + 1);
    std::vector<uint64_t> rows(opt.max_order + 1, 0);
    std::vector<double> log10_min(opt.max_order + 1, 0.0);

    /* Capacity guess: distinct (h,w) pairs are bounded by the number of positions, and for
       order >= 4 they are very close to it. Starting small and growing is cheaper than
       allocating the bound up front for orders 1-3. */
    static const double factor[] = {0, 0.02, 0.12, 0.45, 0.80, 0.90, 0.95, 0.96, 0.97, 0.97, 0.98, 0.98, 0.98};
    for (int o = 1; o <= opt.max_order; ++o) {
        uint64_t positions = corpus.n > (uint64_t)o ? corpus.n - (uint64_t)o : 0;
        double f = factor[std::min(o, 12)];
        size_t cap = 1;
        while ((double)cap * 0.70 < (double)positions * f) cap <<= 1;
        double gib = (double)cap * sizeof(Entry) / (1024.0 * 1024.0 * 1024.0);
        fprintf(stderr, "count_build: order %d initial map %.2f GiB (positions %" PRIu64 ")\n", o, gib, positions);
        if (opt.dry_run) continue;
        if (gib > opt.memory_gib)
            die("order %d needs %.2f GiB but --memory-gib is %.2f. Lower --tokens/--max-order or "
                "raise --memory-gib.", o, gib, opt.memory_gib);
    }
    if (opt.dry_run) {
        printf("dry run: no files written\n");
        return 0;
    }

    for (int o = 1; o <= opt.max_order; ++o) {
        auto t0 = std::chrono::steady_clock::now();
        CountMap map;
        uint64_t positions = corpus.n > (uint64_t)o ? corpus.n - (uint64_t)o : 0;
        double f = factor[std::min(o, 12)];
        size_t cap = 1;
        while ((double)cap * 0.70 < (double)positions * f) cap <<= 1;
        map.init(cap);
        count_order(corpus, o, map);

        /* Compact live slots to the front of the same buffer: no second allocation. */
        std::vector<Entry>& rec = map.slot;
        size_t w = 0;
        for (size_t r = 0; r < rec.size(); ++r) {
            if (rec[r].key) rec[w++] = rec[r];
        }
        rec.resize(w);
        map.size = w;
        map.slot.shrink_to_fit();

        parallel_sort(rec, opt.threads);

        char pkeys[512], pcands[512], pcodes[512], phcnt[512];
        snprintf(pkeys, sizeof(pkeys), "%s/order%d.keys.bin", opt.out_dir.c_str(), o);
        snprintf(pcands, sizeof(pcands), "%s/order%d.cands.bin", opt.out_dir.c_str(), o);
        snprintf(pcodes, sizeof(pcodes), "%s/order%d.codes.bin", opt.out_dir.c_str(), o);
        snprintf(phcnt, sizeof(phcnt), "%s/order%d.hcnt.bin", opt.out_dir.c_str(), o);

        OrderStats pre = walk_order<false>(rec, opt.settings[o], opt.k_store, 0.0, nullptr, nullptr,
                                           nullptr, nullptr);
        OrderStats s = walk_order<true>(rec, opt.settings[o], opt.k_store, pre.log10_min, pkeys,
                                        pcands, pcodes, phcnt);
        s.gib_map = map.gib();
        s.distinct_pairs = w;
        s.log10_min = pre.log10_min;
        auto t1 = std::chrono::steady_clock::now();
        s.seconds = std::chrono::duration<double>(t1 - t0).count();
        rows[o] = s.rows;
        log10_min[o] = s.log10_min;
        stats[o] = s;

        fprintf(stderr,
                "count_build: order %d pairs=%" PRIu64 " rows=%" PRIu64 " cands=%.1f/row dropped_hist=%" PRIu64 " "
                "dropped_pair_floor=%" PRIu64 " map=%.2f GiB log10_min=%.3f  %.1fs\n",
                o, s.distinct_pairs, s.rows, s.rows ? (double)s.candidates / (double)s.rows : 0.0,
                s.dropped_hist, s.dropped_pair_floor, s.gib_map, s.log10_min, s.seconds);
    }

    /* unigram top-k_store: S(w | empty) = C(w) / sum C(w), same 8-bit quantization scheme */
    std::vector<std::pair<uint64_t, uint32_t>> uni_pairs;
    uni_pairs.reserve(opt.vocab);
    for (uint32_t v = 0; v < opt.vocab; ++v) if (uni[v]) uni_pairs.push_back({uni[v], v});
    std::sort(uni_pairs.begin(), uni_pairs.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    });
    int uk = std::min<int>((int)uni_pairs.size(), opt.k_store);
    Unigram unirow;
    double lo_uni = 0.0;
    if (uk > 0) lo_uni = std::log10((double)uni_pairs[uk - 1].first / (double)total);
    for (int t = 0; t < uk; ++t) {
        double l10 = std::log10((double)uni_pairs[t].first / (double)total);
        int code = 255;
        if (-lo_uni > 0.0) {
            code = (int)std::lround((l10 - lo_uni) * 255.0 / (-lo_uni));
            if (code < 0) code = 0;
            if (code > 255) code = 255;
        }
        unirow.ids.push_back(uni_pairs[t].second);
        unirow.codes.push_back((uint8_t)code);
    }

    /* meta.bin: the reader's source of truth (see count_common.h for the layout) */
    std::string meta_bin = opt.out_dir + "/meta.bin";
    FILE* f = fopen(meta_bin.c_str(), "wb");
    if (!f) die("cannot write %s", meta_bin.c_str());
    uint32_t magic = COUNT_MAGIC, version = COUNT_FORMAT_VERSION;
    fwrite(&magic, 4, 1, f);
    fwrite(&version, 4, 1, f);
    fwrite(&opt.vocab, 4, 1, f);
    uint32_t max_order_u = (uint32_t)opt.max_order;
    fwrite(&max_order_u, 4, 1, f);
    uint32_t k_store_u = (uint32_t)opt.k_store;
    fwrite(&k_store_u, 4, 1, f);
    fwrite(&max_order_u, 4, 1, f);   /* n_orders */
    fwrite(&opt.backoff, 8, 1, f);
    fwrite(&total, 8, 1, f);
    for (int o = 1; o <= opt.max_order; ++o) {
        uint32_t ou = (uint32_t)o;
        fwrite(&ou, 4, 1, f);
        uint64_t r = rows[o];
        fwrite(&r, 8, 1, f);
        double lmin = log10_min[o];
        fwrite(&lmin, 8, 1, f);
        fwrite(&opt.settings[o].history_min_count, 4, 1, f);
        fwrite(&opt.settings[o].pair_min_count, 4, 1, f);
        uint64_t dp = stats[o].distinct_pairs;
        fwrite(&dp, 8, 1, f);
    }
    fwrite(&lo_uni, 8, 1, f);
    uint32_t uk_u = (uint32_t)uk;
    fwrite(&uk_u, 4, 1, f);
    fwrite(unirow.ids.data(), 4, uk, f);
    fwrite(unirow.codes.data(), 1, uk, f);
    fclose(f);

    /* meta.json: the human-readable twin, used by count/table.py for reporting and sweeps */
    std::string meta_json = opt.out_dir + "/meta.json";
    f = fopen(meta_json.c_str(), "wb");
    if (!f) die("cannot write %s", meta_json.c_str());
    fprintf(f, "{\n");
    fprintf(f, "  \"format\": \"count-table-v1\",\n");
    fprintf(f, "  \"vocab_size\": %u,\n", opt.vocab);
    fprintf(f, "  \"max_order\": %d,\n", opt.max_order);
    fprintf(f, "  \"k_store\": %d,\n", opt.k_store);
    fprintf(f, "  \"backoff\": %.6f,\n", opt.backoff);
    fprintf(f, "  \"corpus\": {\"file\": \"%s\", \"offset\": %" PRIu64 ", \"tokens\": %" PRIu64 "},\n",
            opt.corpus.c_str(), opt.offset, total);
    fprintf(f, "  \"orders\": [\n");
    for (int o = 1; o <= opt.max_order; ++o) {
        double bytes = (double)rows[o] * (8.0 + 4.0 + (double)opt.k_store * (4.0 + 1.0));
        fprintf(f, "    {\"order\": %d, \"rows\": %" PRIu64 ", \"log10_min\": %.6f, \"history_min_count\": %u,"
                   " \"pair_min_count\": %u, \"distinct_pairs\": %" PRIu64 ", \"candidates_per_row\": %.3f,"
                   " \"bytes\": %.0f}%s\n",
                o, rows[o], log10_min[o], opt.settings[o].history_min_count, opt.settings[o].pair_min_count,
                stats[o].distinct_pairs,
                rows[o] ? (double)stats[o].candidates / (double)rows[o] : 0.0, bytes,
                o == opt.max_order ? "" : ",");
    }
    fprintf(f, "  ],\n");
    fprintf(f, "  \"unigram\": {\"tokens\": %" PRIu64 ", \"k\": %u, \"log10_min\": %.6f, \"ids\": [", total, uk, lo_uni);
    for (int t = 0; t < uk; ++t) fprintf(f, "%s%u", t ? "," : "", unirow.ids[t]);
    fprintf(f, "], \"codes\": [");
    for (int t = 0; t < uk; ++t) fprintf(f, "%s%u", t ? "," : "", unirow.codes[t]);
    fprintf(f, "]}\n");
    fprintf(f, "}\n");
    fclose(f);

    double total_bytes = 0, total_rows = 0;
    for (int o = 1; o <= opt.max_order; ++o) {
        total_rows += rows[o];
        total_bytes += (double)rows[o] * (8.0 + 4.0 + (double)opt.k_store * 5.0);
    }
    fprintf(stderr, "count_build: wrote %s  rows=%.1fM  table=%.2f GiB\n", opt.out_dir.c_str(),
            total_rows / 1e6, total_bytes / (1024.0 * 1024.0 * 1024.0));
    return 0;
}
