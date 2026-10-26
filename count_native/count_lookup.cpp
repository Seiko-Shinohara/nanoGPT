/*
 * libcount_lookup -- COUNT's host-resident lookup path.
 *
 * The paper's requirement (Sec. 3, Appendix B.2): the counts table lives in host RAM, is
 * read-only, and is searched by a compiled multithreaded lookup, so it never occupies GPU
 * memory and does not need to be written to at all. Python reaches this through ctypes
 * (count/table.py); ctypes releases the GIL around the call, so a prefetch thread can search
 * the next batch while the GPUs are busy with the current one.
 *
 * What a lookup returns for one position p of one sequence:
 *   the J longest STORED suffixes of x_1..p+1 (longest first, orders n_max down to 1), and for
 *   each of them the top-k_use candidates of Eq. 2's stupid-backoff score S(w | h), scored from
 *   the decoded table scores with 0.4^d discounts along the suffix chain.
 *   Rows that are absent (C(h) below the build-time history threshold) are skipped, so J is the
 *   number of orders that actually resolved and the model divides by that count.
 *
 * Stupid backoff (Eq. 2) is evaluated here rather than baked into the stored rows:
 *   S(w|h) = C(h,w)/C(h)              if C(h,w) > 0
 *            gamma * S(w | h[1:])     otherwise, gamma = 0.4
 * so the longest suffix in which w is observed wins, and every deeper level multiplies by gamma.
 * max_backoff_depth truncates that chain; depth n reaches S(w | empty), the unigram row the
 * builder stores in meta.bin. Truncating the chain is also what a pruned table does implicitly:
 * a history below the count threshold has no row, so its level contributes nothing.
 *
 * Table mixing (Sec. 5.1, training-free domain adaptation): pass two handles with weights
 * (1-lambda, lambda). For every order the two candidate sets are unioned and the scores are
 * interpolated as S_lambda(w|h) = (1-lambda) S_general(w|h) + lambda S_domain(w|h), with zero
 * contribution from a table that does not hold the candidate in its row, then the top k_use are
 * taken WITHOUT renormalizing -- the paper is explicit that count scores are never normalized.
 */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cinttypes>
#include <cstring>
#include <cmath>
#include <algorithm>
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

#ifndef M_LN10
#define M_LN10 2.30258509299404568402
#endif

namespace {

struct MmapFile {
    void* ptr = nullptr;
    size_t bytes = 0;
    bool open_read(const char* path) {
        int fd = ::open(path, O_RDONLY);
        if (fd < 0) return false;
        struct stat st;
        if (fstat(fd, &st) != 0) { ::close(fd); return false; }
        bytes = (size_t)st.st_size;
        /* No MAP_POPULATE on purpose: a lookup touches a few thousand random rows, not the
           whole table, so populating would thrash the page cache the table depends on. */
        ptr = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        return ptr != MAP_FAILED;
    }
    void close() { if (ptr) munmap(ptr, bytes); ptr = nullptr; }
};

struct OrderTable {
    MmapFile keys, cands, codes, hcnt;
    const uint64_t* key = nullptr;
    const int32_t* cand = nullptr;
    const uint8_t* code = nullptr;
    uint64_t rows = 0;
    int k_store = 0;
    double log10_min = 0.0;
    double lut[256];   /* 8-bit code -> natural-log score, decoded at read time (paper Sec. A.2) */
};

/* Small open-addressing buffers reused by every lookup on this thread.
   SET slots > 2 * the largest candidate set ((max_order+1) * k_store * n_tables) keeps the
   probe chains at ~1 step. */
struct Scratch {
    static const int SET = 4096;
    std::vector<int32_t> tok;
    std::vector<double> score;
    std::vector<int32_t> ord;
    std::vector<int32_t> s_tok, s_gen;     /* merged candidate set: slot -> index in tok[] */
    std::vector<int32_t> t_tok, t_gen;     /* per-(table,order) precedence set: slot -> token */
    uint32_t row_gen = 0, tab_gen = 0;
    int ncand = 0;
    int cap = 0;
    std::vector<int64_t> rows;             /* rows[table * (Lmax+1) + order] */
    std::vector<uint64_t> chain;           /* chain[order] history hash */

    void init(int max_cand, int n_tables, int lmax) {
        cap = max_cand + 16;
        tok.assign(cap, 0);
        score.assign(cap, 0.0);
        ord.assign(cap, 0);
        s_tok.assign(SET, -1); s_gen.assign(SET, 0);
        t_tok.assign(SET, -1); t_gen.assign(SET, 0);
        rows.assign((size_t)(n_tables + 1) * (size_t)(lmax + 2), -1);
        chain.assign(lmax + 2, 0);
    }
    void begin_row() { ++row_gen; ++tab_gen; ncand = 0; }
    void begin_table() { ++tab_gen; }

    inline void add_or_accumulate(int32_t token, double s) {
        if (ncand >= cap) return;   /* cannot happen with a correctly sized cap */
        size_t i = ((uint32_t)token * 2654435761u) & (size_t)(SET - 1);
        while (true) {
            if (s_gen[i] != (int32_t)row_gen) {
                s_gen[i] = (int32_t)row_gen; s_tok[i] = ncand;
                tok[ncand] = token; score[ncand] = s; ++ncand; return;
            }
            if (tok[s_tok[i]] == token) { score[s_tok[i]] += s; return; }
            i = (i + 1) & (size_t)(SET - 1);
        }
    }
    /* true when this table already used this token at a longer suffix (Eq. 2 precedence) */
    inline bool table_seen_before(int32_t token) {
        size_t i = ((uint32_t)token * 2654435761u) & (size_t)(SET - 1);
        while (true) {
            if (t_gen[i] != (int32_t)tab_gen) { t_gen[i] = (int32_t)tab_gen; t_tok[i] = token; return false; }
            if (t_tok[i] == token) return true;
            i = (i + 1) & (size_t)(SET - 1);
        }
    }
};

int64_t find_row(const OrderTable& t, uint64_t key) {
    const uint64_t* k = t.key;
    uint64_t lo = 0, hi = t.rows;
    while (lo < hi) {
        uint64_t mid = (lo + hi) >> 1;
        if (k[mid] < key) lo = mid + 1; else hi = mid;
    }
    if (lo < t.rows && k[lo] == key) return (int64_t)lo;
    return -1;
}

}  // namespace

struct CountHandle {
    int max_order = 0;
    int k_store = 0;
    int vocab = 0;
    double backoff = 0.4;
    uint64_t corpus_tokens = 0;
    std::vector<OrderTable> order;         /* order[n], n = 1..max_order */
    std::vector<int32_t> uni_ids;
    std::vector<double> uni_scores;        /* S(w | empty), decoded */
    std::string dir;
};

/* ------------------------------------------------------------------ open / close */

extern "C" void cl_close(CountHandle* h);   /* cl_open's failure paths use it */

static std::string order_path(const std::string& dir, int order, const char* suffix) {
    char buf[512];
    snprintf(buf, sizeof(buf), "%s/order%d.%s", dir.c_str(), order, suffix);
    return std::string(buf);
}

extern "C" CountHandle* cl_open(const char* dir) {
    std::string meta = std::string(dir) + "/meta.bin";
    FILE* f = fopen(meta.c_str(), "rb");
    if (!f) { fprintf(stderr, "cl_open: cannot open %s\n", meta.c_str()); return nullptr; }

    uint32_t magic, version, vocab, max_order, k_store, n_orders;
    double backoff;
    uint64_t corpus_tokens;
    if (fread(&magic, 4, 1, f) != 1 || magic != COUNT_MAGIC) {
        fprintf(stderr, "cl_open: %s is not a count table (magic %08x)\n", meta.c_str(), magic);
        fclose(f); return nullptr;
    }
    fread(&version, 4, 1, f);
    fread(&vocab, 4, 1, f);
    fread(&max_order, 4, 1, f);
    fread(&k_store, 4, 1, f);
    fread(&n_orders, 4, 1, f);
    fread(&backoff, 8, 1, f);
    fread(&corpus_tokens, 8, 1, f);
    if (version != COUNT_FORMAT_VERSION || max_order > COUNT_MAX_ORDER_LIMIT) {
        fprintf(stderr, "cl_open: unsupported table version %u / max_order %u\n", version, max_order);
        fclose(f); return nullptr;
    }

    CountHandle* h = new CountHandle();
    h->dir = dir;
    h->vocab = (int)vocab;
    h->max_order = (int)max_order;
    h->k_store = (int)k_store;
    h->backoff = backoff;
    h->corpus_tokens = corpus_tokens;
    h->order.assign(max_order + 1, OrderTable());

    bool ok = true;
    for (uint32_t o = 1; o <= max_order && ok; ++o) {
        uint32_t ou; uint64_t rows, distinct; double log10_min; uint32_t hmin, pmin;
        fread(&ou, 4, 1, f); fread(&rows, 8, 1, f); fread(&log10_min, 8, 1, f);
        fread(&hmin, 4, 1, f); fread(&pmin, 4, 1, f); fread(&distinct, 8, 1, f);
        OrderTable& t = h->order[o];
        t.rows = rows;
        t.k_store = (int)k_store;
        t.log10_min = log10_min;
        double decay = (log10_min < 0.0) ? (-log10_min / 255.0) : 0.0;
        for (int c = 0; c < 256; ++c) t.lut[c] = std::exp(M_LN10 * (log10_min + c * decay));
        std::string pk = order_path(dir, (int)o, "keys.bin");
        std::string pc = order_path(dir, (int)o, "cands.bin");
        std::string pq = order_path(dir, (int)o, "codes.bin");
        std::string ph = order_path(dir, (int)o, "hcnt.bin");
        ok = t.keys.open_read(pk.c_str()) && t.cands.open_read(pc.c_str()) && t.codes.open_read(pq.c_str());
        if (!ok) { fprintf(stderr, "cl_open: missing %s\n", pk.c_str()); break; }
        t.hcnt.open_read(ph.c_str());   /* optional: analysis only */
        t.key = (const uint64_t*)t.keys.ptr;
        t.cand = (const int32_t*)t.cands.ptr;
        t.code = (const uint8_t*)t.codes.ptr;
        /* the builder writes an empty order file when a whole order was pruned away */
        if (t.keys.bytes < t.rows * 8 || t.cands.bytes < t.rows * 4 * (size_t)k_store) {
            fprintf(stderr, "cl_open: order %d truncated (%" PRIu64 " rows vs %zu bytes)\n", o, t.rows, t.keys.bytes);
            ok = false;
        }
    }

    double uni_log10_min; uint32_t uni_k;
    if (ok) fread(&uni_log10_min, 8, 1, f); else uni_log10_min = 0.0;
    if (ok) fread(&uni_k, 4, 1, f); else uni_k = 0;
    if (ok && uni_k > 0) {
        std::vector<int32_t> ids(uni_k);
        std::vector<uint8_t> codes(uni_k);
        fread(ids.data(), 4, uni_k, f);
        fread(codes.data(), 1, uni_k, f);
        double decay = (uni_log10_min < 0.0) ? (-uni_log10_min / 255.0) : 0.0;
        for (uint32_t i = 0; i < uni_k; ++i) {
            h->uni_ids.push_back(ids[i]);
            h->uni_scores.push_back(std::exp(M_LN10 * (uni_log10_min + codes[i] * decay)));
        }
    }
    fclose(f);
    if (!ok) { cl_close(h); return nullptr; }
    return h;
}

extern "C" void cl_close(CountHandle* h) {
    if (!h) return;
    for (auto& t : h->order) { t.keys.close(); t.cands.close(); t.codes.close(); t.hcnt.close(); }
    delete h;
}

extern "C" int cl_max_order(CountHandle* h) { return h ? h->max_order : 0; }
extern "C" int cl_k_store(CountHandle* h) { return h ? h->k_store : 0; }
extern "C" int cl_vocab(CountHandle* h) { return h ? h->vocab : 0; }
extern "C" double cl_backoff(CountHandle* h) { return h ? h->backoff : 0.0; }
extern "C" uint64_t cl_corpus_tokens(CountHandle* h) { return h ? h->corpus_tokens : 0; }
extern "C" uint64_t cl_rows(CountHandle* h, int order) {
    if (!h || order < 1 || order > h->max_order) return 0;
    return h->order[order].rows;
}
extern "C" double cl_log10_min(CountHandle* h, int order) {
    if (!h || order < 1 || order > h->max_order) return 0.0;
    return h->order[order].log10_min;
}
extern "C" int32_t cl_history_count(CountHandle* h, int order, int64_t row) {
    if (!h || order < 1 || order > h->max_order) return -1;
    OrderTable& t = h->order[order];
    if (!t.hcnt.ptr || row < 0 || (uint64_t)row >= t.rows) return -1;
    return (int32_t)((const uint32_t*)t.hcnt.ptr)[row];
}
extern "C" void cl_set_threads(int n) {
#ifdef _OPENMP
    if (n > 0) omp_set_num_threads(n);
#endif
}

/* ------------------------------------------------------------------ core lookup */

/*
 * One position. `seq` is the sequence row (chronological), `p` the position whose continuation
 * is being looked up, `left_ctx` how many tokens of left context exist inside this row.
 * out_cands/out_scores point at this position's j_max * k_use slot; out_nfound at its count.
 */
static void lookup_position(CountHandle** tabs, int n_tables, const double* weights,
                            const uint16_t* seq, int64_t p, int64_t left_ctx, int lmax,
                            int max_order_use, int j_max, int k_use, int max_backoff_depth,
                            Scratch& sc, int32_t* out_cands, float* out_scores, int32_t* out_nfound,
                            int32_t* out_orders) {
    int ctx = (int)std::min<int64_t>(left_ctx, lmax);
    /* max_order_use <= 0 means "no cap beyond what the tables store"; lmax already encodes that,
       and using the raw -1 in a std::min below silently disabled every lookup. */
    int order_cap = max_order_use > 0 ? max_order_use : lmax;
    /* History hashes, newest token first: chain[n] hashes the last n tokens of the context. */
    sc.chain[1] = count_hist1(seq[p]);
    for (int n = 2; n <= ctx; ++n) sc.chain[n] = count_hist_extend(sc.chain[n - 1], (uint32_t)seq[p - n + 1]);

    /* One binary search per (table, order); shorter orders are reused as backoff levels. */
    for (int t = 0; t < n_tables; ++t) {
        CountHandle* h = tabs[t];
        for (int n = 1; n <= lmax + 1; ++n) sc.rows[(size_t)t * (size_t)(lmax + 2) + n] = -1;
        int cap_n = std::min(std::min(h->max_order, order_cap), ctx);
        for (int n = 1; n <= cap_n; ++n)
            sc.rows[(size_t)t * (size_t)(lmax + 2) + n] = find_row(h->order[n], sc.chain[n]);
    }

    int jmax_all = std::min(order_cap, ctx);
    int used = 0;
    for (int n = jmax_all; n >= 1 && used < j_max; --n) {
        /* An order counts as resolved only if a table that actually carries weight has a row for
           it. A zero-weight table (lambda = 0 or 1) must not conjure a slot: "lambda = 0 uses the
           original table and lambda = 1 replaces it completely" (Sec. 5.1) is only true if the
           lookup also *selects* its J orders the way that one table would. */
        bool found_any = false;
        for (int t = 0; t < n_tables; ++t)
            if (weights[t] != 0.0 && sc.rows[(size_t)t * (size_t)(lmax + 2) + n] >= 0) { found_any = true; break; }
        if (!found_any) continue;

        sc.begin_row();
        int depth_limit = (max_backoff_depth < 0) ? n : std::min(n, max_backoff_depth);
        for (int t = 0; t < n_tables; ++t) {
            CountHandle* h = tabs[t];
            if (n > h->max_order) continue;
            const double wgt = weights[t];
            sc.begin_table();
            double disc = 1.0;
            for (int d = 0; d <= depth_limit; ++d) {
                int len = n - d;
                if (len == 0) {   /* S(w | empty): the unigram row */
                    for (size_t c = 0; c < h->uni_ids.size(); ++c) {
                        int32_t tok = h->uni_ids[c];
                        if (sc.table_seen_before(tok)) continue;
                        sc.add_or_accumulate(tok, wgt * disc * h->uni_scores[c]);
                    }
                    break;
                }
                int64_t row = sc.rows[(size_t)t * (size_t)(lmax + 2) + len];
                if (row >= 0) {
                    const OrderTable& ot = h->order[len];
                    const int32_t* cid = ot.cand + (size_t)row * (size_t)ot.k_store;
                    const uint8_t* ccode = ot.code + (size_t)row * (size_t)ot.k_store;
                    for (int c = 0; c < ot.k_store; ++c) {
                        int32_t tok = cid[c];
                        if (tok < 0) break;
                        if (sc.table_seen_before(tok)) continue;
                        sc.add_or_accumulate(tok, wgt * disc * ot.lut[ccode[c]]);
                    }
                }
                disc *= h->backoff;
            }
        }

        /* top-k_use of the merged set, no renormalization (paper Sec. 2.3 / 5.1) */
        /* A candidate whose *mixed* score is exactly 0 exists only because a second table brought
           it in with weight 0 (Sec. 5.1's "zero contribution from a table when a candidate is
           absent from its row", at lambda = 0 or 1). Eq. 1 sums score-weighted embedding rows, so
           such a candidate contributes literally nothing while occupying a K slot -- and it would
           make lambda = 0 differ from simply using the general table. Drop it: the row is then
           bit-identical to the single-table row. Stored scores are always > 0 (the quantization
           spans [log10_min, 0] and decodes to exp(ln10 * log10_min) .. 1.0), so a zero here can
           only come from a zero table weight. */
        int nc = 0;
        for (int i = 0; i < sc.ncand; ++i)
            if (sc.score[i] > 0.0) sc.ord[nc++] = i;
        int take = std::min(nc, k_use);
        if (nc > take) {
            /* Tie-break on token id inside the selection as well as the sort: rows are cut at
               exactly k_use candidates, and many candidates share a score (they came from the same
               backoff level), so an unstable selection would keep an arbitrary tied candidate.
               count.reference sorts every candidate by (-score, token), and the two must agree. */
            auto better = [&](int32_t a, int32_t b) {
                if (sc.score[a] != sc.score[b]) return sc.score[a] > sc.score[b];
                return sc.tok[a] < sc.tok[b];
            };
            std::nth_element(sc.ord.begin(), sc.ord.begin() + take, sc.ord.begin() + nc, better);
        }
        std::sort(sc.ord.begin(), sc.ord.begin() + take,
                  [&](int32_t a, int32_t b) {
                      if (sc.score[a] != sc.score[b]) return sc.score[a] > sc.score[b];
                      return sc.tok[a] < sc.tok[b];
                  });

        int32_t* oc = out_cands + (size_t)used * (size_t)k_use;
        float* os = out_scores + (size_t)used * (size_t)k_use;
        for (int i = 0; i < take; ++i) { oc[i] = sc.tok[sc.ord[i]]; os[i] = (float)sc.score[sc.ord[i]]; }
        for (int i = take; i < k_use; ++i) { oc[i] = COUNT_INVALID_TOKEN; os[i] = 0.0f; }
        if (out_orders) out_orders[used] = n;   /* which n-gram order this slot came from */
        ++used;
    }

    for (int j = used; j < j_max; ++j) {
        int32_t* oc = out_cands + (size_t)j * (size_t)k_use;
        float* os = out_scores + (size_t)j * (size_t)k_use;
        for (int i = 0; i < k_use; ++i) { oc[i] = COUNT_INVALID_TOKEN; os[i] = 0.0f; }
        if (out_orders) out_orders[j] = 0;
    }
    *out_nfound = used;
}

static int run_batch(CountHandle** tabs, int n_tables, const double* weights, const uint16_t* seq,
                     int n_rows, int seq_len, int max_order_use, int j_max, int k_use,
                     int max_backoff_depth, int32_t* out_cands, float* out_scores, int32_t* out_nfound,
                     int32_t* out_orders) {
    if (!tabs || n_tables < 1 || n_rows < 1 || seq_len < 1 || j_max < 1 || k_use < 1) return -1;
    int lmax = 0;
    for (int t = 0; t < n_tables; ++t) {
        if (!tabs[t]) return -1;
        lmax = std::max(lmax, tabs[t]->max_order);
    }
    if (max_order_use > 0) lmax = std::min(lmax, max_order_use);
    if (j_max > lmax) j_max = lmax;

    int max_cand = (lmax + 1) * 4096 + 32;   /* see Scratch::init; generous for k_store <= 4096 */
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    std::vector<Scratch> scratch((size_t)nthreads);
    for (auto& s : scratch) s.init(max_cand, n_tables, lmax);

#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < (int64_t)n_rows; ++r) {
#ifdef _OPENMP
        Scratch& sc = scratch[(size_t)omp_get_thread_num()];
#else
        Scratch& sc = scratch[0];
#endif
        const uint16_t* row = seq + (size_t)r * (size_t)seq_len;
        for (int p = 0; p < seq_len; ++p) {
            size_t base = ((size_t)r * (size_t)seq_len + (size_t)p) * (size_t)j_max * (size_t)k_use;
            lookup_position(tabs, n_tables, weights, row, p, (int64_t)p + 1, lmax, max_order_use,
                            j_max, k_use, max_backoff_depth, sc,
                            out_cands + base, out_scores + base,
                            out_nfound + (size_t)r * (size_t)seq_len + (size_t)p,
                            out_orders ? out_orders + base / (size_t)k_use : nullptr);
        }
    }
    return 0;
}

extern "C" int cl_lookup_mix(CountHandle** tabs, int n_tables, const double* weights,
                             const uint16_t* seq, int n_rows, int seq_len, int max_order_use,
                             int j_max, int k_use, int max_backoff_depth, int32_t* out_cands,
                             float* out_scores, int32_t* out_nfound, int32_t* out_orders) {
    return run_batch(tabs, n_tables, weights, seq, n_rows, seq_len, max_order_use, j_max, k_use,
                     max_backoff_depth, out_cands, out_scores, out_nfound, out_orders);
}

extern "C" int cl_lookup_batch(CountHandle* h, const uint16_t* seq, int n_rows, int seq_len,
                               int max_order_use, int j_max, int k_use, int max_backoff_depth,
                               int32_t* out_cands, float* out_scores, int32_t* out_nfound,
                               int32_t* out_orders) {
    double one = 1.0;
    return run_batch(&h, 1, &one, seq, n_rows, seq_len, max_order_use, j_max, k_use,
                     max_backoff_depth, out_cands, out_scores, out_nfound, out_orders);
}

/* Decode step: `tail` is the last up to lmax tokens, chronological, oldest first. */
extern "C" int cl_lookup_next_mix(CountHandle** tabs, int n_tables, const double* weights,
                                  const uint16_t* tail, int n_tail, int max_order_use, int j_max,
                                  int k_use, int max_backoff_depth, int32_t* out_cands,
                                  float* out_scores, int32_t* out_nfound, int32_t* out_orders) {
    if (!tabs || n_tables < 1 || n_tail < 1 || j_max < 1 || k_use < 1) return -1;
    int lmax = 0;
    for (int t = 0; t < n_tables; ++t) {
        if (!tabs[t]) return -1;
        lmax = std::max(lmax, tabs[t]->max_order);
    }
    if (max_order_use > 0) lmax = std::min(lmax, max_order_use);
    if (j_max > lmax) j_max = lmax;
    int ctx = std::min(n_tail, lmax);
    int max_cand = (lmax + 1) * 4096 + 32;
    Scratch sc;
    sc.init(max_cand, n_tables, lmax);
    lookup_position(tabs, n_tables, weights, tail + (n_tail - ctx), (int64_t)(ctx - 1), (int64_t)ctx,
                    lmax, max_order_use, j_max, k_use, max_backoff_depth, sc, out_cands, out_scores,
                    out_nfound, out_orders);
    return 0;
}

extern "C" int cl_lookup_next(CountHandle* h, const uint16_t* tail, int n_tail, int max_order_use,
                              int j_max, int k_use, int max_backoff_depth, int32_t* out_cands,
                              float* out_scores, int32_t* out_nfound, int32_t* out_orders) {
    double one = 1.0;
    return cl_lookup_next_mix(&h, 1, &one, tail, n_tail, max_order_use, j_max, k_use,
                              max_backoff_depth, out_cands, out_scores, out_nfound, out_orders);
}
