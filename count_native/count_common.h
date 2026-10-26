/*
 * Shared constants for the COUNT n-gram tables (Magnusson, Smith & Koh 2026,
 * "The COUNT: Counting N-grams Unlocks Scaling On Host Memory and Training-Free Adaptation").
 *
 * Both the builder (count_build.cpp) and the reader (count_lookup.cpp) must agree on:
 *   1. how an n-gram history becomes a 64-bit key,
 *   2. the on-disk layout of meta.bin,
 * so this header is the single source of truth for both.
 *
 * ---------------------------------------------------------------------------
 * Keys
 * ---------------------------------------------------------------------------
 * A history is the token window x_{t-n+1..t} ending at position t. Its key is a 48-bit
 * splitmix64 chain, built NEWEST token first:
 *
 *     H_1 = smix(x_t)
 *     H_n = smix(H_{n-1} * C1 ^ x_{t-n+1} * C2 ^ C3)      (n >= 2)
 *
 * The chain direction matters: H_2("ab") != H_2("ba"). 48 bits is a deliberate choice, not
 * an oversight: a pair key = (H_48 << 16) | next_token fits in exactly 64 bits, which lets the
 * counting map store a (history, continuation) pair in 12 bytes with no separate history array.
 * That is what makes a 100-200M-token order-6 build fit in a few GB on a 62 GB box.
 *
 * Collision budget (48-bit history hash, R stored rows, Q queries per pass):
 *   colliding stored pairs ~ R^2 / 2^49          (R = 1e7  -> 0.04 expected)
 *   false row hits         ~ R * Q / 2^48        (R = 1e7, Q = 1e8 -> 0.004 expected)
 * Both are far below "one wrong row out of 1e8 lookups", so hashed keys cost nothing here.
 * Infini-gram-style tables (which COUNT cites) make the same trade.
 *
 * ---------------------------------------------------------------------------
 * meta.bin layout (little endian, all fields packed)
 * ---------------------------------------------------------------------------
 *   header:  magic u32 'CTB1', version u32, vocab u32, max_order u32, k_store u32,
 *            n_orders u32, backoff f64, corpus_tokens u64
 *   per order (1..max_order): order u32, rows u64, log10_min f64,
 *                            history_min_count u32, pair_min_count u32, distinct_pairs u64
 *   unigram: uni_log10_min f64, uni_k u32, uni_ids i32[uni_k], uni_codes u8[uni_k]
 *
 * Data files (raw, no headers), one triple per order n:
 *   order{n}.keys.bin   u64 * rows            sorted ascending (binary-searchable)
 *   order{n}.cands.bin  i32 * rows * k_store  candidate token ids, -1 where the row is shorter
 *   order{n}.codes.bin  u8  * rows * k_store  8-bit quantized log10 score
 *   order{n}.hcnt.bin   u32 * rows            history count C(h) (analysis/ablation only)
 *
 * Scores are stored as in the paper: log10 S(w|h) quantized to 256 levels per order, spanning
 * [log10_min(order), 0], decoded to natural-log scores at read time and exponentiated. The
 * reader applies stupid backoff (Eq. 2) across suffixes, so a stored row only has to hold the
 * *observed* top-K continuations; the backed-off candidates are assembled at read time. That
 * keeps the table small and lets K, J and the lambda interpolation of Sec. 5.1 be changed at
 * inference time instead of at build time.
 */
#pragma once

#include <cstdint>
#include <cstddef>

#define COUNT_MAGIC 0x31425443u  /* 'C' 'T' 'B' '1' */
#define COUNT_FORMAT_VERSION 1u
#define COUNT_MAX_ORDER_LIMIT 12
#define COUNT_KEY_BITS 48
#define COUNT_KEY_MASK ((1ull << COUNT_KEY_BITS) - 1)
#define COUNT_INVALID_TOKEN (-1)

/* splitmix64 finalizer: cheap, well-mixed, no tables. */
static inline uint64_t count_smix(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/* Hash of a 1-token history. */
static inline uint64_t count_hist1(uint32_t x) {
    return count_smix((uint64_t)x + 0x243F6A8885A308D3ull) & COUNT_KEY_MASK;
}

/* Extend a history hash by prepending an older token. Not commutative (C1 != C2). */
static inline uint64_t count_hist_extend(uint64_t h, uint32_t older) {
    uint64_t z = h * 0x9E3779B97F4A7C15ull ^ ((uint64_t)older + 0x1234ull) * 0xD6E8FEB86659FD93ull
                 ^ 0xA0761D6478BD642Full;
    return count_smix(z) & COUNT_KEY_MASK;
}

/* Map key for the pair (history, continuation). 48-bit history + 16-bit continuation. */
static inline uint64_t count_pair_key(uint64_t hist_hash, uint32_t w) {
    return (hist_hash << 16) | (uint64_t)(w & 0xFFFFu);
}
static inline uint64_t count_pair_hist(uint64_t pair_key) { return pair_key >> 16; }
static inline uint32_t count_pair_tok(uint64_t pair_key) { return (uint32_t)(pair_key & 0xFFFFu); }

/* Second, independent avalanche: map slots must not be addressed by the same bits the key
   is compared with, or consecutive n-grams cluster in neighbouring buckets. */
static inline size_t count_slot(uint64_t key, size_t mask) {
    return (size_t)(count_smix(key) & (uint64_t)mask);
}
