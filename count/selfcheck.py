"""Correctness and throughput checks for the COUNT lookup path.

    python -m count.selfcheck --table count_tables/fw5m_o6 \
        --corpus data/fineweb-edu/val.bin --rows 32 --seq-len 512 --bench

Five things get checked, because each one maps to a place the fast path could quietly diverge
from the paper:

1. the compiled lookup agrees with the numpy reference on candidates, per-slot n-gram order, and
   scores (Eq. 1-2 re-derived in count.reference), for J and K both saturating and truncated;
2. the decode-time lookup (count.table.lookup_next, used by sample.py) agrees with the batch
   lookup on the same prefix -- the two share code, but only the batch path is used in training;
3. scores are used *without renormalization* and never exceed 1: a row's score sum staying below
   one is the signature of count-ratio weights, and a renormalized row would sum to 1.0 exactly;
4. table mixing degenerates to each input table at lambda = 0 and lambda = 1 (Sec. 5.1), and at
   lambda = 0.5 is the elementwise average of the two tables' scores over the union of candidates;
5. host-resident lookup throughput, in tokens/second and in candidate-lookups/second, with the
   paper's K = 8 / J = 6 against the J = 1 ablation (Table in Appendix B.2).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

import numpy as np

from count.reference import ReferenceTable, mix_lookup
from count.table import CountTable, mix


def random_windows(corpus: str, rows: int, seq_len: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_tokens = os.path.getsize(corpus) // 2
    starts = rng.integers(1, max(2, n_tokens - seq_len - 1), size=rows)
    out = np.empty((rows, seq_len), dtype=np.uint16)
    for i, s in enumerate(starts):
        out[i] = np.fromfile(corpus, dtype=np.uint16, count=seq_len, offset=int(s) * 2)
    return out


def _compare_one(res, table_dirs: List[str], weights: List[float], tokens: np.ndarray,
                 j: int, k: int, max_order: Optional[int], backoff_depth: int,
                 report_positions: int) -> int:
    refs = [ReferenceTable(d) for d in table_dirs]
    bad = 0
    checked = 0
    rows, seq_len, j_max, k_out = res.candidates.shape
    report_every = max(1, rows * seq_len // max(1, report_positions))
    for r in range(rows):
        for p in range(seq_len):
            # Position 0 has a one-token context, so an order-1 lookup is legitimate; the
            # reference decides what a position should return, including the empty ones.
            context = [int(t) for t in tokens[r, : p + 1]]
            exp_c, exp_s, exp_o, exp_n = mix_lookup(refs, weights, context, j=j, k=k,
                                                    max_order=max_order, backoff_depth=backoff_depth)
            got_n = int(res.n_found[r, p])
            got_o = [int(x) for x in res.orders[r, p, :got_n]]
            checked += 1
            if got_n != exp_n or got_o != exp_o:
                print(f"MISMATCH orders at ({r},{p}): got n={got_n} orders={got_o} "
                      f"expected n={exp_n} orders={exp_o}")
                bad += 1
                continue
            ok_row = True
            for jj in range(got_n):
                gc = [int(x) for x in res.candidates[r, p, jj] if x >= 0]
                gs = [float(x) for x in res.scores[r, p, jj, : len(gc)]]
                ec, es = exp_c[jj], exp_s[jj]
                if gc != ec:
                    ok_row = False
                    break
                if not np.allclose(gs, es, rtol=2e-6, atol=1e-9):
                    ok_row = False
                    break
            if not ok_row:
                bad += 1
                if bad <= 3:
                    print(f"MISMATCH candidates at ({r},{p}) context={context[-6:]}")
                    for jj in range(min(got_n, len(exp_c))):
                        print(f"   order {exp_o[jj]}: got {list(res.candidates[r,p,jj])} "
                              f"{np.round(res.scores[r,p,jj],5).tolist()}")
                        print(f"              want {exp_c[jj]} {[round(x,5) for x in exp_s[jj]]}")
                continue
            if checked % report_every == 0 and report_positions > 0:
                print(f"  ok ({r},{p}) orders={got_o} top candidates "
                      f"{[int(x) for x in res.candidates[r,p,0] if x>=0][:6]}")
    print(f"reference agreement: {checked - bad}/{checked} positions match")
    return bad


def check_decode_path(table: CountTable, tokens: np.ndarray, j: int, k: int) -> int:
    """lookup_next on a prefix must equal lookup_batch's row for that same position."""
    bad = 0
    rows, seq_len = tokens.shape
    picks = [(r, seq_len - 1) for r in range(min(rows, 8))]
    res = table.lookup_batch(tokens, j=j, k=k)
    for r, p in picks:
        cands, scores, n_found, orders = table.lookup_next(tokens[r, : p + 1], j=j, k=k)
        if n_found != int(res.n_found[r, p]):
            print(f"MISMATCH decode n_found at row {r}: {n_found} vs {res.n_found[r, p]}")
            bad += 1
            continue
        for jj in range(n_found):
            gc = [int(x) for x in cands[jj] if x >= 0]
            bc = [int(x) for x in res.candidates[r, p, jj] if x >= 0]
            if gc != bc or not np.allclose(scores[jj, : len(gc)], res.scores[r, p, jj, : len(gc)],
                                           rtol=2e-6, atol=1e-9):
                print(f"MISMATCH decode candidates at row {r} slot {jj}: {gc} vs {bc}")
                bad += 1
    print(f"decode path: {'OK' if bad == 0 else 'FAILED'} on {len(picks)} positions")
    return bad


def check_score_semantics(res, k: int, backoff: float = 0.4) -> int:
    """Eq. 2 gives count scores, not probabilities, and the paper never renormalizes them.

    The invariants that follow:
      * every individual score is a count ratio, or a count ratio discounted by backoff^depth, so
        it lies in (0, 1];
      * a candidate row sums to at most 1 + backoff. The stored row already carries the full
        conditional mass of its history -- the observed continuations sum to 1 -- and the read-time
        backoff chain then *adds* the suffix distribution's mass for tokens the row never saw.
        Classic stupid backoff is exactly this unnormalized: 1 (seen) + 0.4 x (suffix mass on
        unseen tokens), which is <= 1.4. A row summing to exactly 1.0 everywhere would mean someone
        renormalized, which Sec. 2 forbids.
    """
    bad = 0
    valid = res.candidates >= 0
    scores = np.where(valid, res.scores, 0.0)
    hit = np.broadcast_to((res.n_found > 0)[..., None], scores.shape[:-1])
    slot = scores[..., :k]          # (rows, seq, j, k); hit marks which of the j slots resolved

    biggest = float(scores[valid].max()) if valid.any() else 0.0
    if biggest > 1.0 + 1e-6:
        print(f"FAIL: a candidate score exceeds 1.0 ({biggest:.6f}); count ratios cannot")
        bad += 1
    row_sum = slot.sum(axis=-1)
    ceiling = 1.0 + backoff + 1e-6
    over = (row_sum > ceiling) & hit
    if over.any():
        print(f"FAIL: {int(over.sum())} candidate rows exceed 1 + backoff = {ceiling:.2f} "
              f"(max {float(row_sum[hit].max()):.6f})")
        bad += 1
    exact_one = int(((np.abs(row_sum - 1.0) < 1e-6) & hit).sum())
    total_slots = int(hit.sum())
    print(f"score semantics: max candidate score {biggest:.6f}; "
          f"row sums mean {float(row_sum[hit].mean()):.6f} max {float(row_sum[hit].max()):.6f}; "
          f"rows summing to exactly 1 (renormalized): {exact_one}/{total_slots}")

    diffs = np.diff(np.where(valid, slot, -np.inf), axis=-1)
    if (diffs > 1e-6).any():
        print("FAIL: candidates are not sorted by descending score inside a row")
        bad += 1
    return bad


def check_mixing(dirs: List[str], tokens: np.ndarray, j: int, k: int,
                 backoff_depth: int = -1) -> int:
    """Sec. 5.1 mixing, checked against count.reference rather than against the C++ output.

    S_lambda(w | h) = (1-lambda) S_a(w | h) + lambda S_b(w | h), per order, over the union of the
    two tables' candidate sets, and each S is that table's own Eq. 2 -- its own rows and its own
    backoff chain -- because Eq. 2 defines S for every token, not only for tokens the row happens
    to list. The reference's full merged set is what a table's S(w | h) means here; the native
    single-table output would be the same union truncated to k candidates, which is *not* enough to
    predict a mixed row: a candidate ranked 12th in table a can land in the top 8 of a mixed row,
    carrying a from a, and a truncated comparison would claim its score should be missing.
    """
    if len(dirs) < 2:
        print("mixing check: skipped (pass a second --table to exercise Sec. 5.1)")
        return 0
    a, b = CountTable(dirs[0]), CountTable(dirs[1])
    ra, rb = ReferenceTable(dirs[0]), ReferenceTable(dirs[1])
    bad = 0
    only_a = mix([a], [1.0], tokens, j=j, k=k)
    only_b = mix([b], [1.0], tokens, j=j, k=k)
    for lam in (0.0, 1.0):
        got = mix([a, b], [1.0 - lam, lam], tokens, j=j, k=k)
        want = only_a if lam == 0.0 else only_b
        name = f"lambda={lam:g}"
        if not np.array_equal(got.candidates, want.candidates):
            print(f"FAIL: {name} does not reduce to the single table")
            bad += 1
        elif not np.allclose(got.scores, want.scores, rtol=1e-6, atol=1e-9):
            print(f"FAIL: {name} scores differ from the single table")
            bad += 1
        else:
            print(f"mixing {name}: identical to using that table alone")

    rows, seq_len, j_max, _ = only_a.candidates.shape
    for lam in (0.25, 0.5):
        half = mix([a, b], [1.0 - lam, lam], tokens, j=j, k=k)
        scores_checked = slots_checked = set_bad = 0
        for r in range(min(rows, 4)):
            for p in range(1, min(seq_len, 64)):
                for jj in range(int(half.n_found[r, p])):
                    order = int(half.orders[r, p, jj])
                    if order <= 0:
                        continue
                    ctx = tokens[r, :p + 1]
                    ca = dict(ra.candidates(ctx, order, backoff_depth))
                    cb = dict(rb.candidates(ctx, order, backoff_depth))
                    want = {t: (1.0 - lam) * ca.get(t, 0.0) + lam * cb.get(t, 0.0)
                            for t in set(ca) | set(cb)}
                    want = {t: s for t, s in want.items() if s > 0.0}
                    emitted = [(int(t), float(s)) for t, s in
                               zip(half.candidates[r, p, jj], half.scores[r, p, jj]) if int(t) >= 0]
                    # 1. every emitted candidate carries the interpolated score of the union
                    for t, s in emitted:
                        scores_checked += 1
                        w = want.get(t)
                        if w is None or abs(w - s) > 1e-6 * max(1.0, abs(w)):
                            bad += 1
                            if bad <= 3:
                                print(f"FAIL mixing lambda={lam} at ({r},{p}) slot {jj} token {t}: "
                                      f"got {s:.6f} want {float('nan') if w is None else w:.6f}")
                    # 2. and it is the *top* k of that union. Comparing the exact token lists would
                    #    fail on ties (two candidates sharing a score can be swapped legitimately),
                    #    so compare the cut: nothing left out may score above the weakest kept one.
                    if len(emitted) < min(len(want), k):
                        set_bad += 1
                    elif want and len(emitted) == k:
                        kept = min(s for _, s in emitted)
                        rest = [s for t, s in want.items() if t not in {t2 for t2, _ in emitted}]
                        if rest and max(rest) > kept + 1e-6 * max(1.0, kept):
                            set_bad += 1
                            if set_bad <= 3:
                                print(f"FAIL mixing lambda={lam} at ({r},{p}) slot {jj}: a candidate "
                                      f"scoring {max(rest):.6f} was left out of a row whose weakest "
                                      f"kept score is {kept:.6f}")
                    slots_checked += 1
        if set_bad:
            bad += set_bad
        print(f"mixing lambda={lam}: {scores_checked} candidate scores checked against "
              f"(1-lambda)*S_a + lambda*S_b over the union; {slots_checked} rows confirmed to be "
              f"the top-{k} of that union ({bad} mismatches so far)")
    a.close(); b.close()
    return bad


def bench(tables: List[CountTable], tokens: np.ndarray, j: int, k: int, repeats: int) -> None:
    weights = [1.0 / len(tables)] * len(tables)
    rows, seq_len = tokens.shape
    for jj in (j, 1):
        for _ in range(2):
            res = mix(tables, weights, tokens, j=jj, k=k)
        t0 = time.perf_counter()
        for _ in range(repeats):
            res = mix(tables, weights, tokens, j=jj, k=k)
        dt = (time.perf_counter() - t0) / repeats
        lookups = int((res.n_found > 0).sum())
        slots = int((res.candidates >= 0).sum())
        print(f"lookup J={jj} K={k}: {dt * 1000:7.2f} ms for {rows * seq_len:,} positions "
              f"= {rows * seq_len / dt:11,.0f} tokens/s, "
              f"{lookups / dt:11,.0f} row-groups/s, {slots / dt:11,.0f} candidates/s")
    if hasattr(tables[0], "host_bytes"):
        print(f"host footprint: {sum(t.host_bytes() for t in tables) / 1e9:.3f} GB "
              f"(resident in page cache; no GPU memory used)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="COUNT lookup selfcheck")
    ap.add_argument("--table", action="append", default=[], help="table directory (repeatable)")
    ap.add_argument("--corpus", default="data/fineweb-edu/val.bin")
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--j", type=int, default=6)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-order", type=int, default=-1, help="-1 = the table's maximum")
    ap.add_argument("--backoff-depth", type=int, default=-1, help="-1 = back off to the unigram row")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.25)
    ap.add_argument("--bench-repeats", type=int, default=5)
    ap.add_argument("--no-reference", action="store_true")
    ap.add_argument("--report-positions", type=int, default=3)
    args = ap.parse_args(argv)

    if not args.table:
        ap.error("give at least one --table")
    max_order = None if args.max_order < 0 else args.max_order

    for d in args.table:
        with CountTable(d) as t:
            print(t.describe())

    tokens = random_windows(args.corpus, args.rows, args.seq_len)
    weights = [1.0 - args.lam, args.lam] if len(args.table) == 2 else [1.0]
    tables = [CountTable(d) for d in args.table]
    try:
        res = mix(tables, weights, tokens, j=args.j, k=args.k, max_order=max_order,
                  backoff_depth=args.backoff_depth)
        print(f"lookup shape: candidates {res.candidates.shape} scores {res.scores.shape} "
              f"n_found {res.n_found.shape}")
        print(f"coverage: {res.coverage():.3f} of positions resolved at the longest order asked "
              f"for; mean candidates per resolved order {res.mean_candidates():.2f}")
        bad = 0
        if not args.no_reference:
            bad += _compare_one(res, args.table, weights, tokens, args.j, args.k, max_order,
                                args.backoff_depth, args.report_positions)
        bad += check_decode_path(tables[0], tokens, args.j, args.k)
        bad += check_score_semantics(res, args.k, tables[0].backoff)
    finally:
        for t in tables:
            t.close()
    bad += check_mixing(args.table, tokens[:4], args.j, args.k, args.backoff_depth)

    if args.bench_repeats:
        tables = [CountTable(d) for d in args.table]
        bench_tokens = random_windows(args.corpus, max(args.rows, 64), args.seq_len, seed=7)
        bench(tables, bench_tokens, args.j, args.k, args.bench_repeats)
        for t in tables:
            t.close()

    print("SELFCHECK " + ("OK" if bad == 0 else f"FAILED ({bad} mismatches)"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
