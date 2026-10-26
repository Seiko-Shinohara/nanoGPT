"""Table-side evaluation of COUNT: what the n-gram tables themselves buy a model.

There is no trained model in this repo yet (no pretraining runs), so this script measures the half
of the paper that does not need weights -- how well the stored continuations predict the tokens a
held-out corpus actually contains. Those are the quantities the paper's LM loss and BPB numbers
track: a table that proposes the corpus's next token more often, with a higher score, is a table
whose f_t carries more usable signal.

Three experiments from the paper are covered:

  Sec. 5.1  inference-time domain adaptation. Score the *same* held-out documents under
            S_lambda = (1 - lambda) S_general + lambda S_domain for lambda in {0, 0.25, 0.5, 1}.
            No retraining, no new table format -- the tables are just read with mixed weights.
  Sec. 5.2  shrinking a table to fit a RAM-constrained deployment: several tables built from the
            same corpus with different history-count / order thresholds, scored on the same data.
  Sec. 3    coverage and candidate density, the two numbers that decide whether lookup is cheap
            (a position with 6 orders x 8 candidates resolves 48 lookups worth of summing, a
            position that misses resolves nothing).

Metrics, all over the J x K candidate union at each position (targets are the corpus's own next
tokens, so this is "would COUNT have proposed what the text actually said"):

  coverage        fraction of positions where the longest stored suffix resolved to a row
  cand/pos        mean candidates summed over contributing orders (out of J * K)
  hit@union       fraction of positions where the corpus's next token is in the candidate union
  top1            argmax over all candidate scores == the corpus's next token
  S(true)         max stored score of the corpus's next token at that position (0 if absent)
  mass            mean sum of a position's candidate scores -- the paper never renormalizes, so
                  this is the raw conditional mass Eq. 1 adds to the residual stream
  rank            1 + how many candidates outscore the corpus's next token (mean over hit rows)

Example (a general table, then the same corpus scored with a math table mixed in):

  venv/bin/python eval_count.py \\
      --table count_tables/general150m \\
      --domain-table count_tables/math_s47 \\
      --corpus data/fineweb-edu/val.bin \\
      --corpus domain@data/fineweb-edu/domain_math_s47_val.bin \\
      --lambdas 0,0.25,0.5,1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from count.data import CountDataPipeline, open_tables, table_weights
from count.table import CountTable, set_threads


METRICS = ["coverage", "cand/pos", "hit@union", "top1", "S(true)", "mass", "rank"]


def score_positions(
    candidates: np.ndarray,
    scores: np.ndarray,
    n_found: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """Aggregate the per-position metrics. candidates/scores: (rows, seq, J, K)."""
    rows, seq, j, k = candidates.shape
    cand = candidates.reshape(rows, seq, j * k)
    sc = scores.reshape(rows, seq, j * k)
    # A candidate that a mixed row carries with score exactly 0 (Sec. 5.1: "zero contribution from a
    # table when a candidate is absent from its row") contributes nothing to Eq. 1, so it is counted
    # as absent here. It is still in the buffer because the C++ reader pads a row out to K slots.
    valid = (cand >= 0) & (sc > 0)
    touched = n_found > 0  # positions where at least one order resolved

    match = valid & (cand == targets[:, :, None])
    hit = match.any(axis=-1)

    # The union is not globally sorted (each order's row is sorted on its own), so the best score
    # for a token is the max over the slots holding it, and the rank counts strictly better scores.
    score_true = np.where(match, sc, -np.inf).max(axis=-1)
    score_true_hit = np.where(hit, score_true, 0.0)
    rank = np.zeros_like(hit, dtype=np.int64)
    if hit.any():
        # how many candidates strictly outscore the corpus's next token. Ties are not counted
        # against it: a candidate that ties the true token is still a candidate for it.
        strict = valid & (sc > score_true_hit[:, :, None])
        rank = strict.sum(axis=-1) + 1

    best_slot = np.where(valid, sc, -np.inf).argmax(axis=-1)
    top1_cand = np.take_along_axis(cand, best_slot[:, :, None], axis=-1)[:, :, 0]
    top1_ok = touched & (top1_cand == targets)

    mass = np.where(valid, sc, 0.0).sum(axis=-1)

    n_touched = max(1, int(touched.sum()))
    n_pos = rows * seq
    return {
        "coverage": float(touched.sum()) / n_pos,
        "cand/pos": float(valid.sum()) / n_pos,
        "hit@union": float(hit.sum()) / n_pos,
        "top1": float(top1_ok.sum()) / n_pos,
        "S(true)": float(score_true_hit[touched].mean()) if touched.any() else 0.0,
        "mass": float(mass[touched].mean()) if touched.any() else 0.0,
        "rank": float(rank[hit].mean()) if hit.any() else float(j * k),
    }


def evaluate(
    tables: Sequence[CountTable],
    weights: Sequence[float],
    path: str,
    rows: int,
    seq_len: int,
    batches: int,
    j: int,
    k: int,
    max_order: Optional[int],
    backoff_depth: int,
    workers: int,
    seed: int,
) -> Dict[str, float]:
    """Sample `batches` windows from `path` and score them under these tables."""
    pipeline = CountDataPipeline(
        tables, weights, [path], seq_len, rows, j=j, k=k, max_order=max_order,
        backoff_depth=backoff_depth, device="cpu", workers=workers, prefetch_depth=1, seed=seed,
    )
    totals: Dict[str, float] = {m: 0.0 for m in METRICS}
    done = 0
    try:
        for _ in range(batches):
            batch = pipeline.next_batch()
            if batch is None:
                break
            targets = batch["targets"].numpy()
            res = score_positions(batch["candidates"].numpy(), batch["scores"].numpy(),
                                  batch["n_found"].numpy(), targets)
            for m in METRICS:
                totals[m] += res[m]
            done += 1
    finally:
        pipeline.close()
    if done == 0:
        raise RuntimeError(f"no batches sampled from {path}")
    out = {m: totals[m] / done for m in METRICS}
    out["tokens"] = float(done * rows * seq_len)
    return out


def table_label(tables: Sequence[CountTable], weights: Sequence[float]) -> str:
    if len(tables) == 1:
        return os.path.basename(tables[0].directory)
    parts = []
    for t, w in zip(tables, weights):
        if w > 0:
            parts.append(f"{w:.2f}*{os.path.basename(t.directory)}")
    return " + ".join(parts)


def corpus_arg(spec: str) -> Tuple[str, str]:
    """'path' or 'name@path' so a run can label the corpus it is scoring."""
    if "@" in spec:
        name, path = spec.split("@", 1)
        return name, path
    return os.path.basename(spec), spec


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", action="append", default=[], help="general table directory")
    ap.add_argument("--domain-table", action="append", default=[],
                    help="domain table directory, mixed in over the --lambdas grid (Sec. 5.1)")
    ap.add_argument("--shrink", action="append", default=[],
                    help="extra table directories to score as shrink candidates (Sec. 5.2)")
    ap.add_argument("--corpus", action="append", default=[], required=True,
                    help="eval .bin file, optionally 'label@path'. Repeat for several corpora.")
    ap.add_argument("--lambdas", default="0,0.25,0.5,1")
    ap.add_argument("--rows", type=int, default=256, help="sequences per batch")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--j", type=int, default=6)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-order", type=int, default=0, help="0 = every order the table stores")
    ap.add_argument("--backoff-depth", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(argv)

    if not args.table:
        ap.error("--table is required (a directory written by count_native/count_build)")
    set_threads(args.workers)
    general = open_tables(args.table)
    domains = open_tables(args.domain_table + args.shrink)
    lambdas = [float(x) for x in args.lambdas.split(",") if x != ""]
    corpora = [corpus_arg(c) for c in args.corpus]
    for _, path in corpora:
        if not os.path.exists(path):
            sys.exit(f"corpus not found: {path}")
    max_order = args.max_order if args.max_order > 0 else None

    configs: List[Tuple[str, List[CountTable], List[float]]] = []
    # Every --table (and every --shrink) is scored on its own: that is Sec. 5.2's comparison, tables
    # of different sizes read by the same procedure. Domain mixing is defined against one general
    # table -- the first --table given -- because S_lambda is a two-table interpolation.
    for t in general:
        configs.append((table_label([t], [1.0]), [t], [1.0]))
    if domains:
        if len(general) > 1:
            print(f"note: mixing the domain tables into {os.path.basename(general[0].directory)} "
                  f"(the first --table). S_lambda interpolates exactly two tables.")
        for lam in lambdas:
            for domain in domains:
                configs.append((table_label([general[0], domain], table_weights(lam, 2)),
                                [general[0], domain], table_weights(lam, 2)))

    print(f"COUNT table evaluation: J={args.j} K={args.k} orders="
          f"{'all' if max_order is None else max_order} backoff_depth={args.backoff_depth} "
          f"rows={args.rows} seq_len={args.seq_len} batches={args.batches}")
    for t in general + domains:
        print(t.describe())
    print()

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, tables, weights in configs:
        results[name] = {}
        for label, path in corpora:
            t0 = time.time()
            res = evaluate(tables, weights, path, args.rows, args.seq_len, args.batches, args.j,
                           args.k, max_order, args.backoff_depth, args.workers, args.seed)
            res["seconds"] = time.time() - t0
            results[name][label] = res
            print(f"  scored {name} on {label}: {res['tokens']:,.0f} tokens in {res['seconds']:.1f}s")

    header = f"{'config':<46}{'corpus':<26}" + "".join(f"{m:>11}" for m in METRICS)
    for label, path in corpora:
        print()
        print(f"=== corpus: {label}  ({path})")
        print(header)
        for name, _, _ in configs:
            res = results[name][label]
            row = f"{name:<46}{label:<26}" + "".join(f"{res[m]:>11.4f}" for m in METRICS)
            print(row)

    if domains and lambdas:
        print()
        print("=== Sec. 5.1 domain-adaptation deltas (vs lambda=0 on the same corpus)")
        base_name = table_label([general[0], domains[0]], table_weights(0.0, 2))
        for label, _ in corpora:
            base = results[base_name][label]
            for lam in lambdas[1:]:
                name = table_label([general[0], domains[0]], table_weights(lam, 2))
                res = results[name][label]
                print(f"  lambda={lam:<5} {label:<26} hit@union {res['hit@union'] - base['hit@union']:+.4f}"
                      f"  top1 {res['top1'] - base['top1']:+.4f}"
                      f"  S(true) {res['S(true)'] - base['S(true)']:+.4f}"
                      f"  rank {res['rank'] - base['rank']:+.4f}")

    for t in general + domains:
        t.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
