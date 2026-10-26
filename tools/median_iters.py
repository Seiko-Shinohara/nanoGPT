"""Median iteration time of a train.py log, over the paper's measurement window.

Run:  venv/bin/python tools/median_iters.py logs/timing_*.log [more logs...]

Sec. 6 measures training throughput as "median tokens per second (TPS) of optimization
steps 50-100". This does the same thing to a train.py log: it reads the per-iteration
times, keeps iterations 52-100 (the window, minus the boundary iterations the paper
excludes), and reports the median time, the median tokens/s implied by the run's tokens
per iteration, and each arm's throughput as a share of the first log listed -- which is
how "retaining 93.1-97.8% of standard transformer training throughput" should be read.

Pass --window to move the window, and --tokens-per-iter if the log's config is not the
one you want to score against (it is read out of the log's own "tokens per iteration
will be:" line when present).
"""

import argparse
import os
import re
import statistics
import sys

ITER = re.compile(r'iter (\d+): loss [\d.]+, time ([\d.]+)ms, mfu ([\d.]+)%'
                  r'(?:, count_wait ([\d.]+)ms/micro-batch)?')
TOKENS = re.compile(r'tokens per iteration will be: ([\d,]+)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--window', default='52-100', help="iteration range to average over")
    ap.add_argument('--tokens-per-iter', type=int, default=0)
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.window.split('-'))

    rows = []
    for path in args.logs:
        if not os.path.exists(path):
            print(f"{path}: missing")
            continue
        tokens = args.tokens_per_iter
        samples, waits = [], []
        for line in open(path, errors='replace'):
            m = TOKENS.search(line)
            if m and not tokens:
                tokens = int(m.group(1).replace(',', ''))
            m = ITER.search(line)
            if m and lo <= int(m.group(1)) <= hi:
                samples.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
                if m.group(4) is not None:
                    waits.append(float(m.group(4)))
        if not samples:
            print(f"{path}: no iterations inside window {lo}-{hi}")
            continue
        times = [t for _, t, _ in samples]
        median_ms = statistics.median(times)
        rows.append({
            'name': os.path.basename(path).removeprefix('timing_').removesuffix('.log'),
            'n': len(samples),
            'median_ms': median_ms,
            'p10_ms': sorted(times)[max(0, len(times) // 10)],
            'p90_ms': sorted(times)[min(len(times) - 1, 9 * len(times) // 10)],
            'tps': tokens / median_ms * 1000.0 if tokens else float('nan'),
            'mfu': statistics.median([m for _, _, m in samples]),
            'count_wait_ms': statistics.median(waits) if waits else None,
            'tokens': tokens,
        })

    if not rows:
        return 1
    base = rows[0]
    print(f"window: iterations {lo}-{hi}   tokens/iteration: {base['tokens']:,}   "
          f"(share is against {base['name']})")
    print(f"{'arm':16s} {'n':>4s} {'median ms':>10s} {'p10':>8s} {'p90':>8s} "
          f"{'median TPS':>12s} {'share':>8s} {'mfu %*':>8s} {'count_wait ms':>13s}")
    for r in rows:
        share = base['median_ms'] / r['median_ms']
        wait = f"{r['count_wait_ms']:.2f}" if r['count_wait_ms'] is not None else "-"
        print(f"{r['name']:16s} {r['n']:4d} {r['median_ms']:10.2f} {r['p10_ms']:8.2f} "
              f"{r['p90_ms']:8.2f} {r['tps']:12,.0f} {share * 100:7.1f}% {r['mfu']:7.2f} "
              f"{wait:>13s}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
