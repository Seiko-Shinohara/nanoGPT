"""
Scaling-law analysis for the byte-distillation study (paper Sections 4, 6, 7).

Reads <run_dir>/scaling.jsonl from each run, fits the validation-BPB vs
training-FLOPs power law

    BPB(F) = a * F^b        (log-log linear regression)

and produces:
  * a comparison table (final BPB, fitted a/b, predicted BPB at a reference
    FLOP budget)
  * bpb_vs_flops.png  -- BPB vs training FLOPs with fitted curves
  * feather.png       -- iso-FLOP connections between the End-Of-Token
    distilled and Token distilled curves (paper Section 7.1 "Feather Plots")

Usage:
  python byte_distill/analyze.py --study_dir out/byte_distill_study
"""

import argparse
import json
import os

import numpy as np


def load_scaling(run_dir):
    path = os.path.join(run_dir, "scaling.jsonl")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fit_power_law(flops, bpb):
    """Fit BPB = a * F^b in log space. Returns (a, b, r2)."""
    F = np.array(flops, dtype=np.float64)
    L = np.array(bpb, dtype=np.float64)
    m = F > 0
    lf, ll = np.log(F[m]), np.log(L[m])
    if len(lf) < 3:
        return None
    A = np.stack([lf, np.ones_like(lf)], axis=1)
    (b, log_a), res, *_ = np.linalg.lstsq(A, ll, rcond=None)
    pred = A @ np.array([b, log_a])
    ss_res = float(((ll - pred) ** 2).sum())
    ss_tot = float(((ll - ll.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(np.exp(log_a)), float(b), r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study_dir", required=True)
    ap.add_argument("--ref_flops", type=float, default=1e17,
                    help="reference FLOP budget for asymptotic comparison")
    args = ap.parse_args()

    runs = {}
    for name in sorted(os.listdir(args.study_dir)):
        rows = load_scaling(os.path.join(args.study_dir, name))
        if rows:
            runs[name] = rows

    if not runs:
        print("no scaling.jsonl found under", args.study_dir)
        return

    fits = {}
    print(f"{'run':<28} {'final BPB':>10} {'a':>12} {'b':>8} {'R2':>7} {'BPB@ref':>10}")
    print("-" * 80)
    for name, rows in runs.items():
        flops = [r["flops"] for r in rows]
        bpb = [r["bpb"] for r in rows]
        fit = fit_power_law(flops, bpb)
        fits[name] = fit
        final = bpb[-1]
        if fit:
            a, b, r2 = fit
            print(f"{name:<28} {final:>10.4f} {a:>12.4e} {b:>8.4f} {r2:>7.3f} "
                  f"{a * args.ref_flops ** b:>10.4f}")
        else:
            print(f"{name:<28} {final:>10.4f}   (too few points to fit)")

    # ------------------------------------------------------------------
    # plots
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available -- skipping plots")
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, rows in runs.items():
        flops = [r["flops"] for r in rows]
        bpb = [r["bpb"] for r in rows]
        ax.scatter(flops, bpb, s=14, label=name, zorder=3)
        fit = fits[name]
        if fit:
            a, b, _ = fit
            fgrid = np.logspace(np.log10(min(f for f in flops if f > 0)),
                                np.log10(max(flops) * 10), 100)
            ax.plot(fgrid, a * fgrid ** b, lw=1.2, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("Training FLOPs (paper Table 9 formula)")
    ax.set_ylabel("Validation BPB")
    ax.set_title("BPB vs Training FLOPs")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.study_dir, "bpb_vs_flops.png"), dpi=140)
    print(f"\nwrote {os.path.join(args.study_dir, 'bpb_vs_flops.png')}")

    # feather plot: iso-FLOP connections between EOT-distilled and token-distilled
    eot_name = next((n for n in runs if "eot" in n and "distill" in n), None)
    tok_name = next((n for n in runs if n.startswith("token") and "distill" in n), None)
    if eot_name and tok_name and fits.get(eot_name) and fits.get(tok_name):
        fig2, ax2 = plt.subplots(figsize=(8, 5.5))
        for name, color in ((tok_name, "tab:blue"), (eot_name, "tab:red")):
            rows = runs[name]
            ax2.scatter([r["flops"] for r in rows], [r["bpb"] for r in rows],
                        s=14, color=color, zorder=3, label=name)
            a, b, _ = fits[name]
            fgrid = np.logspace(np.log10(min(r["flops"] for r in rows if r["flops"] > 0)),
                                np.log10(max(r["flops"] for r in rows) * 10), 100)
            ax2.plot(fgrid, a * fgrid ** b, lw=1.2, color=color, zorder=2)
        # iso-FLOP connection lines
        f_lo = max(min(r["flops"] for r in runs[eot_name] if r["flops"] > 0),
                   min(r["flops"] for r in runs[tok_name] if r["flops"] > 0))
        f_hi = min(max(r["flops"] for r in runs[eot_name]),
                   max(r["flops"] for r in runs[tok_name]))
        a_t, b_t, _ = fits[tok_name]
        a_e, b_e, _ = fits[eot_name]
        for f in np.logspace(np.log10(f_lo), np.log10(f_hi), 12):
            ax2.plot([f, f], [a_t * f ** b_t, a_e * f ** b_e],
                     color="gray", lw=0.8, alpha=0.6, zorder=1)
        ax2.set_xscale("log")
        ax2.set_xlabel("Training FLOPs")
        ax2.set_ylabel("Validation BPB")
        ax2.set_title("Feather plot: iso-FLOP connections (Token vs End-Of-Token distilled)")
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend(fontsize=8)
        fig2.tight_layout()
        fig2.savefig(os.path.join(args.study_dir, "feather.png"), dpi=140)
        print(f"wrote {os.path.join(args.study_dir, 'feather.png')}")


if __name__ == "__main__":
    main()