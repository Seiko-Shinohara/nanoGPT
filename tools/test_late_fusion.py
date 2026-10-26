"""Tests for count/late_fusion.py: do Equations 5-7 come out the way Appendix B.1 writes them?

Run:  venv/bin/python tools/test_late_fusion.py [--table count_tables/fw5m_lo]

Five things worth not taking on faith, because each is a place where a fusion implementation goes
wrong quietly and the only symptom is a loss curve:

1. weight=0 is an exact no-op in both variants (otherwise the "baseline" arm is contaminated).
2. A row that stores nothing leaves p untouched. Additive: m_t = min{A_t, 1} = 0. Multiplicative:
   Eq. 6 degenerates to a uniform 1/|V|, whose normalized product with p is p. If a padded slot is
   scattered as -inf instead, this test fails -- and the token it kills is id 0, at every short row.
3. Both fused distributions are normalized. Eq. 5 is normalized by construction (sum q = m_t);
   Eq. 7 needs its denominator, and forgetting it shifts every loss by a constant.
4. A pure-Python transcription of Eq. 5-7, written straight from the paper over dictionaries,
   agrees with the vectorized torch version to 1e-7 on a small vocab.
5. With a real table and J=1 (the paper's "only the longest matching history"), fusion where the
   table proposes the corpus's next token lowers that token's NLL, and the gain tracks S(w_true | h).
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from count.late_fusion import (fuse_additive, fuse_logits, fuse_multiplicative,
                               multiplicative_prior, retained_mass, scores_for_targets)


EPS = 1e-4


def reference_additive(logp, cands, sc, lam):
    """Eq. 5, one row at a time, in Python."""
    vocab = logp.shape[1]
    out = np.zeros_like(logp)
    for r in range(logp.shape[0]):
        w = [(c, s) for c, s in zip(cands[r], sc[r]) if c >= 0 and s > 0]
        a = sum(s for _, s in w)
        m = min(a, 1.0)
        q = {c: s / max(1.0, a) for c, s in w}
        p = np.exp(logp[r])
        p_add = (1.0 - lam * m) * p + lam * np.array([q.get(v, 0.0) for v in range(vocab)])
        out[r] = np.log(p_add)
    return out


def reference_multiplicative(logp, cands, sc, alpha, eps=EPS):
    """Eq. 6 and Eq. 7, one row at a time, in Python."""
    vocab = logp.shape[1]
    out = np.zeros_like(logp)
    for r in range(logp.shape[0]):
        w = [(c, s) for c, s in zip(cands[r], sc[r]) if c >= 0 and s > 0]
        a = sum(s for _, s in w)
        m_e = min(a, 1.0 - eps)
        denom = max(1.0, a / (1.0 - eps))
        q = {c: s / denom for c, s in w}
        other = (1.0 - m_e) / (vocab - len(w))
        p = np.exp(logp[r])
        fused = np.array([p[v] * (q.get(v, other) ** alpha) for v in range(vocab)])
        out[r] = np.log(fused / fused.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', default='count_tables/fw5m_lo')
    ap.add_argument('--tokens', default='data/fineweb-edu/val.bin')
    ap.add_argument('--rows', type=int, default=8)
    ap.add_argument('--seq-len', type=int, default=32)
    args = ap.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    vocab, rows, k = 97, 6, 8

    # A synthetic batch: some rows full of candidates, some empty, some with mass > 1.
    cands = np.full((rows, k), -1, dtype=np.int32)
    sc = np.zeros((rows, k), dtype=np.float32)
    for r in range(rows):
        n = {0: k, 1: 3, 2: 0, 3: k, 4: 1, 5: k}[r]
        if n == 0:
            continue
        picks = rng.choice(vocab, size=n, replace=False)
        cands[r, :n] = picks
        # order 0 rows deliberately exceed 1.0 of mass: the max{1, A_t} cap is the point of Eq. 5
        scale = {0: 0.6, 1: 0.3, 3: 1.9, 4: 0.9, 5: 0.05}[r]
        sc[r, :n] = (rng.dirichlet(np.ones(n)) * scale).astype(np.float32)

    logp_np = rng.dirichlet(np.ones(vocab), size=rows)
    logp_np = np.log(logp_np).astype(np.float64)
    logp = torch.from_numpy(logp_np).float()
    ct = torch.from_numpy(cands).long()
    st = torch.from_numpy(sc).float()

    # 1. weight 0 is a no-op -------------------------------------------------
    for name, fn in (("additive", fuse_additive), ("multiplicative", fuse_multiplicative)):
        same = fn(logp, ct, st, 0.0)
        assert torch.equal(same, logp), f"{name} with weight 0 changed the distribution"
    print("1. weight=0 no-op: both variants return the input log-probs bitwise")

    # 2. an empty row is left alone, and padding poisons nothing --------------
    assert ct[2].min().item() == -1, "test batch lost its empty row"
    w = 0.5
    fused_a, fused_m = fuse_additive(logp, ct, st, w), fuse_multiplicative(logp, ct, st, w)
    stored_4 = [(c, s) for c, s in zip(cands[4], sc[4]) if c >= 0 and s > 0]
    assert len(stored_4) == 1, "row 4 should hold exactly one candidate"
    non_candidates = [v for v in range(vocab) if v != stored_4[0][0]]
    for name, fused in (("additive", fused_a), ("multiplicative", fused_m)):
        d = (fused[2].exp() - logp[2].exp()).abs().max().item()
        assert d < 1e-6, f"{name}: an unresolved row changed (max diff {d:.2e})"
        ratio = fused[4, non_candidates].exp() / logp[4, non_candidates].exp()
        spread = (ratio.max() / ratio.min()).item()
        assert ratio.min().item() > 0, f"{name}: a non-candidate got zero probability"
        assert abs(spread - 1.0) < 1e-4, (
            f"{name}: non-candidates are not scaled by one common factor (spread {spread:.6f})")
    scale_a = (fused_a[4, non_candidates[0]].exp() / logp[4, non_candidates[0]].exp()).item()
    scale_m = (fused_m[4, non_candidates[0]].exp() / logp[4, non_candidates[0]].exp()).item()
    print(f"2. A_t=0 row untouched; at the one-candidate row the other {len(non_candidates)} tokens "
          f"scale by one factor: {scale_a:.4f} additive (1 - lambda m), {scale_m:.4f} multiplicative")

    # 3. normalized ----------------------------------------------------------
    for name, fused in (("additive", fused_a), ("multiplicative", fused_m)):
        sums = fused.exp().sum(dim=-1)
        print(f"3. {name}: row masses {sums.min().item():.8f}..{sums.max().item():.8f}")
        assert (sums - 1).abs().max().item() < 1e-5, f"{name} is not normalized"

    # 4. the vectorized torch path vs a Python transcription of Eq. 5-7 ------
    lam, alpha = 0.1, 0.1
    ref_a = reference_additive(logp_np, cands, sc, lam)
    got_a = fuse_additive(logp, ct, st, lam).numpy().astype(np.float64)
    ref_m = reference_multiplicative(logp_np, cands, sc, alpha)
    got_m = fuse_multiplicative(logp, ct, st, alpha).numpy().astype(np.float64)
    da, dm = np.abs(ref_a - got_a).max(), np.abs(ref_m - got_m).max()
    print(f"4. Eq. 5 vs torch: max |log p| diff {da:.2e};  Eq. 6-7 vs torch: {dm:.2e}")
    # The multiplicative gap is float32, not the equation: the row whose A_t exceeds 1-eps puts
    # candidates near 1.0 and non-candidates near 1e-6, so its log-probs span 14 orders and fp32
    # carries about 1e-5 of absolute error across that span (the same comparison in float64 lands at
    # 7e-8). Eq. 5's additive blend never opens that gap, which is why it matches two orders tighter.
    assert da < 1e-5 and dm < 5e-5, "the vectorized fusion does not match the paper's equations"
    exact = fuse_multiplicative(logp.double(), ct, st.double(), alpha).numpy()
    dm64 = np.abs(ref_m - exact).max()
    print(f"4a. the same Eq. 6-7 check in float64: {dm64:.2e} -- the gap above is fp32 rounding")
    assert dm64 < 1e-6

    # 4c. scores_for_targets indexes candidate *slots*, not vocabulary ids ----------------------
    absent = next(v for v in range(vocab) if v not in set(cands[1][cands[1] >= 0].tolist()))
    probe = torch.full((rows,), absent, dtype=torch.long)
    probe[4] = int(stored_4[0][0])
    s_probe = scores_for_targets(ct, st, probe)
    assert abs(s_probe[4].item() - stored_4[0][1]) < 1e-6, (
        f"scores_for_targets missed a stored token ({s_probe[4].item()} vs {stored_4[0][1]})")
    assert s_probe[1].item() == 0 and s_probe[2].item() == 0, (
        "scores_for_targets reported a mass for a token the row does not store")
    print(f"4c. scores_for_targets: stored -> {s_probe[4].item():.4f}, "
          f"absent -> {s_probe[2].item():.4f} / {s_probe[1].item():.4f}")

    # mass cap: row 0 sums to 0.6, row 3 to 1.9 -> both must cap q at mass 1
    logq, m_e = multiplicative_prior(ct, st, vocab)
    a = retained_mass(ct, st)
    print(f"4b. retained mass A_t {np.round(a.numpy(), 3).tolist()} "
          f"-> candidate mass kept {np.round(m_e.numpy(), 3).tolist()}")
    assert abs(m_e[3].item() - (1 - EPS)) < 1e-6, "A_t > 1-eps is not capped"

    # 5. a real table, J=1: does fusion help where the table is right? -------
    from count.data import open_tables
    from count.generate import count_batch

    table = open_tables([args.table])[0]
    data = np.memmap(args.tokens, dtype=np.uint16, mode='r')
    window = np.asarray(data[:args.rows * args.seq_len], dtype=np.int64).reshape(
        args.rows, args.seq_len)
    idx = torch.from_numpy(window)
    targets = torch.roll(idx, -1, dims=1)
    c = count_batch([table], [1.0], idx, j=1, k=8)  # Eq. 5-7 use h_{t,1}: only the longest order
    ct2, st2 = c['candidates'].reshape(-1, 8), c['scores'].reshape(-1, 8).float()
    tg = targets.reshape(-1)
    lp = torch.log_softmax(torch.randn(ct2.size(0), min(table.vocab_size, 50304)) * 2.0, dim=-1)

    s_true = scores_for_targets(ct2, st2, tg)
    hit = s_true > 0
    nll_p = -lp.gather(1, tg[:, None]).squeeze(1)
    nll_add = -fuse_additive(lp, ct2, st2, 0.1).gather(1, tg[:, None]).squeeze(1)
    nll_mul = -fuse_multiplicative(lp, ct2, st2, 0.1).gather(1, tg[:, None]).squeeze(1)
    print(f"5. table {os.path.basename(table.directory)} J=1 K=8 over {ct2.size(0)} positions: "
          f"{hit.sum().item()}/{hit.numel()} proposals contain the corpus's next token, "
          f"mean S(true) {s_true[hit].mean().item():.4f}")
    for name, nll in (("additive", nll_add), ("multiplicative", nll_mul)):
        d_hit = (nll - nll_p)[hit].mean().item()
        d_miss = (nll - nll_p)[~hit].mean().item()
        print(f"   {name}: NLL change on proposals {d_hit:+.4f} nats, "
              f"on non-proposals {d_miss:+.4f} nats (fusion weight 0.1)")
        assert d_hit < 0, f"{name} fusion hurt a token the table proposed"
        assert fuse_logits(lp, ct2, st2, "none", 0.0).equal(lp)
        try:
            fuse_logits(lp, ct2, st2, "sideways", 0.1)
            raise AssertionError("fuse_logits accepted an unknown mode")
        except ValueError:
            pass
    table.close()

    print("\nLATE FUSION TESTS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
