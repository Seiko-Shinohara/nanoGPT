"""
Token-logits -> byte-logits conversion (paper Section 2, Appendix F.1).

Given the teacher's next-token distribution P(v) over the BPE vocabulary V and
the gold (ground-truth) token, we form a distribution over the student's byte
vocabulary at each byte position of the gold token.

Marginalize-It (approximate)
-----------------------------
Student alphabet: {0..255} U specials. Each token v has symbol sequence
s_v = bytes(v) of length l_v. For byte position k of the gold token g:

    active_k = { v : s_v[:k] == s_g[:k] and l_v > k }
    dist_k   = normalize( sum_{v in active_k} P(v) * onehot(s_v[k]) )

The first byte (k=0) is exact; for k>0 the probability mass of tokens that
are strict prefixes of the gold (l_v == k) is dropped and the remainder is
renormalized -- hence "approximate".

End-Of-Token (exact)
---------------------
Student alphabet: {0..255} U specials U {EOT}. Each token's symbol sequence
is s_v = bytes(v) + [EOT] (length l_v + 1); the gold sequence is
s_g = bytes(g) + [EOT]. The same marginalization is applied, but now tokens
that end at position k contribute their mass to the EOT symbol, so no
probability is lost and the conversion is exact:

    prod_k P_dist_k(s_g[k])  ==  P(g)          (likelihood is preserved)

Both methods are single-pass: only the teacher distribution at the boundary
before the gold token is needed (no extra teacher inferences).

The live sets (active_k) depend only on the gold token, not on the teacher
probabilities, so they are precomputed once per tokenizer (see
precompute_live_sets) and the per-position conversion is a few gathers and a
scatter_add -- fully vectorized over a chunk of positions.
"""

import numpy as np
import torch


def precompute_live_sets(sym_table, lengths):
    """Precompute, for every gold token g and depth k < len(g), the set of
    vocabulary tokens that are active at depth k and their k-th symbol.

    sym_table: (V, max_len) int64, -1 padded
    lengths:   (V,) int64

    Returns live_sets: list of length V; live_sets[g] is a list of
    (k, live_idx, sym) tuples where
      live_idx: (K,) int64 token indices v with s_v[:k] == s_g[:k] and l_v > k
      sym:      (K,) int64 k-th symbol of those tokens
    """
    V, max_len = sym_table.shape
    all_idx = np.arange(V)
    live_sets = []
    for g in range(V):
        L = int(lengths[g])
        s_g = sym_table[g]
        live = all_idx  # k = 0: every token has length > 0
        entries = []
        for k in range(L):
            entries.append((k, live.copy(), sym_table[live, k].copy()))
            m = (sym_table[live, k] == s_g[k]) & (lengths[live] > k + 1)
            live = live[m]
        live_sets.append(entries)
    return live_sets


def first_symbol_matrix(sym_table, v_byte):
    """One-hot matrix M (V, v_byte) with M[v, s_v[0]] = 1.
    The k=0 distribution is gold-independent: dist_0 = P @ M."""
    M = np.zeros((sym_table.shape[0], v_byte), dtype=np.float32)
    M[np.arange(sym_table.shape[0]), sym_table[:, 0]] = 1.0
    return M


def convert_chunk(probs, gold, live_sets, first_mat, max_len, v_byte):
    """Convert teacher token probabilities to per-position byte distributions.

    probs:     (N, V) float32 tensor -- teacher probabilities at N boundaries
    gold:      (N,) int64 tensor -- gold token id at each boundary
    live_sets: from precompute_live_sets (CPU numpy)
    first_mat: (V, v_byte) float32 tensor (CPU or same device)
    max_len:   max symbol length (rows of the output)
    v_byte:    student byte vocab size (260 or 261)

    Returns (N, max_len, v_byte) float32 tensor; row n, column k is the
    distribution over the k-th symbol of gold[n] (zeros for k >= len(gold[n])).
    """
    device = probs.device
    N = probs.shape[0]
    out = torch.zeros(N, max_len, v_byte, device=device, dtype=torch.float32)

    # k = 0: exact, gold-independent marginalization over the full vocabulary
    out[:, 0] = probs @ first_mat.to(device)

    # k >= 1: group positions by gold token (live sets depend only on gold)
    for g in torch.unique(gold).tolist():
        idx = torch.nonzero(gold == g).squeeze(-1)
        if idx.numel() == 0:
            continue
        P = probs[idx]  # (n, V)
        for (k, live_idx, sym) in live_sets[g]:
            if k == 0:
                continue
            live_idx = torch.as_tensor(live_idx, dtype=torch.long, device=device)
            sym = torch.as_tensor(sym, dtype=torch.long, device=device)
            Pk = P[:, live_idx]  # (n, K)
            dist = torch.zeros(idx.numel(), v_byte, device=device, dtype=torch.float32)
            dist.scatter_add_(1, sym.expand(idx.numel(), -1), Pk)
            dist.div_(dist.sum(dim=-1, keepdim=True).clamp_min(1e-30))
            out[idx, k] = dist
    return out


# ---------------------------------------------------------------------------
# naive reference implementation (for tests): direct transcription of the
# paper's Appendix F.1 pseudocode, one position at a time
# ---------------------------------------------------------------------------
def byte_probabilities_reference(probs, token_to_bytes, lengths, gold, use_eot=False):
    """Paper pseudocode, verbatim semantics.
    probs: (V,) float array; token_to_bytes: (V, max_len) int array, -1 padded
    (EOT value at the terminal slot when use_eot); lengths: (V,) int array;
    gold: int array, the gold token's symbol sequence.
    Returns list of distributions (arrays of size 257 if use_eot else 256).
    """
    n_bytes = 257 if use_eot else 256
    live = np.ones(len(probs), dtype=bool)
    dist = []
    for k in range(len(gold)):
        active = live & (lengths > k)
        bytes_at_k = token_to_bytes[:, k]
        accum = np.bincount(bytes_at_k[active], weights=probs[active], minlength=n_bytes)
        total = accum.sum()
        dist.append(accum / total if total > 0 else accum)
        live &= (bytes_at_k == gold[k]) & (lengths > k + 1)
    return dist