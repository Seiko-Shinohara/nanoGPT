"""Learned n-gram embeddings: the baseline COUNT is compared against (paper Sec. 3.3, Eq. 3).

Over-Encoding (Huang et al., 2025) as implemented in LongCat (Liu et al., 2026). Where COUNT looks
n-grams up in a read-only table of *scores* built offline, this baseline learns a *vector* per hashed
n-gram jointly with the backbone:

    key_{n,r}(t) = ( sum_{j=0}^{n-1} x_{t-j} |V|^j )  mod M_{n,r}
    q_t          = sum_{n=2..NL} sum_{r=1..R} P_{n,r} T_{n,r}[ key_{n,r}(t) ]
    h_t^(0)      = e_t + q_t

T_{n,r} is an embedding table of M_{n,r} rows and width d_s = d/8; P_{n,r} in R^{d x d_s} projects the
retrieved vector to model width. Orders 2..3 with R = 4 tables per order is the paper's configuration.

The difference that motivates COUNT is entirely in where the parameters live: these tables are learned
parameters, so their optimizer state (two moments per row) sits on the GPU next to the backbone, and
the table has to be re-learned for every training run. COUNT's table is host-resident, carries no
optimizer state, and is reused across runs.

`ngram_keys` implements Eq. 3's key with the modulo applied at every Horner step, so the arithmetic
never exceeds int64 for any order. The paper's product |V|^j reaches |V|^3 at order 3, which is
1.3e14 for |V| = 50304: inside int64, but only just, and order 4 would not be.
"""

from __future__ import annotations

from typing import List, Sequence

import torch


def ngram_keys(
    idx: torch.Tensor,
    orders: Sequence[int],
    moduli: Sequence[Sequence[int]],
    vocab: int,
) -> torch.Tensor:
    """Eq. 3's hash keys for every position, order and table.

    idx: (b, t) token ids. Returns (b, t, n_orders, R) int64 keys in [0, M_{n,r}).

    Horner from the oldest token to the newest:
        k = ((x_{t-(n-1)} |V| + x_{t-(n-2)}) |V| + ... ) |V| + x_t
    which expands to sum_{j=0}^{n-1} x_{t-j} |V|^j: the newest token has weight |V|^0, as in the paper.
    Reducing mod M at each step keeps the values bounded by M |V| instead of |V|^n.

    Positions nearer the start of the window than order n have less history than n tokens. They are
    zero-padded on the left (history tokens = token 0), so every position gets a key and partial
    n-grams hash into the same space as complete ones.
    """
    b, t = idx.shape
    if len(moduli) != len(orders):
        raise ValueError("moduli must have one list of R moduli per order")
    out = torch.empty((b, t, len(orders), max(len(m) for m in moduli)), dtype=torch.int64,
                      device=idx.device)
    idx64 = idx.to(torch.int64)
    max_order = max(orders)
    padded = torch.nn.functional.pad(idx64, (max_order - 1, 0), value=0)
    for oi, n in enumerate(orders):
        if n < 2:
            raise ValueError(f"n-gram orders start at 2, got {n}")
        # Position p of the window ends at padded index p + max_order - 1 (the padding shifted every
        # real index right by max_order - 1) and its order-n window starts n-1 tokens before that.
        start = max_order - n
        for r, M in enumerate(moduli[oi]):
            key = torch.zeros((b, t), dtype=torch.int64, device=idx.device)
            for i in range(n): # oldest first, so the newest token lands with weight |V|^0
                key = (key * vocab + padded[:, start + i:start + i + t]) % M
            out[:, :, oi, r] = key
    return out


def prime_at_least(n: int) -> int:
    """Smallest prime >= n. Distinct prime moduli are the cheapest way to give the R tables of one
    order independent hash images (a composite M shares factors with |V| and collides systematically)."""
    n = max(2, int(n))
    if n % 2 == 0:
        n += 1
    while True:
        d = 3
        while d * d <= n:
            if n % d == 0:
                break
            d += 2
        else:
            return n
        n += 2


def make_moduli(rows_per_table: int, n_tables: int, n_orders: int) -> List[List[int]]:
    """R distinct prime moduli per order, each the smallest prime at or above rows_per_table.

    The primes keep advancing across orders, so all R * (NL-1) moduli are distinct: no two tables of
    any order share their hash image. Distinctness across orders costs nothing here (a prime gap near
    10^5 is a few hundred) and it keeps every table the requested size, which the 1:1 parameter rule
    of Sec. 3.3 depends on.
    """
    moduli: List[List[int]] = []
    want = max(2, int(rows_per_table))
    for _ in range(n_orders):
        chosen = []
        for _ in range(n_tables):
            want = prime_at_least(want + 1)
            chosen.append(want)
            want += 1
        moduli.append(chosen)
    return moduli


def table_param_count(moduli: Sequence[Sequence[int]], width: int) -> int:
    return sum(M for row in moduli for M in row) * width


def rows_for_ratio(
    other_params: int,
    input_embedding_params: int,
    n_embd: int,
    width_div: int,
    n_tables_total: int,
    ratio: float = 1.0,
) -> int:
    """Rows per table under the paper's 1:1 rule (Sec. 3.3):

        input-token embeddings + n-gram table parameters == all remaining learned parameters

    so table parameters = ratio * (all other parameters - input embeddings), spread over
    n_tables_total = R * (NL - 1) tables of width d/8. Returns the rows_per_table that makes that
    product hit the target, or raises when the rule cannot be met.

    It cannot be met when the tied input embeddings are already larger than the rest of the model,
    which is the case for a 50k-vocab model at n_embd=256: the embeddings are 12.9M parameters and
    four blocks are 3.1M. The rule is meaningful at the paper's sizes (and at nanoGPT's 124M), not at
    smoke size -- there the row count has to be set by hand, and the raise says so.
    """
    width = n_embd // width_div
    backbone = other_params - input_embedding_params
    if backbone <= 0:
        raise ValueError(
            f"the 1:1 ratio cannot be met with this architecture: input embeddings hold "
            f"{input_embedding_params:,} parameters and the rest of the model holds {backbone:,}. "
            "Set ngram_rows to an explicit row count instead.")
    return max(1, int(round(ratio * backbone)) // (max(1, n_tables_total) * width))
