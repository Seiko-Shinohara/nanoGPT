"""COUNT lookups for generation: the count=... batch that model.forward wants, per decode step.

nanoGPT's generate() re-runs the forward pass over the whole (cropped) prefix every step and keeps
only the last position's logits. That decides the shape here: f_t has to exist for *every* position
in the window, not just the newest one, because the forward pass recomputes all of them.

Doing the full window every step sounds wasteful and is not. The C++ reader resolves a position in
~0.1-0.3 ms per 16k positions with J=6, K=8, so a 1 x 1024 window costs well under a millisecond --
less than the model's own attention over that window. It also keeps training and sampling exactly
consistent: the same table, the same J and K, the same Eq. 1, applied to the same window the
transformer is looking at.

The tail-only alternative (look up just the newest position) would be cheaper but wrong here: the
earlier positions' f_t would silently disappear from the recomputed forward pass, and the logits
would come from a different network than the one that was trained.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from count.table import CountTable, mix


def count_batch(
    tables: Sequence[CountTable],
    weights: Optional[Sequence[float]],
    idx,
    j: int = 6,
    k: int = 8,
    max_order: Optional[int] = None,
    backoff_depth: int = -1,
    score_dtype=None,
):
    """Build the count=... dict for a (b, t) window of token ids (torch tensor or numpy array).

    Handles position 0 the way the lookup does: at the first position of a window the only history
    available is that one token, so only the order-1 row can resolve. A window cropped from a longer
    sequence inherits the context that was generated before it, which is what the table wants.
    """
    import torch

    if torch.is_tensor(idx):
        device = idx.device
        ids = idx.detach().cpu().numpy()
    else:
        device = None
        ids = np.asarray(idx)
    if ids.dtype != np.uint16:
        # The table keys pack a token id into 16 bits, so this implementation's vocab ceiling is
        # 65535. Vocab sizes above that need a wider key in count_native/count_common.h.
        ids = ids.astype(np.uint16)

    res = mix(tables, weights, ids, j=j, k=k, max_order=max_order, backoff_depth=backoff_depth)

    candidates = torch.from_numpy(res.candidates.astype(np.int64)).to(device if device is not None else 'cpu')
    scores = torch.from_numpy(res.scores).to(device if device is not None else 'cpu')
    if score_dtype is not None:
        scores = scores.to(score_dtype)
    n_found = torch.from_numpy(res.n_found.astype(np.int64)).to(device if device is not None else 'cpu')
    return {"candidates": candidates, "scores": scores, "n_found": n_found}
