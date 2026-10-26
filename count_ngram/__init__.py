"""The learned n-gram embedding baseline from Section 3.3 of The COUNT.

COUNT reads n-gram continuation *scores* out of a host-resident table that was built offline and has
no parameters. This package is the alternative the paper measures that against (Over-Encoding, as
deployed in LongCat): a hashed n-gram indexes a *learned* table row of width d/8, a matrix lifts it to
d, and it is added to the token embedding. It works -- it is what LongCat ships -- and the cost is the
point of the comparison: those tables are parameters, they need optimizer state next to the model, and
every run has to learn them again.

    hash.py   Eq. 3's hash keys, the prime moduli behind them, and the 1:1 parameter-ratio rule
    embed.py  the T_{n,r} tables and P_{n,r} projections, on device or offloaded to host RAM
    optim.py  SparseAdam for retrieved rows only, and keeping sparse gradients out of grad clipping

model.py carries the GPTConfig fields (ngram_orders and friends, off by default) and adds q_t to
forward_features; train.py carries the knobs, the second optimizer, and the resume guards.
"""

from count_ngram.embed import LearnedNgramEmbeddings, ngram_from_config
from count_ngram.hash import make_moduli, ngram_keys, prime_at_least, rows_for_ratio, table_param_count
from count_ngram.optim import build_sparse_optimizer, dense_grad_params, set_lr, step_sparse, zero_grad_all

__all__ = [
    "LearnedNgramEmbeddings", "ngram_from_config",
    "ngram_keys", "make_moduli", "prime_at_least", "rows_for_ratio", "table_param_count",
    "build_sparse_optimizer", "dense_grad_params", "set_lr", "step_sparse", "zero_grad_all",
]
