"""Optimizer wiring for the learned n-gram baseline (paper Sec. 3.3, Appendix B).

The paper's rule for these tables:

    "the table embeddings use sparse optimizer updates for only the retrieved embedding rows",
    using the backbone learning-rate schedule and betas, with zero weight decay

with the throughput numbers that make it necessary: full-table updates ran at 4.4% of standard
training throughput, retrieved-rows-only at 96.3%. A dense AdamW over an 0.4M-4.2M row table would be
paying that 4.4% price on every iteration, and storing two fp32 moments per row on the GPU next to the
model -- which is exactly the memory cost COUNT's design is arguing against.

torch.optim.AdamW cannot take these gradients. nn.Embedding(sparse=True) keeps the *parameter* dense
but produces a sparse COO *gradient*, and AdamW's foreach kernels reject a sparse tensor. torch's
answer is torch.optim.SparseAdam: Adam applied to the rows that appear in the sparse gradient, and to
nothing else. It has no weight decay (which is what the paper wants here anyway) and no fused variant,
so it lives in its own optimizer object next to the backbone's AdamW. That is the honest shape of it:
two optimizers, one learning-rate schedule driving both, two state dicts in the checkpoint.

train.py therefore has three small obligations, and this module keeps them to three functions:
build the sparse optimizer, keep sparse gradients out of gradient clipping, and step/zero both.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch.optim import SparseAdam


def build_sparse_optimizer(model, lr: float, betas=(0.9, 0.95), eps: float = 1e-8):
    """Adam over retrieved rows only, for this model's n-gram tables. None when there is nothing to build.

    Excluded from the backbone's AdamW as well (model.configure_optimizers skips these tensors): a
    parameter in both optimizers would be stepped twice.
    """
    ngram = getattr(model, "ngram", None)
    if ngram is None or not ngram.sparse:
        return None
    params = [w for w in ngram.table_weights() if w.requires_grad]
    if not params:
        return None
    rows = sum(w.shape[0] for w in params)
    print(f"n-gram tables: {len(params)} tables, {rows:,} rows, sparse row-wise Adam "
          f"(SparseAdam, no weight decay, same lr schedule and betas as the backbone)")
    return SparseAdam(params, lr=lr, betas=betas, eps=eps)


def dense_grad_params(model) -> List[torch.nn.Parameter]:
    """The parameters gradient clipping may touch: those whose gradients are dense.

    clip_grad_norm_ reduces gradients with foreach kernels and a sparse COO gradient in the list is a
    type error. Leaving the table rows out of the clip norm is not a shortcut around clipping: their
    update is a per-row Adam step already bounded by the learning rate, and folding rows that happened
    to be retrieved in this batch into a global norm would clip the whole model on which n-grams the
    batch contained.
    """
    return [p for p in model.parameters() if p.grad is not None and not p.grad.is_sparse]


def zero_grad_all(*optimizers) -> None:
    for opt in optimizers:
        if opt is not None:
            opt.zero_grad(set_to_none=True)


def step_sparse(optimizer: Optional[torch.optim.Optimizer]) -> None:
    """Step the sparse optimizer. Not run through GradScaler.step: the tables are trained in the same
    autocast dtype as the backbone, and fp16 loss scaling on a sparse gradient is not something
    SparseAdam supports -- an fp16 run with learned n-grams needs the tables in fp32 instead."""
    if optimizer is not None:
        optimizer.step()


def set_lr(optimizers, lr: float) -> None:
    """One schedule, both optimizers (the paper: the tables use "the backbone learning-rate schedule")."""
    for opt in optimizers:
        if opt is None:
            continue
        for group in opt.param_groups:
            group['lr'] = lr
