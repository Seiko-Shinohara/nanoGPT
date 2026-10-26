"""Turning .bin corpora into COUNT-ready GPU batches (used by train.py and eval_count.py).

nanoGPT's `get_batch` returns (x, y) and nothing else. A COUNT step needs a third thing: the
candidate/score batch Eq. 1 consumes, looked up on the CPU from host-resident tables. This module
owns that plumbing so train.py keeps its shape:

    sample a (batch, block_size + 1) window  ->  lookup every position  ->  pinned buffers
    ->  non_blocking copy to the device     ->  dict(idx, targets, candidates, scores, n_found)

The window is block_size + 1 tokens wide rather than block_size because targets are the window
shifted by one, exactly as nanoGPT does it. The lookup runs on the unshifted window: position p
predicts token p+1 from context x[..p], so the left context the table sees is the same context the
transformer's causal attention sees. The last token of the window is a target, never a lookup
input, which is why lookups are computed over `idx` and not over the wider window.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from count.prefetch import CountPrefetcher, PinnedBuffers, to_device
from count.table import CountTable, mix


class CountDataPipeline:
    """Random windows from one or more .bin files, with COUNT lookups attached.

    Passing several files is what the domain experiments need: one pipeline can interleave the
    general corpus and a domain corpus, and eval can point at a held-out domain file.
    """

    def __init__(
        self,
        tables: Sequence[CountTable],
        weights: Optional[Sequence[float]],
        paths: Sequence[str],
        block_size: int,
        batch_size: int,
        j: int = 6,
        k: int = 8,
        max_order: Optional[int] = None,
        backoff_depth: int = -1,
        device: str = "cuda",
        score_dtype=None,
        workers: int = 8,
        prefetch_depth: int = 0,
        seed: Optional[int] = 0,
        limit_tokens: Optional[int] = None,
    ):
        self.tables = list(tables)
        self.weights = list(weights) if weights is not None else [1.0] * len(self.tables)
        if len(self.weights) != len(self.tables):
            raise ValueError("weights must match tables")
        self.paths = [os.path.abspath(p) for p in paths]
        for p in self.paths:
            if not os.path.exists(p):
                raise FileNotFoundError(p)
            if os.path.getsize(p) // 2 <= block_size + 1:
                raise ValueError(f"{p} is shorter than block_size + 1 tokens")
        self.block_size = int(block_size)
        self.batch_size = int(batch_size)
        self.j = int(j)
        self.k = int(k)
        self.max_order = max_order
        self.backoff_depth = int(backoff_depth)
        self.device = device
        self.score_dtype = score_dtype
        self.workers = int(workers)
        self.prefetch_depth = int(prefetch_depth)
        # Windows are drawn from the first `limit_tokens` tokens of each corpus. A table built with
        # --tokens 150000000 describes that span and nothing after it, so training on windows drawn
        # from the far end of a 1.9B-token corpus asks f_t to describe contexts the table never saw:
        # coverage collapses and the arm looks worse than the method is. The paper builds its tables
        # from the training corpus, so the honest default for a run whose table covers N tokens is to
        # train inside those N tokens.
        self.limit_tokens = int(limit_tokens) if limit_tokens else None
        self._rng = np.random.default_rng(seed)
        self._buffers = PinnedBuffers()
        self._memmaps: List[np.memmap] = []
        self._prefetcher: Optional[CountPrefetcher] = None
        if self.prefetch_depth > 0:
            self._prefetcher = CountPrefetcher(
                self.tables, self.weights, self.sample_tokens, j=self.j, k=self.k,
                max_order=self.max_order, backoff_depth=self.backoff_depth,
                depth=self.prefetch_depth, workers=self.workers, seq_len=self.block_size,
            )

    # -- sampling -----------------------------------------------------------
    def sample_tokens(self) -> np.ndarray:
        """One (batch_size, block_size + 1) uint16 window. Called by the prefetch worker."""
        width = self.block_size + 1
        out = np.empty((self.batch_size, width), dtype=np.uint16)
        # np.memmap is reopened every call, matching nanoGPT's get_batch: a long-lived memmap of a
        # multi-GB corpus is what makes the page cache thrash on this box.
        self._memmaps.clear()
        for row in range(self.batch_size):
            path = self.paths[int(self._rng.integers(len(self.paths)))]
            n_tokens = os.path.getsize(path) // 2
            if self.limit_tokens is not None:
                n_tokens = min(n_tokens, self.limit_tokens)
            start = int(self._rng.integers(0, n_tokens - width))
            data = np.memmap(path, dtype=np.uint16, mode="r")
            self._memmaps.append(data)
            out[row] = data[start:start + width]
        return out

    # -- batch assembly -----------------------------------------------------
    def _to_device(self, tokens: np.ndarray, targets: Optional[np.ndarray], res) -> Dict:
        if targets is None:
            targets = tokens[:, 1:]
            tokens = tokens[:, : self.block_size]
        return to_device(self._buffers.stage(tokens, targets, res), self.device, self.score_dtype)

    def next_batch(self) -> Optional[Dict]:
        """The next batch: from the prefetch pipeline if one is running, else looked up now."""
        if self._prefetcher is not None:
            item = self._prefetcher.next()
            if item is None:
                return None
            tokens, targets, res = item
            return self._to_device(tokens, targets, res)
        tokens = self.sample_tokens()
        res = mix(self.tables, self.weights, tokens[:, : self.block_size], j=self.j, k=self.k,
                  max_order=self.max_order, backoff_depth=self.backoff_depth)
        # the prefetch path hands back (tokens, targets) already sliced; here the whole
        # block_size + 1 window goes in and _to_device does the slicing
        return self._to_device(tokens, None, res)

    def close(self) -> None:
        if self._prefetcher is not None:
            self._prefetcher.close()
            self._prefetcher = None
        self._memmaps.clear()

    def __enter__(self) -> "CountDataPipeline":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_tables(directories: Sequence[str]) -> List[CountTable]:
    return [CountTable(d) for d in directories if d]


def table_weights(lam: float, n_tables: int) -> List[float]:
    """Sec. 5.1's interpolation weights for one general table and one domain table."""
    if n_tables == 1:
        return [1.0]
    if n_tables != 2:
        raise ValueError("this implementation mixes exactly two tables: general and domain")
    return [1.0 - float(lam), float(lam)]
