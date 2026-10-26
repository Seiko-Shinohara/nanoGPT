"""Prefetching COUNT lookups so table search overlaps GPU compute (paper Appendix B.2).

The lookup is host work: hashing a suffix, a binary search over a sorted key array, a merge over
the backoff chain. On the GPU the model is doing matmuls. The paper's throughput result -- COUNT
at 93-98% of a standard transformer with a 797 GB table -- depends on overlapping the two, and
"prefetch prepares lookups while the GPU processes the preceding batch" is the named optimization.

Concretely, this class runs a small bounded pipeline:

    sampler (returns uint16 token windows)  ->  C++ lookup on N CPU threads  ->  pinned buffers
                                                                            ->  the caller hands
                                                                                them to the GPU
                                                                                with
                                                                                non_blocking=True

Because ctypes releases the GIL, the C++ lookup threads really do run in parallel with the
previous batch's backward pass. ``depth=1`` already hides most of the lookup cost; the paper's
prefill experiment used ten batches of lookahead, which ``depth`` can reproduce.

Buffers are reused between batches. Pinned (page-locked) host memory is what makes the H2D copy
asynchronous, so a batch of candidates costs a copy, not a stall.
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from count.table import CountTable, mix

TokenSampler = Callable[[], np.ndarray]


class CountPrefetcher:
    """Look up ``depth`` batches ahead of the consumer.

    ``sampler`` returns a (batch, seq_len) uint16 numpy array; it is called from this class's
    worker thread, so the sampler itself must be thread-safe (nanoGPT's sampler indexes a
    memory-mapped corpus with a private RNG, which is fine when only the worker calls it).
    """

    def __init__(
        self,
        tables: Sequence[CountTable],
        weights: Optional[Sequence[float]],
        sampler: TokenSampler,
        j: int = 6,
        k: int = 8,
        max_order: Optional[int] = None,
        backoff_depth: int = -1,
        depth: int = 2,
        workers: int = 8,
        seq_len: Optional[int] = None,
    ):
        self.tables = list(tables)
        self.weights = list(weights) if weights is not None else [1.0]
        if len(self.weights) != len(self.tables):
            raise ValueError("weights must match tables")
        self.sampler = sampler
        self.j = int(j)
        self.k = int(k)
        self.max_order = max_order
        self.backoff_depth = int(backoff_depth)
        self.depth = max(1, int(depth))
        self.workers = int(workers)
        # nanoGPT samples block_size + 1 tokens so that targets are the window shifted by one. Only
        # the first seq_len of them are ever *inputs*, so the lookup is computed on that prefix and
        # the extra token travels with the batch as target material (see _run below).
        self.seq_len = int(seq_len) if seq_len is not None else None

        self._q: "queue.Queue" = queue.Queue(maxsize=self.depth)
        self._stop = threading.Event()
        self._error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, name="count-prefetch", daemon=True)
        self._thread.start()

    # -- producer -----------------------------------------------------------
    def _run(self) -> None:
        try:
            from count.table import set_threads
            set_threads(self.workers)
            while not self._stop.is_set():
                window = self.sampler()
                if window is None:
                    break
                window = np.ascontiguousarray(window, dtype=np.uint16)
                if self.seq_len is not None and window.shape[1] > self.seq_len:
                    tokens = window[:, :self.seq_len]
                    targets = window[:, 1:self.seq_len + 1]
                else:
                    tokens, targets = window, None
                res = mix(
                    self.tables, self.weights, tokens,
                    j=self.j, k=self.k, max_order=self.max_order,
                    backoff_depth=self.backoff_depth,
                )
                self._q.put((tokens, targets, res))
        except BaseException as exc:  # surfaced by the consumer on the next get()
            self._error = exc
            try:
                self._q.put(None)
            except Exception:
                pass

    # -- consumer -----------------------------------------------------------
    def next(self):
        item = self._q.get()
        if item is None:
            if self._error is not None:
                raise self._error
            return None
        return item

    def close(self) -> None:
        self._stop.set()
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=5.0)

    def __enter__(self) -> "CountPrefetcher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ------------------------------------------------------------------ host -> device

class PinnedBuffers:
    """Reusable page-locked host buffers so a prefetched batch costs a copy, not a stall.

    Allocated lazily on the first batch and reused while the shapes match; the shapes only change
    at the end of an epoch in practice, where reallocation is cheap.
    """

    def __init__(self):
        self._bufs = {}

    def get(self, name: str, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        import torch

        key = (name, tuple(shape), dtype.name)
        entry = self._bufs.get(key)
        if entry is None:
            t = torch.empty(shape, dtype=_torch_dtype(dtype), pin_memory=True)
            entry = t.numpy()
            self._bufs[key] = entry
        return entry

    def stage(self, tokens: np.ndarray, targets: np.ndarray, res) -> tuple:
        """Copy a prefetched batch into pinned buffers ready for non-blocking .to(device)."""
        tok_buf = self.get("tokens", tokens.shape, np.dtype(np.int32))
        tok_buf[...] = tokens
        tgt_buf = self.get("targets", targets.shape, np.dtype(np.int32))
        tgt_buf[...] = targets
        c_buf = self.get("cands", res.candidates.shape, np.dtype(np.int32))
        c_buf[...] = res.candidates
        s_buf = self.get("scores", res.scores.shape, np.dtype(np.float32))
        s_buf[...] = res.scores
        n_buf = self.get("nfound", res.n_found.shape, np.dtype(np.int32))
        n_buf[...] = res.n_found
        return tok_buf, tgt_buf, c_buf, s_buf, n_buf


def _torch_dtype(dtype: np.dtype):
    import torch

    return {
        "int32": torch.int32,
        "int64": torch.int64,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[dtype.name]


def to_device(arrays, device, score_dtype=None) -> Dict:
    """Move a staged (pinned) batch onto the GPU without synchronizing.

    Returns the dict model.forward(..., count=...) wants: idx, targets, candidates (long token ids,
    -1 where a stored row ran out of candidates), scores (count scores, 0.0 on those padded slots),
    n_found (how many suffix orders resolved, read only when count_norm == 'nfound').
    """
    import torch

    tok_buf, tgt_buf, c_buf, s_buf, n_buf = arrays
    idx = torch.from_numpy(tok_buf).to(device, non_blocking=True).long()
    targets = torch.from_numpy(tgt_buf).to(device, non_blocking=True).long()
    candidates = torch.from_numpy(c_buf).to(device, non_blocking=True).long()
    scores = torch.from_numpy(s_buf).to(device, non_blocking=True)
    if score_dtype is not None:
        scores = scores.to(score_dtype)
    n_found = torch.from_numpy(n_buf).to(device, non_blocking=True)
    return {"idx": idx, "targets": targets, "candidates": candidates, "scores": scores,
            "n_found": n_found}
