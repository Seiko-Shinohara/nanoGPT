"""COUNT n-gram tables on host RAM: the Python front end for count_native/libcount_lookup.so.

The paper's whole premise is that the next-token statistics live in host memory and are searched
by a compiled multithreaded lookup (Sec. 3, Appendix B.2). So this module owns no data of its
own: it opens a table directory written by ``count_native/count_build`` and hands numpy buffers to
the C++ reader through ctypes. ctypes releases the GIL for the call, which is what lets
``count.prefetch`` search the next batch while the GPU is still busy with the current one.

Two knobs from the paper stay inference-time decisions rather than build-time ones:

* ``k``   candidates kept per order for the embedding sum (paper: tables store K = 32, training
          and inference use K = 8), and
* ``j``   how many stored suffix orders contribute (paper: J = 6, ablated against J = 1),

plus ``lambda`` for mixing a domain table into the general table at inference (Sec. 5.1), which
needs no retraining and no new table format -- see ``mix``.
"""

from __future__ import annotations

import ctypes
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

_INVALID = -1


def _default_lib_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "count_native", "libcount_lookup.so")


_LIB: Optional[ctypes.CDLL] = None


def load_library(path: Optional[str] = None) -> ctypes.CDLL:
    """Load (once) and bind the native lookup library."""
    global _LIB
    if _LIB is not None and path is None:
        return _LIB
    path = path or os.environ.get("COUNT_LIBRARY") or _default_lib_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run ./count_native/build.sh (or set COUNT_LIBRARY)"
        )
    lib = ctypes.CDLL(path)

    c_i32 = ctypes.POINTER(ctypes.c_int32)
    c_f32 = ctypes.POINTER(ctypes.c_float)
    c_u16 = ctypes.POINTER(ctypes.c_uint16)
    c_f64 = ctypes.POINTER(ctypes.c_double)
    handle = ctypes.c_void_p

    lib.cl_open.restype = handle
    lib.cl_open.argtypes = [ctypes.c_char_p]
    lib.cl_close.restype = None
    lib.cl_close.argtypes = [handle]

    lib.cl_max_order.restype = ctypes.c_int
    lib.cl_max_order.argtypes = [handle]
    lib.cl_k_store.restype = ctypes.c_int
    lib.cl_k_store.argtypes = [handle]
    lib.cl_vocab.restype = ctypes.c_int
    lib.cl_vocab.argtypes = [handle]
    lib.cl_backoff.restype = ctypes.c_double
    lib.cl_backoff.argtypes = [handle]
    lib.cl_corpus_tokens.restype = ctypes.c_uint64
    lib.cl_corpus_tokens.argtypes = [handle]
    lib.cl_rows.restype = ctypes.c_uint64
    lib.cl_rows.argtypes = [handle, ctypes.c_int]
    lib.cl_log10_min.restype = ctypes.c_double
    lib.cl_log10_min.argtypes = [handle, ctypes.c_int]
    lib.cl_history_count.restype = ctypes.c_int32
    lib.cl_history_count.argtypes = [handle, ctypes.c_int, ctypes.c_int64]
    lib.cl_set_threads.restype = None
    lib.cl_set_threads.argtypes = [ctypes.c_int]

    lib.cl_lookup_batch.restype = ctypes.c_int
    lib.cl_lookup_batch.argtypes = [handle, c_u16, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int, c_i32, c_f32, c_i32,
                                    c_i32]
    lib.cl_lookup_mix.restype = ctypes.c_int
    # arg 0 is a CountHandle** array: passed by address, which ctypes can check as c_void_p
    lib.cl_lookup_mix.argtypes = [ctypes.c_void_p, ctypes.c_int, c_f64, c_u16, ctypes.c_int,
                                  ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_int, c_i32, c_f32, c_i32, c_i32]
    lib.cl_lookup_next.restype = ctypes.c_int
    lib.cl_lookup_next.argtypes = [handle, c_u16, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int, c_i32, c_f32, c_i32, c_i32]
    lib.cl_lookup_next_mix.restype = ctypes.c_int
    lib.cl_lookup_next_mix.argtypes = [ctypes.c_void_p, ctypes.c_int, c_f64, c_u16, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                       c_i32, c_f32, c_i32, c_i32]
    _LIB = lib
    return lib


@dataclass
class CountResult:
    """Lookups for every position of a batch, ordered longest suffix first.

    candidates: (rows, seq_len, j_max, k) int32 token ids, -1 where a row is shorter than k
    scores:     (rows, seq_len, j_max, k) float32 stupid-backoff scores S(w | h), not renormalized
    n_found:    (rows, seq_len) int32, how many of the j_max orders resolved to a stored row
                (at position 0 the context is one token long, so at most the order-1 row can hit)
    orders:     (rows, seq_len, j_max) int32, the n-gram order each slot came from, 0 when unused
    """

    candidates: np.ndarray
    scores: np.ndarray
    n_found: np.ndarray
    orders: Optional[np.ndarray] = None

    def coverage(self) -> float:
        """Fraction of positions whose longest stored suffix reached the maximum order asked for."""
        if self.orders is None:
            return float((self.n_found > 0).mean())
        return float((self.orders[..., 0] > 0).mean())

    def mean_candidates(self) -> float:
        """Mean stored candidates per resolved (position, order) slot, always <= k."""
        return float((self.candidates >= 0).sum() / max(1, int(self.n_found.sum())))

    def candidates_per_position(self) -> float:
        """Mean candidates summed over all contributing orders at a position (the K x J budget)."""
        return float((self.candidates >= 0).sum() / max(1, int(self.candidates.shape[0]
                                                              * self.candidates.shape[1])))


def set_threads(n: int) -> None:
    """How many CPU threads the compiled lookup may use (paper's Table 6 uses eight)."""
    load_library().cl_set_threads(int(n))


class CountTable:
    """One read-only n-gram continuation table resident in host RAM."""

    def __init__(self, directory: str, lib: Optional[ctypes.CDLL] = None):
        self.directory = os.path.abspath(directory)
        self.lib = lib or load_library()
        meta_json = os.path.join(self.directory, "meta.json")
        self.meta = {}
        if os.path.exists(meta_json):
            with open(meta_json) as f:
                self.meta = json.load(f)
        self._handle = self.lib.cl_open(self.directory.encode())
        if not self._handle:
            raise RuntimeError(f"could not open count table at {self.directory}")

    # -- metadata -----------------------------------------------------------
    @property
    def handle(self) -> int:
        if not self._handle:
            raise RuntimeError("count table is closed")
        return self._handle

    @property
    def max_order(self) -> int:
        return self.lib.cl_max_order(self.handle)

    @property
    def k_store(self) -> int:
        return self.lib.cl_k_store(self.handle)

    @property
    def vocab_size(self) -> int:
        return self.lib.cl_vocab(self.handle)

    @property
    def backoff(self) -> float:
        return self.lib.cl_backoff(self.handle)

    @property
    def corpus_tokens(self) -> int:
        return int(self.lib.cl_corpus_tokens(self.handle))

    def rows(self, order: int) -> int:
        return int(self.lib.cl_rows(self.handle, int(order)))

    def total_rows(self) -> int:
        return sum(self.rows(n) for n in range(1, self.max_order + 1))

    def host_bytes(self) -> int:
        """Size of the memory-mapped files: what the table costs in host RAM/page cache."""
        total = 0
        for name in os.listdir(self.directory):
            if name.endswith(".bin") or name.endswith(".json"):
                total += os.path.getsize(os.path.join(self.directory, name))
        return total

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self.lib.cl_close(self._handle)
            self._handle = None

    def __enter__(self) -> "CountTable":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def describe(self) -> str:
        lines = [
            f"count table {os.path.basename(self.directory)}  vocab={self.vocab_size} "
            f"orders=1..{self.max_order} K_store={self.k_store} backoff={self.backoff} "
            f"corpus_tokens={self.corpus_tokens:,}"
        ]
        for n in range(1, self.max_order + 1):
            lines.append(f"  order {n}: {self.rows(n):>12,} rows  log10_min={self.lib.cl_log10_min(self.handle, n): .3f}")
        lines.append(f"  host footprint {self.host_bytes() / 1e9:.3f} GB, {self.total_rows():,} rows")
        return "\n".join(lines)

    # -- lookup -------------------------------------------------------------
    def lookup_batch(
        self,
        tokens: np.ndarray,
        j: int = 6,
        k: int = 8,
        max_order: Optional[int] = None,
        backoff_depth: int = -1,
        out: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        workers: Optional[int] = None,
    ) -> CountResult:
        """Lookups for every position of every sequence in ``tokens`` (rows, seq_len) uint16.

        ``max_order`` caps which orders are searched (paper ablations use orders 1-6; J = 1 uses
        only the longest), ``backoff_depth`` truncates the Eq. 2 discount chain (-1 = follow it to
        the unigram row, 0 = no backoff at all).
        """
        if workers is not None:
            self.lib.cl_set_threads(int(workers))
        tokens = _as_uint16(tokens)
        rows, seq_len = tokens.shape
        j_max = min(int(j), max_order if max_order else self.max_order)
        cands, scores, nfound, orders = out if out else _alloc_out(rows, seq_len, j_max, k)
        rc = self.lib.cl_lookup_batch(
            self.handle,
            tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            rows, seq_len,
            int(max_order) if max_order else -1,
            j_max, int(k), int(backoff_depth),
            cands.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            nfound.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            orders.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        if rc != 0:
            raise RuntimeError("cl_lookup_batch failed (check token dtype/contiguity and j, k >= 1)")
        return CountResult(cands, scores, nfound, orders)

    def lookup_next(
        self,
        tail: Sequence[int],
        j: int = 6,
        k: int = 8,
        max_order: Optional[int] = None,
        backoff_depth: int = -1,
    ) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
        """Decode step: candidates for the token after ``tail`` (chronological, oldest first)."""
        tail_arr = np.asarray(tail, dtype=np.uint16)
        if tail_arr.ndim != 1 or tail_arr.size == 0:
            raise ValueError("tail must be a non-empty 1-D token list")
        j_max = min(int(j), max_order if max_order else self.max_order)
        cands = np.full((j_max, k), _INVALID, dtype=np.int32)
        scores = np.zeros((j_max, k), dtype=np.float32)
        nfound = np.zeros(1, dtype=np.int32)
        orders = np.zeros(j_max, dtype=np.int32)
        rc = self.lib.cl_lookup_next(
            self.handle,
            tail_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            int(tail_arr.size),
            int(max_order) if max_order else -1,
            j_max, int(k), int(backoff_depth),
            cands.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            nfound.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            orders.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        if rc != 0:
            raise RuntimeError("cl_lookup_next failed")
        return cands, scores, int(nfound[0]), orders


def mix(
    tables: Sequence[CountTable],
    weights: Sequence[float],
    tokens: np.ndarray,
    j: int = 6,
    k: int = 8,
    max_order: Optional[int] = None,
    backoff_depth: int = -1,
    workers: Optional[int] = None,
) -> CountResult:
    """Sec. 5.1 table mixing over a batch: S_lambda = (1-lambda) S_general + lambda S_domain.

    Weights need not sum to one; the merged scores are never renormalized, which is what makes
    swapping in a domain table change the *magnitude* of the COUNT embedding too.
    """
    if len(tables) != len(weights) or not tables:
        raise ValueError("tables and weights must be the same non-empty length")
    if workers is not None:
        load_library().cl_set_threads(int(workers))
    tokens = _as_uint16(tokens)
    rows, seq_len = tokens.shape
    j_max = min(int(j), max_order if max_order else max(t.max_order for t in tables))
    handles = (ctypes.c_void_p * len(tables))(*[t.handle for t in tables])
    w = (ctypes.c_double * len(weights))(*[float(x) for x in weights])
    cands, scores, nfound, orders = _alloc_out(rows, seq_len, j_max, k)
    rc = load_library().cl_lookup_mix(
        ctypes.addressof(handles), len(tables), w,
        tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        rows, seq_len,
        int(max_order) if max_order else -1,
        j_max, int(k), int(backoff_depth),
        cands.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        nfound.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        orders.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    if rc != 0:
        raise RuntimeError("cl_lookup_mix failed")
    return CountResult(cands, scores, nfound, orders)


def mix_next(
    tables: Sequence[CountTable],
    weights: Sequence[float],
    tail: Sequence[int],
    j: int = 6,
    k: int = 8,
    max_order: Optional[int] = None,
    backoff_depth: int = -1,
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Sec. 5.1 table mixing for a single decode step."""
    tail_arr = np.asarray(tail, dtype=np.uint16)
    if tail_arr.ndim != 1 or tail_arr.size == 0:
        raise ValueError("tail must be a non-empty 1-D token list")
    j_max = min(int(j), max_order if max_order else max(t.max_order for t in tables))
    handles = (ctypes.c_void_p * len(tables))(*[t.handle for t in tables])
    w = (ctypes.c_double * len(weights))(*[float(x) for x in weights])
    cands = np.full((j_max, k), _INVALID, dtype=np.int32)
    scores = np.zeros((j_max, k), dtype=np.float32)
    nfound = np.zeros(1, dtype=np.int32)
    orders = np.zeros(j_max, dtype=np.int32)
    rc = load_library().cl_lookup_next_mix(
        ctypes.addressof(handles), len(tables), w,
        tail_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        int(tail_arr.size),
        int(max_order) if max_order else -1,
        j_max, int(k), int(backoff_depth),
        cands.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        nfound.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        orders.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    if rc != 0:
        raise RuntimeError("cl_lookup_next_mix failed")
    return cands, scores, int(nfound[0]), orders


# ---------------------------------------------------------------- helpers

def _as_uint16(tokens: np.ndarray) -> np.ndarray:
    if not isinstance(tokens, np.ndarray):
        tokens = np.asarray(tokens)
    if tokens.dtype != np.uint16:
        tokens = tokens.astype(np.uint16)
    if not tokens.flags["C_CONTIGUOUS"]:
        tokens = np.ascontiguousarray(tokens)
    if tokens.ndim != 2:
        raise ValueError("expected a (rows, seq_len) token array")
    return tokens


def _alloc_out(rows: int, seq_len: int, j_max: int, k: int):
    return (
        np.full((rows, seq_len, j_max, k), _INVALID, dtype=np.int32),
        np.zeros((rows, seq_len, j_max, k), dtype=np.float32),
        np.zeros((rows, seq_len), dtype=np.int32),
        np.zeros((rows, seq_len, j_max), dtype=np.int32),
    )


def open_tables(specs: Sequence[str]) -> List[CountTable]:
    return [CountTable(s) for s in specs]
