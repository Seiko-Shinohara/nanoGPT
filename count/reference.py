"""A deliberately slow, transparent numpy/Python implementation of COUNT's lookup.

This exists for one reason: to check count_native/libcount_lookup.so. The C++ reader is the thing
that has to be fast (hashed keys, binary search over sorted rows, four flat arrays, OpenMP), and
fast code is easy to get subtly wrong -- wrong suffix direction, an off-by-one in the backoff
chain, a renormalization that the paper forbids, a score decoded with the wrong quantization
range. The reference here re-derives Eq. 1-2 straight from the paper in about a hundred lines of
Python, reads the same four files, and must agree with the C++ result on candidates, order and
scores. Run it with ``python -m count.selfcheck``.

It reproduces the C++ semantics exactly, including the two places where we chose an approximation:
  * a stored row holds only *observed* continuations, so backoff candidates come from the suffix
    chain at read time rather than from a merged row baked in at build time;
  * a candidate absent from a row because the row was truncated to K_store is treated the same as
    a candidate with count zero, i.e. the chain backs off past it.
"""

from __future__ import annotations

import json
import os
import struct
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MAGIC = 0x31425443
VERSION = 1
KEY_BITS = 48
KEY_MASK = (1 << KEY_BITS) - 1
U64 = (1 << 64) - 1
INVALID = -1


def _smix(z: int) -> int:
    z = (z + 0x9E3779B97F4A7C15) & U64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & U64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & U64
    return z ^ (z >> 31)


def hist1(x: int) -> int:
    return _smix((x + 0x243F6A8885A308D3) & U64) & KEY_MASK


def hist_extend(h: int, older: int) -> int:
    z = ((h * 0x9E3779B97F4A7C15) & U64) ^ (((older + 0x1234) * 0xD6E8FEB86659FD93) & U64)
    z ^= 0xA0761D6478BD642F
    return _smix(z & U64) & KEY_MASK


def pair_key(hist: int, w: int) -> int:
    return ((hist << 16) | (w & 0xFFFF)) & U64


class ReferenceTable:
    """Everything the C++ reader knows, but as numpy arrays plus a searchsorted row finder."""

    def __init__(self, directory: str):
        self.directory = os.path.abspath(directory)
        with open(os.path.join(self.directory, "meta.bin"), "rb") as f:
            blob = f.read()
        magic, version, vocab, max_order, k_store, n_orders = struct.unpack_from("<IIIIII", blob, 0)
        if magic != MAGIC or version != VERSION:
            raise ValueError(f"{directory}: not a count table (magic={magic:#x} version={version})")
        backoff, corpus_tokens = struct.unpack_from("<dQ", blob, 24)
        self.vocab = vocab
        self.max_order = max_order
        self.k_store = k_store
        self.n_orders = n_orders
        self.backoff = backoff
        self.corpus_tokens = corpus_tokens

        off = 40
        self.keys: Dict[int, np.ndarray] = {}
        self.cands: Dict[int, np.ndarray] = {}
        self.scores: Dict[int, np.ndarray] = {}
        self.rows: Dict[int, int] = {}
        self.log10_min: Dict[int, float] = {}
        for _ in range(n_orders):
            order, nrows = struct.unpack_from("<IQ", blob, off)
            lmin = struct.unpack_from("<d", blob, off + 12)[0]
            off += 36   # order u32 + rows u64 + log10_min f64 + hmin u32 + pmin u32 + distinct u64
            self.rows[order] = int(nrows)
            self.log10_min[order] = lmin
            keys = np.fromfile(os.path.join(self.directory, f"order{order}.keys.bin"), dtype=np.uint64)
            cands = np.fromfile(os.path.join(self.directory, f"order{order}.cands.bin"), dtype=np.int32)
            codes = np.fromfile(os.path.join(self.directory, f"order{order}.codes.bin"), dtype=np.uint8)
            # 256 levels spanning [log10_min, 0], decoded and exponentiated: back to S(w | h).
            decay = (-lmin / 255.0) if lmin < 0 else 0.0
            lut = np.exp(np.log(10.0) * (lmin + np.arange(256, dtype=np.float64) * decay))
            nrows = int(nrows)
            self.keys[order] = keys
            self.cands[order] = cands.reshape(nrows, k_store)
            self.scores[order] = lut[codes.reshape(nrows, k_store).astype(np.int64)]

        uni_lmin = struct.unpack_from("<d", blob, off)[0]
        uni_k = struct.unpack_from("<I", blob, off + 8)[0]
        ids = np.frombuffer(blob, dtype=np.int32, count=uni_k, offset=off + 12)
        codes = np.frombuffer(blob, dtype=np.uint8, count=uni_k, offset=off + 12 + 4 * uni_k)
        decay = (-uni_lmin / 255.0) if uni_lmin < 0 else 0.0
        self.uni_ids = ids.tolist()
        self.uni_scores = np.exp(np.log(10.0) * (uni_lmin + codes.astype(np.float64) * decay))

        with open(os.path.join(self.directory, "meta.json")) as f:
            self.meta = json.load(f)

    # -- row access ---------------------------------------------------------
    def row(self, order: int, key: int) -> Optional[Tuple[List[int], List[float]]]:
        keys = self.keys.get(order)
        if keys is None or len(keys) == 0:
            return None
        i = int(np.searchsorted(keys, np.uint64(key)))
        if i >= self.rows[order] or int(keys[i]) != key:
            return None
        cands = self.cands[order][i]
        scores = self.scores[order][i]
        out_c, out_s = [], []
        for c, s in zip(cands, scores):
            if int(c) == INVALID:
                break
            out_c.append(int(c))
            out_s.append(float(s))
        return out_c, out_s

    # -- Eq. 2 at read time ------------------------------------------------
    def candidates(self, context: Sequence[int], order: int, backoff_depth: int = -1
                   ) -> List[Tuple[int, float]]:
        """Merged candidate set for one history of the given order, over this one table."""
        return _merge_tables([self], [1.0], context, order, backoff_depth)

    def lookup_position(self, context: Sequence[int], j: int = 6, k: int = 8,
                        max_order: Optional[int] = None, backoff_depth: int = -1
                        ) -> Tuple[List[List[int]], List[List[float]], List[int], int]:
        """The J longest stored suffixes, longest first; each row truncated to its top-k."""
        limit = self.max_order if max_order is None else min(max_order, self.max_order)
        limit = min(limit, len(context))
        rows_c: List[List[int]] = []
        rows_s: List[List[float]] = []
        orders: List[int] = []
        for order in range(limit, 0, -1):
            if len(orders) >= j:
                break
            suffix = context[len(context) - order:]
            key = _hist(suffix)
            if self.row(order, key) is None:
                continue
            merged = _merge_tables([self], [1.0], context, order, backoff_depth)
            top = merged[:k]
            rows_c.append([t for t, _ in top])
            rows_s.append([s for _, s in top])
            orders.append(order)
        return rows_c, rows_s, orders, len(orders)


def _hist(tokens: Sequence[int]) -> int:
    h = hist1(int(tokens[-1]))
    for i in range(len(tokens) - 2, -1, -1):
        h = hist_extend(h, int(tokens[i]))
    return h


def _merge_tables(tables: Sequence[ReferenceTable], weights: Sequence[float],
                  context: Sequence[int], order: int, backoff_depth: int) -> List[Tuple[int, float]]:
    """Sec. 5.1: union the candidate sets, interpolate the scores, never renormalize."""
    merged: Dict[int, float] = {}
    depth_limit = order if backoff_depth < 0 else min(order, backoff_depth)
    for table, weight in zip(tables, weights):
        if order > table.max_order:
            continue
        seen: set = set()
        for depth in range(depth_limit + 1):
            length = order - depth
            discount = table.backoff ** depth
            if length == 0:
                for tok, score in zip(table.uni_ids, table.uni_scores):
                    if tok in seen:
                        continue
                    seen.add(tok)
                    merged[tok] = merged.get(tok, 0.0) + weight * discount * float(score)
                break
            suffix = context[len(context) - length:]
            hit = table.row(order=length, key=_hist(suffix))
            if hit is not None:
                for tok, score in zip(*hit):
                    if tok in seen:
                        continue
                    seen.add(tok)
                    merged[tok] = merged.get(tok, 0.0) + weight * discount * float(score)
    # A candidate whose interpolated score is exactly 0 was contributed only by a table with weight
    # 0 (lambda = 0 or 1). It adds nothing to Eq. 1 and would make a lambda = 0 mix differ from
    # simply using one table, so it is dropped -- the C++ reader does the same in one comparison.
    return sorted([(t, v) for t, v in merged.items() if v > 0.0], key=lambda kv: (-kv[1], kv[0]))


def mix_lookup(tables: Sequence[ReferenceTable], weights: Sequence[float], context: Sequence[int],
               j: int = 6, k: int = 8, max_order: Optional[int] = None,
               backoff_depth: int = -1) -> Tuple[List[List[int]], List[List[float]], List[int], int]:
    """Python twin of count.table.mix for the selfcheck."""
    limit = max(t.max_order for t in tables) if max_order is None else min(max_order, max(
        t.max_order for t in tables))
    limit = min(limit, len(context))
    rows_c: List[List[int]] = []
    rows_s: List[List[float]] = []
    orders: List[int] = []
    for order in range(limit, 0, -1):
        if len(orders) >= j:
            break
        key = _hist(context[len(context) - order:])
        # only a table with nonzero weight may make this order one of the J resolved slots
        if all(w == 0.0 or t.row(order, key) is None for t, w in zip(tables, weights)):
            continue
        top = _merge_tables(tables, weights, context, order, backoff_depth)[:k]
        rows_c.append([t for t, _ in top])
        rows_s.append([s for _, s in top])
        orders.append(order)
    return rows_c, rows_s, orders, len(orders)
