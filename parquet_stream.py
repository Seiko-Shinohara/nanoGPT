"""
Streaming parquet loader for nanoGPT: reads *.parquet files in filename
order, tokenizes the text column on the fly with a tiktoken encoder, and
serves random (x, y) token windows for GPT training.

A background producer thread keeps a rolling in-memory buffer of tokenized
chunks full (with worker threads for parallel tokenization - tiktoken
releases the GIL), so the training loop is never blocked on disk I/O or
tokenization. A token budget caps how many tokens each split streams, which
is how you pick "how many tokens to train on" for a parquet dataset.
"""
import os
import time
import glob
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq
import tiktoken


def _extract_text(value):
    # FineWeb(-Edu) stores the text as a struct: {'text': '...'}; plain strings work too
    if isinstance(value, dict):
        value = value.get('text', '')
    return value if isinstance(value, str) else ''


def _tokenize_texts(enc, texts):
    """Tokenize a batch of documents; one end-of-text token per non-empty doc."""
    ids = []
    for t in texts:
        t = _extract_text(t)
        if not t:
            continue
        ids.extend(enc.encode_ordinary(t))
        ids.append(enc.eot_token)
    return np.array(ids, dtype=np.uint16)


def estimate_file_tokens(path, text_column, enc, sample_rows=2000):
    """Rough estimate of the token count of one parquet file.

    Reads a small sample of rows, tokenizes it, and extrapolates to the
    file's total row count (from metadata; no full file read).
    """
    pf = pq.ParquetFile(path)
    total_rows = pf.metadata.num_rows
    if total_rows == 0:
        return 0
    batch = next(pf.iter_batches(batch_size=min(sample_rows, total_rows),
                                columns=[text_column]), None)
    if batch is None:
        return 0
    ids = _tokenize_texts(enc, batch.to_pylist())
    if ids.size == 0:
        return 0
    return int(ids.size / batch.num_rows * total_rows)


def plan_parquet_splits(data_dir, text_column, tokenizer_name,
                        train_tokens=-1, val_tokens=1_000_000, sample_rows=2000):
    """Split the parquet files of a dataset dir into train/val file lists.

    Train streams the files from the start (filename order) up to
    `train_tokens`; val streams the last file(s) up to `val_tokens`, so
    the two splits never overlap. Returns (train_files, val_files, est_train_tokens).
    """
    files = sorted(glob.glob(os.path.join(data_dir, '*.parquet')))
    assert files, f"no *.parquet files found in {data_dir}"
    enc = tiktoken.get_encoding(tokenizer_name)
    ests = []
    for p in files:
        e = estimate_file_tokens(p, text_column, enc, sample_rows=sample_rows)
        ests.append(e)
        print(f"[parquet] {os.path.basename(p)}: {e:,} tokens (estimated)")
    total_est = sum(ests)
    print(f"[parquet] {len(files)} files, ~{total_est:,} tokens in total")

    # val: take whole files from the END until val_tokens is covered
    val_files = []
    acc = 0
    for p, e in zip(reversed(files), reversed(ests)):
        if acc >= val_tokens:
            break
        val_files.append(p)
        acc += e
    val_files = list(reversed(val_files))
    val_set = set(val_files)

    train_files = [p for p in files if p not in val_set]
    if not train_files:
        raise ValueError(
            f"val_tokens={val_tokens:,} leaves no files for training "
            f"(dataset is ~{total_est:,} tokens); lower val_tokens")
    est_train = sum(e for p, e in zip(files, ests) if p not in val_set)
    if train_tokens > 0 and train_tokens > est_train:
        print(f"[parquet] WARNING: train_tokens={train_tokens:,} exceeds the "
              f"~{est_train:,} estimated train tokens; the train stream ends early")
    return train_files, val_files, est_train


class ParquetStream:
    """Streams a list of parquet files and serves random (x, y) windows.

    The producer runs in a background thread and keeps up to `buffer_tokens`
    of recent tokens in memory; `workers` threads tokenize each batch in
    parallel (tiktoken releases the GIL). `get_batch` samples random windows
    from the buffer.
    """

    def __init__(self, files, text_column, tokenizer_name, token_budget=-1,
                 block_size=1024, batch_rows=20_000, buffer_tokens=256_000_000,
                 workers=4, label=''):
        self.files = files
        self.text_column = text_column
        self.token_budget = token_budget  # -1 = stream all files
        self.block_size = block_size
        self.batch_rows = batch_rows
        self.buffer_tokens_cap = buffer_tokens
        self.workers = workers
        self.label = label
        self.enc = tiktoken.get_encoding(tokenizer_name)
        self.chunks = deque()            # np.uint16 arrays of token ids
        self.buffer_tokens = 0
        self.tokens_produced = 0
        self.done = False
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.thread = threading.Thread(target=self._produce, daemon=True)

    # ---- producer ----

    def start(self):
        self.thread.start()

    def _produce(self):
        with ThreadPoolExecutor(max_workers=max(1, self.workers),
                                thread_name_prefix=f'parquet-{self.label}') as ex:
            try:
                for path in self.files:
                    pf = pq.ParquetFile(path)
                    for batch in pf.iter_batches(batch_size=self.batch_rows,
                                                columns=[self.text_column]):
                        if self.token_budget > 0 and self.tokens_produced >= self.token_budget:
                            break
                        texts = batch.to_pylist()
                        w = max(1, self.workers)
                        n = len(texts)
                        sub = [texts[i * n // w:(i + 1) * n // w] for i in range(w)]
                        parts = list(ex.map(lambda s: _tokenize_texts(self.enc, s), sub))
                        ids = np.concatenate(parts)
                        if ids.size == 0:
                            continue
                        if self.token_budget > 0:
                            ids = ids[:self.token_budget - self.tokens_produced]
                        with self.cv:
                            self.chunks.append(ids)
                            self.buffer_tokens += ids.size
                            self.tokens_produced += ids.size
                            # drop oldest chunks past the buffer cap (keep at least one)
                            while self.buffer_tokens > self.buffer_tokens_cap and len(self.chunks) > 1:
                                self.buffer_tokens -= self.chunks.popleft().size
                            self.cv.notify_all()
                    if self.token_budget > 0 and self.tokens_produced >= self.token_budget:
                        break
            finally:
                with self.cv:
                    self.done = True
                    self.cv.notify_all()

    # ---- consumer ----

    def get_batch(self, batch_size, timeout=300.0):
        """A random (x, y) batch: int64 numpy arrays of shape (batch_size, block_size)."""
        bs = self.block_size
        need = batch_size * (bs + 1)
        deadline = time.time() + timeout
        while True:
            with self.cv:
                if self.buffer_tokens >= need:
                    # pick a uniform random token position in the buffer
                    sizes = np.fromiter((c.size for c in self.chunks),
                                        dtype=np.int64, count=len(self.chunks))
                    cum = np.cumsum(sizes)
                    pos = np.random.randint(cum[-1], size=batch_size)
                    ci = np.searchsorted(cum, pos, side='right')
                    start = np.where(ci == 0, 0, np.take(cum, ci - 1))
                    off = pos - start
                    off = np.minimum(off, np.take(sizes, ci) - bs - 1)
                    x = np.empty((batch_size, bs), dtype=np.int64)
                    y = np.empty((batch_size, bs), dtype=np.int64)
                    for i in range(batch_size):
                        c = self.chunks[int(ci[i])]
                        o = int(off[i])
                        x[i] = c[o:o + bs]
                        y[i] = c[o + 1:o + 1 + bs]
                    return x, y
                if self.done:
                    raise StopIteration(
                        f"parquet stream '{self.label}' exhausted after "
                        f"{self.tokens_produced:,} tokens")
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError(
                        f"parquet stream '{self.label}' made no progress for {timeout:.0f}s")
                self.cv.wait(remaining)
