"""The learned n-gram embedding module (paper Sec. 3.3, Eq. 3): T_{n,r} tables plus P_{n,r} projections.

This is a *baseline*, not part of COUNT: it is the method the paper measures COUNT against, and the
comparison is about where the parameters have to live. These tables are learned jointly with the
backbone, so they occupy GPU memory for the table weights *and* two AdamW moments per row, and a new
run has to learn them again. COUNT's tables are built offline, carry no optimizer state, and sit on
host RAM.

Two properties the tests check, because they are the ones a shortcut implementation gets wrong:
* only the retrieved rows receive a gradient (`sparse=True` on the embeddings, which is the paper's
  "sparse optimizer updates for only the retrieved embedding rows": full-table updates ran at 4.4% of
  standard throughput, retrieved-rows-only at 96.3%);
* `host_tables=True` gathers rows on the CPU and transfers only what was retrieved, which is the
  paper's inference-time host offloading. It computes the same q_t, so it must agree numerically.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from count_ngram.hash import make_moduli, ngram_keys, rows_for_ratio, table_param_count


class LearnedNgramEmbeddings(nn.Module):
    """q_t = sum_{n,r} P_{n,r} T_{n,r}[key_{n,r}(t)], added to the token embedding (Eq. 3)."""

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        orders: Sequence[int] = (2, 3),
        n_tables: int = 4,
        rows_per_table: int = 0,
        width_div: int = 8,
        sparse: bool = True,
        host_tables: bool = False,
        init_std: float = 0.02,
    ):
        super().__init__()
        orders = list(orders)
        if not orders or min(orders) < 2:
            raise ValueError("learned n-gram orders start at 2 (a unigram is the token embedding)")
        if n_embd % width_div:
            raise ValueError(f"n_embd={n_embd} is not divisible by width_div={width_div}")
        if rows_per_table < 1:
            raise ValueError("rows_per_table must be at least 1")
        self.vocab_size = int(vocab_size)
        self.n_embd = int(n_embd)
        self.orders = orders
        self.n_tables = int(n_tables)
        self.width = n_embd // width_div
        self.sparse = bool(sparse)
        self.host_tables = bool(host_tables)
        # The requested row count. The actual M_{n,r} are the next primes at or above it (see
        # make_moduli), so table_rows() is the honest total and this is what a checkpoint records.
        self.rows_per_table = int(rows_per_table)
        self.moduli = make_moduli(rows_per_table, self.n_tables, len(self.orders))

        flat = [M for row in self.moduli for M in row]
        self.tables = nn.ModuleList([nn.Embedding(M, self.width, sparse=self.sparse) for M in flat])
        self.projs = nn.ModuleList([nn.Linear(self.width, self.n_embd, bias=False) for _ in flat])
        for table, proj in zip(self.tables, self.projs):
            nn.init.normal_(table.weight, mean=0.0, std=init_std)
            nn.init.normal_(proj.weight, mean=0.0, std=init_std)

    # -- bookkeeping ---------------------------------------------------------
    @property
    def num_tables(self) -> int:
        return len(self.tables)

    def table_weights(self) -> List[nn.Parameter]:
        """The T_{n,r} rows. These are the parameters the paper exempts from weight decay and updates
        row-wise; the projections are ordinary dense weights and keep the backbone's decay."""
        return [t.weight for t in self.tables]

    def is_table_param(self, p: nn.Parameter) -> bool:
        return any(p is w for w in self.table_weights())

    def param_counts(self) -> Dict[str, int]:
        tables = sum(t.weight.numel() for t in self.tables)
        projs = sum(p.weight.numel() for p in self.projs)
        return {"table_params": tables, "projection_params": projs, "total_params": tables + projs}

    def table_rows(self) -> int:
        return sum(t.weight.shape[0] for t in self.tables)

    # -- keys and forward ----------------------------------------------------
    def keys(self, idx: torch.Tensor) -> torch.Tensor:
        return ngram_keys(idx, self.orders, self.moduli, self.vocab_size)

    def forward(self, idx: torch.Tensor, keys: Optional[torch.Tensor] = None) -> torch.Tensor:
        """q_t for every position: (b, t, n_embd)."""
        if keys is None:
            keys = self.keys(idx)
        device = idx.device
        q = None
        i = 0
        for oi in range(len(self.orders)):
            for r in range(self.n_tables):
                table, proj = self.tables[i], self.projs[i]
                i += 1
                key = keys[:, :, oi, r]
                if self.host_tables and device.type == "cuda":
                    # Gather on host, ship only the retrieved rows. Autograd carries the gradient back
                    # to the host-resident table, so the table and its optimizer state never need to
                    # fit in VRAM -- the same argument COUNT makes about its score tables, but paying
                    # for a copy of every retrieved row on every step.
                    rows = table(key.cpu()).to(device, non_blocking=True)
                else:
                    rows = table(key)
                out = proj(rows)
                q = out if q is None else q + out
        return q

    # -- construction helpers ------------------------------------------------
    def to_host(self) -> "LearnedNgramEmbeddings":
        """Move the T_{n,r} tables to host RAM and leave the projections on the device.

        model.to(device) moves everything, so this has to be called *after* it. It is the baseline's
        answer to COUNT's host-residency argument: the tables can live off-GPU, but unlike COUNT's
        score tables they are being updated, so every retrieved row crosses the PCIe boundary in both
        directions on every step, and its optimizer state lives on the host too.
        """
        for t in self.tables:
            t.to('cpu')
        return self

    @classmethod
    def for_ratio(cls, config, other_params: int, input_embedding_params: int,
                  ratio: float = 1.0, **kwargs) -> "LearnedNgramEmbeddings":
        """Build with rows_per_table set by the paper's 1:1 input-embedding-to-backbone rule."""
        orders = list(range(2, config.ngram_orders + 1))
        rows = rows_for_ratio(other_params, input_embedding_params, config.n_embd,
                              config.ngram_width_div, len(orders) * config.ngram_tables, ratio)
        return cls(config.vocab_size, config.n_embd, orders=orders, n_tables=config.ngram_tables,
                   rows_per_table=rows, width_div=config.ngram_width_div, **kwargs)


def ngram_from_config(config, other_params: Optional[int] = None,
                      input_embedding_params: Optional[int] = None
                      ) -> Optional[LearnedNgramEmbeddings]:
    """Build the baseline from a GPTConfig, or return None when this is not a learned-n-gram run.

    ngram_rows=0 means "derive it from the 1:1 ratio", which needs the parameter counts of the rest of
    the model: GPT passes them in after it has built its transformer, its head and its COUNT
    amplification layer, and before this module exists.
    """
    if not getattr(config, "ngram_orders", 0) or config.ngram_orders < 2:
        return None
    orders = list(range(2, config.ngram_orders + 1))
    rows = int(config.ngram_rows)
    if rows <= 0:
        if other_params is None or input_embedding_params is None:
            raise ValueError("ngram_rows=0 asks for the 1:1 ratio, which needs the model's own "
                             "parameter counts: build this through GPT, not from_config")
        rows = rows_for_ratio(other_params, input_embedding_params, config.n_embd,
                              config.ngram_width_div, len(orders) * config.ngram_tables,
                              config.ngram_ratio)
    return LearnedNgramEmbeddings(
        config.vocab_size, config.n_embd, orders=orders, n_tables=config.ngram_tables,
        rows_per_table=rows, width_div=config.ngram_width_div,
        sparse=config.ngram_sparse, host_tables=config.ngram_host_tables,
    )


__all__ = ["LearnedNgramEmbeddings", "ngram_from_config", "table_param_count"]
