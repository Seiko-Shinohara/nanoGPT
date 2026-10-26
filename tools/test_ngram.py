"""Tests for the Section 3.3 baseline: learned n-gram embeddings (Eq. 3).

Everything runs on CPU with a toy model: this is a check that Eq. 3 is implemented as written and that
the sparse-optimizer path actually touches only the rows a batch retrieved. It says nothing about
whether the baseline works, which is what a training run is for.

    venv/bin/python tools/test_ngram.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from count_ngram.embed import LearnedNgramEmbeddings, ngram_from_config
from count_ngram.hash import make_moduli, ngram_keys, prime_at_least, rows_for_ratio, table_param_count
from count_ngram.optim import build_sparse_optimizer, dense_grad_params
from model import GPT, GPTConfig, NO_NGRAM

torch.manual_seed(0)

VOCAB = 11
failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        failures.append(name)


# 1. Eq. 3's key, computed the slow way, against the vectorized one.
#    key_{n,r}(t) = ( sum_{j=0}^{n-1} x_{t-j} |V|^j ) mod M_{n,r}
def slow_key(idx, t, n, M, vocab):
    acc = 0
    for j in range(n):
        src = t - j
        x = 0 if src < 0 else int(idx[0, src]) # left-pad, same rule as the lookup
        acc = (acc + x * (vocab ** j)) % M
    return acc

idx = torch.randint(0, VOCAB, (1, 7))
orders = [2, 3]
moduli = make_moduli(64, 2, len(orders))
keys = ngram_keys(idx, orders, moduli, VOCAB)
worst = 0
for oi, n in enumerate(orders):
    for r in range(2):
        M = moduli[oi][r]
        for t in range(idx.shape[1]):
            worst = max(worst, abs(int(keys[0, t, oi, r]) - slow_key(idx, t, n, M, VOCAB)))
check("1. keys match Eq. 3 written out by hand", worst == 0, f"max |difference| = {worst}")

# 2. The key is a function of the last n tokens only, and the padding rule matches the n-gram lookup.
k_last2 = int(keys[0, 5, 0, 0])
idx2 = idx.clone()
idx2[0, 0] = (idx2[0, 0] + 1) % VOCAB # a token 5 positions back
k2 = int(ngram_keys(idx2, orders, moduli, VOCAB)[0, 5, 0, 0])
check("2. the order-2 key ignores everything older than 2 tokens", k_last2 == k2)
short = ngram_keys(idx[:, -2:], orders, moduli, VOCAB)
hand = slow_key(idx[:, -2:], 1, 3, moduli[1][0], VOCAB) # (pad, x_{t-1}, x_t)
check("3. a window shorter than the order hashes the left-padded history it should",
      int(short[0, 1, 1, 0]) == hand, f"{int(short[0, 1, 1, 0])} vs {hand}")

# 4. Prime moduli: coprime across tables and orders, and at least as large as asked for.
moduli = make_moduli(1000, 4, 2)
flat = [M for row in moduli for M in row]
coprime = all(prime_at_least(M) == M for M in flat)
import math
pairwise = all(math.gcd(a, b) == 1 for i, a in enumerate(flat) for b in flat[i + 1:])
check("4. every M is prime and pairwise coprime (Eq. 3 wants coprime moduli)",
      coprime and pairwise, f"{flat}")

# 5. The 1:1 parameter rule of Sec. 3.3, and the size it cannot reach.
n_embd, width_div = 768, 8
width = n_embd // width_div
input_emb = 50257 * n_embd
backbone = 124_000_000 - input_emb
rows = rows_for_ratio(124_000_000, input_emb, n_embd, width_div, 8, ratio=1.0)
table_params = table_param_count(make_moduli(rows, 4, 2), width)
# The moduli are primes at or above the requested rows, so the tables come out a hair larger than
# the target; the check is that they land on it, not that they hit it to the parameter.
check("5. rows_for_ratio puts table parameters at the 1:1 target",
      abs(table_params - backbone) <= 0.002 * backbone,
      f"{rows} rows/table -> {table_params:,} table params vs backbone {backbone:,} "
      f"({100 * table_params / backbone - 100:+.2f}%)")
try:
    rows_for_ratio(12_900_000, 12_900_000, 256, 8, 8, 1.0)
    ok = False
except ValueError:
    ok = True
check("6. the 1:1 rule refuses instead of inventing a model when embeddings already outweigh it", ok)

# 7. Sparse gradients: only rows this batch retrieved get gradients, and only they move.
class Holder(torch.nn.Module):
    """Whatever GPT is to this code: an object with a .ngram attribute."""
    def __init__(self, ngram):
        super().__init__()
        self.ngram = ngram

emb = LearnedNgramEmbeddings(VOCAB, 32, orders=[2, 3], n_tables=2, rows_per_table=256,
                             width_div=4, sparse=True)
holder = Holder(emb)
opt = build_sparse_optimizer(holder, 1e-3)
idx = torch.randint(0, VOCAB, (2, 6))
keys = emb.keys(idx)
out = emb(idx, keys=keys)
assert out.shape == (2, 6, 32)
loss = out.square().sum()
loss.backward()
grads = [t.grad for t in emb.table_weights()]
touched = set()
for g in grads:
    assert g.is_sparse, "nn.Embedding(sparse=True) did not produce a sparse gradient"
    touched |= {int(i) for i in g.coalesce().indices()[0]}
before = emb.table_weights()[0].detach().clone()
opt.step()
opt.zero_grad(set_to_none=True)
after = emb.table_weights()[0].detach()
changed = {int(i) for i in (before - after).abs().sum(dim=1).nonzero().flatten()}
check("7. only retrieved rows have gradients and only retrieved rows move",
      changed.issubset(touched), f"{len(touched)} rows retrieved, {len(changed)} updated")
untouched = [int(i) for i in range(before.shape[0]) if i not in touched]
check("8. rows no context reached are bit-identical after the step",
      torch.equal(before[untouched], after[untouched]),
      f"{len(untouched)} of {before.shape[0]} rows untouched")
check("9. clip_grad_norm_ is only ever shown dense gradients",
      all(p.grad is None or not p.grad.is_sparse for p in dense_grad_params(holder))
      and all(id(p) not in {id(t) for t in emb.table_weights()} for p in dense_grad_params(holder)))

# 3b. Nothing from the right of a position leaks into its key.
head = idx[:1, :4]
same_prefix = torch.cat([head, torch.randint(0, VOCAB, (1, 5))], dim=1)
k_a = int(ngram_keys(head, orders, moduli, VOCAB)[0, 3, 1, 0])
k_b = int(ngram_keys(same_prefix, orders, moduli, VOCAB)[0, 3, 1, 0])
check("3b. a position's key is unchanged by the tokens that follow it", k_a == k_b)

# 10. Host-resident tables compute the same q_t as device-resident ones.
emb_dev = LearnedNgramEmbeddings(VOCAB, 32, orders=[2, 3], n_tables=2, rows_per_table=256,
                                 width_div=4, sparse=True)
emb_host = LearnedNgramEmbeddings(VOCAB, 32, orders=[2, 3], n_tables=2, rows_per_table=256,
                                  width_div=4, sparse=True, host_tables=True)
emb_host.load_state_dict(emb_dev.state_dict())
a = emb_dev(idx)
b = emb_host(idx)
check("10. host_tables gives the same q_t", torch.allclose(a, b, atol=1e-6),
      f"max diff {(a - b).abs().max():.2e}")

# 11. A toy GPT with learned n-grams: forward, backward, and the parameter bookkeeping.
conf = GPTConfig(vocab_size=VOCAB, block_size=16, n_layer=2, n_head=2, n_embd=32,
                 ngram_orders=3, ngram_tables=2, ngram_rows=64, ngram_width_div=4)
model = GPT(conf)
n_params = model.get_num_params()
conf_off = GPTConfig(vocab_size=VOCAB, block_size=16, n_layer=2, n_head=2, n_embd=32)
base = GPT(conf_off).get_num_params()
pc = model.ngram.param_counts()
check("11. enabling n-grams adds exactly the table + projection parameters",
      n_params - base == pc['table_params'] + pc['projection_params'],
      f"{n_params - base:,} vs {pc['table_params'] + pc['projection_params']:,}")

x = torch.randint(0, VOCAB, (2, 16))
logits, loss = model(x, targets=x)
loss.backward()
table_grads = [t.grad for t in model.ngram.table_weights()]
check("12. gradients reach the tables through the backbone",
      all(g is not None and g.is_sparse for g in table_grads))

opt_main = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), 'cpu')
opt_sparse = model.configure_ngram_optimizer(1e-3, (0.9, 0.95))
in_main = {id(p) for g in opt_main.param_groups for p in g['params']}
in_sparse = {id(p) for g in opt_sparse.param_groups for p in g['params']}
table_ids = {id(p) for p in model.ngram.table_weights()}
check("13. AdamW has no table rows and SparseAdam has nothing else",
      not (in_main & table_ids) and in_sparse == table_ids,
      f"AdamW {len(in_main)} tensors, SparseAdam {len(in_sparse)}")
# torch.optim.SparseAdam has no weight_decay in its param groups at all: it is Adam, and Table 2
# wants these tables to carry no weight decay, so "absent" is the right answer.
check("14. the SparseAdam groups carry no weight decay",
      all(g.get('weight_decay', 0) == 0 for g in opt_sparse.param_groups))

# 15. Checkpoint round trip: the tables are weights, so they have to survive save and reload.
sd = model.state_dict()
ngram_keys_in_sd = [k for k in sd if k.startswith('ngram.')]
check("15a. every table and every projection is in the state dict",
      len(ngram_keys_in_sd) == 2 * 2 + 2 * 2,
      f"{len(ngram_keys_in_sd)} entries: 4 hash tables + 4 projections")
model2 = GPT(GPTConfig(**{**conf.__dict__}))
model2.load_state_dict(sd)
l1, _ = model(x)
l2, _ = model2(x)
check("15b. a reloaded model gives identical logits", torch.equal(l1, l2))

# 16. A checkpoint written before this existed still loads: no ngram keys, no ngram module.
model3 = GPT(GPTConfig(**{k: v for k, v in conf.__dict__.items() if k not in NO_NGRAM}))
check("16. a state dict without n-gram keys builds a model with ngram=None",
      model3.ngram is None and model3.get_num_params() == base)

# 17. The eval-side gate eval_lm.py uses for `standard:no-learned-tables`. Flipping
#     ngram_enabled has to switch q_t out of the forward pass and nothing else -- the parameters
#     still exist, still load, and switching the gate off must equal running the same weights with
#     tables that contain only zeros.
with torch.no_grad():
    logits_on = model(x)[0].clone()
    zeroed = {k: v.clone() for k, v in model.state_dict().items() if k.startswith('ngram.table')}
    for k in zeroed:
        model.state_dict()[k].zero_()
    logits_zero_tables = model(x)[0].clone()
    model.ngram_enabled = False
    logits_gated = model(x)[0].clone()
    model.ngram_enabled = True
    for k, v in zeroed.items():
        model.state_dict()[k].copy_(v)
check("17. ngram_enabled=False is the same forward pass as zero tables",
      torch.equal(logits_gated, logits_zero_tables)
      and not torch.allclose(logits_on, logits_zero_tables, atol=1e-6),
      f"gate vs zero tables max diff {(logits_gated - logits_zero_tables).abs().max():.1e}, "
      f"live tables move logits by {max(0.0, (logits_on - logits_zero_tables).abs().max().item()):.3f}")
check("17b. the gate is not a parameter change", model.get_num_params() == n_params)

print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("LEARNED N-GRAM TESTS OK")
