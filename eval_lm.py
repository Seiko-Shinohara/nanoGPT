"""Model-side evaluation: the paper's LM loss and task loss, measured on a checkpoint.

eval_count.py scores the *tables* -- coverage, candidate quality, S(w_true | h) -- with no weights in
the loop. This file scores the *model*: it loads a checkpoint, runs it over held-out text, and
reports the two numbers the paper's headline tables are made of.

LM loss (Table 1, Sec. 4)
    mean next-token cross-entropy, in **nats per token**, over held-out text. nanoGPT's own val loss
    is the same quantity, so a COUNT-free arm here is directly comparable to a train.py run.

Task loss (Table 1, Sec. 4; the BPB convention used for downstream evaluation)
    bits per byte: total cross-entropy in bits divided by the number of UTF-8 bytes the scored token
    span decodes to. BPB is tokenizer-robust in a way per-token loss is not -- it does not care how
    the text was chopped -- which is why the paper uses it to compare across models and vocabularies.
    It is exact here: gpt2's byte-level BPE round-trips, so `decode(tokens)` reproduces the source
    span byte for byte, and the byte count is a constant per corpus slice shared by every arm.

What varies between arms, and why each one is in the paper

    standard                    no tables at all. The baseline every delta is measured against.
    count:lambda=...            Sec. 5.1's inference-time domain mixing: S_lambda = (1-lambda) S_general
                                + lambda S_domain, on a model that was trained with COUNT. Never
                                renormalized -- the tables' own mass is the point.
    count:K=.. J=..             Appendix B.4's inference-time ablation of how many candidates and how
                                many history orders get folded into f_t. Requires an explicit override
                                (model.attach_count_tables(..., allow_jk_override=True)): a checkpoint
                                fit at J=6 K=8 does not have to behave at K=32, and the honest report
                                says which shape it was trained at.
    count:lam..:ord..           Sec. 5.2's shorter-history variant (--max-orders) and its table-shrink
                                variants, which are just other table directories. A cap on the order is
                                a cap on J -- one candidate set per order the lookup may read -- so the
                                arm is labelled with the J its batch really has and goes through the same
                                override as the B.4 sweep.
    additive:lambda / multiplicative:alpha   Appendix B.1's late fusion, Equations 5-7: same tables,
                                same K, applied to the output distribution instead of the input
                                embedding, over the longest matching history only (J=1).

Everything is fp32 at the head (autocast may run the blocks in bf16) and the vocab head is evaluated
in row chunks, so the arms differ only in what the model is fed and how its logits are post-processed.
Results are printed as per-corpus deltas against the baseline arm and written to a JSON ledger for the
later scaling fits.

Run:
    venv/bin/python eval_lm.py --ckpt out_count_smoke/ckpt.pt --device cuda:1 \
        --count-table count_tables/general150m --count-domain-table count_tables/math_s47 \
        --lambdas 0,0.25,0.5,1 --k-sweep 8,32 --j-sweep 1,6 \
        --fusion additive:0.1,multiplicative:0.1
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# amd_opt pins this process to the box's allowed GPU (it sets HIP_VISIBLE_DEVICES) and, like in
# train.py, has to be imported before torch so the pinning lands before anything touches CUDA.
# Without it, 'cuda' silently means whichever card torch enumerates first.
import amd_opt

import torch

from model import GPT, GPTConfig, STOCK_ARCH, NO_COUNT, NO_NGRAM


DEFAULT_CORPORA = [
    ("general", "data/fineweb-edu/val.bin"),
    ("math", "data/fineweb-edu/domain_math_val.bin"),
    ("law", "data/fineweb-edu/domain_law_val.bin"),
    ("biomedical", "data/fineweb-edu/domain_biomedical_val.bin"),
]


def parse_args():
    ap = argparse.ArgumentParser(description="COUNT model-side eval: LM loss (nats/token) and BPB")
    ap.add_argument("--ckpt", default="out_count_smoke/ckpt.pt")
    # 'cuda', not 'cuda:1': importing amd_opt below pins HIP_VISIBLE_DEVICES to the box's allowed
    # card (index 1 on this machine), which renumbers it to index 0 in-process. Asking for cuda:1
    # after that is an invalid-ordinal crash; asking for plain 'cuda' without the pinning is what put
    # this script on the other GPU and OOMed.
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"],
                    help="autocast dtype for the blocks; the head and every loss are fp32 regardless")
    ap.add_argument("--corpus", action="append", default=[],
                    help="name=path, repeatable. Overrides --corpus-default when given.")
    ap.add_argument("--eval-tokens", type=int, default=262_144,
                    help="tokens scored per corpus, per arm. Same slice for every arm.")
    ap.add_argument("--batch-windows", type=int, default=8,
                    help="windows per forward pass; a window is block_size tokens plus its targets")
    ap.add_argument("--row-chunk", type=int, default=1024,
                    help="rows through the vocab head at a time (1024 x 50304 fp32 = 206 MB)")
    ap.add_argument("--count-table", action="append", default=[],
                    help="general-side table directory, repeatable")
    ap.add_argument("--count-domain-table", action="append", default=[],
                    help="domain-side table directory, mixed in at --lambdas")
    ap.add_argument("--lambdas", default="", help="comma separated mixing weights, e.g. 0,0.25,0.5,1")
    ap.add_argument("--k-sweep", default="", help="comma separated K values (inference-time ablation)")
    ap.add_argument("--j-sweep", default="", help="comma separated J values (inference-time ablation)")
    ap.add_argument("--max-orders", default="", help="comma separated max_order caps (0 = the table's "
                        "own); a cap also caps J to one candidate set per order it allows")
    ap.add_argument("--backoff-depth", type=int, default=-1)
    ap.add_argument("--fusion", default="",
                    help="late-fusion arms: additive:0.1,multiplicative:0.1 (Eq. 5 and Eq. 7)")
    ap.add_argument("--no-standard", action="store_true", help="skip the table-free baseline arm")
    ap.add_argument("--out", default="", help="JSON ledger path (default eval_results/lm_<ckpt>_<time>.json)")
    ap.add_argument("--seed", type=int, default=1337)
    return ap.parse_args()


def load_checkpoint(path, device):
    """Rebuild the exact model the checkpoint describes, COUNT flags included."""
    ckpt = torch.load(path, map_location=device)
    model_args = ckpt['model_args']
    conf = GPTConfig(**{**STOCK_ARCH, **NO_COUNT, **NO_NGRAM, **model_args})
    model = GPT(conf)
    state = ckpt['model']
    for k, v in list(state.items()):
        if k.startswith('_orig_mod.'):
            state[k[len('_orig_mod.'):]] = state.pop(k)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    # A checkpoint trained with ngram_host_tables=True has its tables on the host by configuration;
    # model.to(device) just moved them onto the GPU while forward() still gathers on the CPU.
    model.rehost_ngram_tables()
    trained_count = {k: model_args.get(k, NO_COUNT[k]) for k in
                     ('count', 'count_j', 'count_k', 'count_norm', 'count_amplify')}
    return model, conf, trained_count, ckpt


class Corpus:
    """Held-out tokens, cut into block-size windows, with the byte count BPB divides by."""

    def __init__(self, name, path, block_size, max_tokens, encoding):
        data = np.memmap(path, dtype=np.uint16, mode='r')
        n = min(int(max_tokens) + 1, len(data))
        self.tokens = np.asarray(data[:n], dtype=np.int64)
        self.name = name
        self.path = path
        self.block_size = int(block_size)
        # A window is (block_size inputs, block_size targets) with a one-token offset, so the last
        # full window starts at len - block_size - 1. No wrap-around: nanoGPT's torch.roll would feed
        # the corpus's first token as the target of its last, which is a free prediction the paper's
        # metric does not grant.
        self.starts = list(range(0, len(self.tokens) - self.block_size, self.block_size))
        self.n_positions = len(self.starts) * self.block_size
        self.n_tokens = len(self.tokens) - 1
        text = encoding.decode(self.tokens.tolist())
        self.n_bytes = len(text.encode('utf-8'))

    def window(self, start):
        """A window given its *offset* into the corpus, not its index in the window list."""
        b = self.block_size
        idx = self.tokens[start:start + b]
        tgt = self.tokens[start + 1:start + 1 + b]
        return idx, tgt

    def describe(self):
        return (f"{self.name}: {self.n_positions:,} scored tokens, {self.n_bytes:,} bytes, "
                f"{self.n_bytes / max(1, self.n_positions):.3f} bytes/token, "
                f"{len(self.starts)} windows of {self.block_size}")


def build_arms(args, trained_jk, has_ngram=False):
    """Every arm as a small dict. Kept declarative so the ledger records exactly what was run."""
    arms = []
    if not args.no_standard:
        arms.append(dict(name="standard", kind="none"))
        if has_ngram:
            # For an Eq. 3 checkpoint 'standard' is its trained path *with* q_t, because the tables
            # are weights, not a lookup. The ablation that withdraws q_t is the arm that answers
            # "what did the learned tables buy", and it is the exact counterpart of withdrawing f_t
            # from a COUNT checkpoint.
            arms.append(dict(name="standard:no-learned-tables", kind="none", ngram_off=True))

    ks = [int(v) for v in args.k_sweep.split(',') if v.strip()] or [trained_jk[1]]
    js = [int(v) for v in args.j_sweep.split(',') if v.strip()] or [trained_jk[0]]
    lams = [float(v) for v in args.lambdas.split(',') if v.strip()] or [0.0]
    orders = [int(v) for v in args.max_orders.split(',') if v.strip()] or [0]

    tables = list(args.count_table)
    domains = list(args.count_domain_table)
    multi_table = bool(tables and domains)

    for lam in lams:
        for j in js:
            for k in ks:
                for mo in orders:
                    dirs = tables + domains
                    weights = [1.0 - lam, lam] if multi_table else [1.0]
                    if not dirs:
                        continue
                    # A max_order cap (Sec. 5.2) shortens the history ladder: the lookup returns one
                    # candidate set per order it is allowed to read, so the batch ends up with
                    # J = min(J, max_order) sets. That is an override in exactly the sense
                    # Appendix B.4's J sweep is, and the model's guard has to be configured with the J
                    # the batch will actually have, not the J that was asked for.
                    jj = j if not mo else min(j, mo)
                    label = f"lam{lam:g}"
                    if jj != trained_jk[0]:
                        label += f":J{jj}"
                    if k != trained_jk[1]:
                        label += f":K{k}"
                    if mo:
                        label += f":ord{mo}"
                    arms.append(dict(name=f"count:{label}", kind="count", tables=dirs,
                                     weights=weights, j=jj, k=k,
                                     max_order=(mo or None),
                                     override=(jj != trained_jk[0] or k != trained_jk[1])))

    fusion_tables = list(args.count_table) or list(args.count_domain_table)
    for spec in [s for s in args.fusion.split(',') if s.strip()]:
        if not fusion_tables:
            print(f"skipping late-fusion arm {spec}: no table directory given "
                  "(--count-table or --count-domain-table)")
            continue
        mode, _, weight = spec.partition(':')
        arms.append(dict(name=f"fusion:{spec}", kind="fusion", mode=mode,
                         weight=float(weight or 0.1), tables=fusion_tables,
                         weights=[1.0], j=1, k=8, max_order=None, override=True))
    return arms


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if torch.cuda.is_available():
        # Same setup train.py gets: one visible (pinned) device, AOTriton flash attention, and the
        # amdgpu VRAM cap that keeps a run from pushing the GPU holding the desktop into swap.
        amd_opt.configure(dtype=args.dtype, tunable=False)
    device = torch.device(args.device)
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    if device_type == 'cuda' and torch.cuda.device_count() > 1:
        raise SystemExit(
            f"{torch.cuda.device_count()} GPUs are visible, so '{args.device}' is ambiguous and this "
            "box has a card that must not be used. Set HIP_VISIBLE_DEVICES (or NANOAMD_DEVICE) to the "
            "allowed one and rerun.")
    if device_type == 'cuda':
        print(f"[eval_lm] running on: {torch.cuda.get_device_name(device.index or 0)}")
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = torch.autocast(device_type=device_type, dtype=ptdtype) if device_type == 'cuda' else None

    import tiktoken
    encoding = tiktoken.get_encoding("gpt2")

    model, conf, trained_count, ckpt = load_checkpoint(args.ckpt, device)
    trained_jk = (model.config.count_j, model.config.count_k)
    # An Eq. 3 checkpoint carries its tables as weights, so what it was trained with is in model_args.
    # Worth printing: the arms below then read as "this baseline, plus COUNT's tables on top of it",
    # which is a comparison the paper does not run but a reader of this repo will want to see.
    ngram = f", trained with learned n-grams orders 2..{conf.ngram_orders} " \
            f"(R={conf.ngram_tables}, {conf.ngram_rows} rows/table, width {conf.n_embd // conf.ngram_width_div})" \
            if conf.ngram_orders >= 2 else ""
    print(f"checkpoint {args.ckpt}: iter {ckpt.get('iter_num', '?')}, "
          f"{sum(p.numel() for p in model.parameters()):,} params, "
          f"block_size {conf.block_size}, vocab {conf.vocab_size}, "
          f"trained with COUNT={trained_count['count']} "
          f"(J={trained_count['count_j']} K={trained_count['count_k']} "
          f"amplify={trained_count['count_amplify']}){ngram}")
    if conf.ngram_orders >= 2:
        print("note: for a checkpoint with learned n-gram tables, the 'standard' arm keeps q_t live "
              "(the tables are weights); 'standard:no-learned-tables' is the ablation that withdraws it")

    corpora_specs = args.corpus or [f"{n}={p}" for n, p in DEFAULT_CORPORA]
    corpora = []
    for spec in corpora_specs:
        name, _, path = spec.partition('=')
        if not path:
            name, path = "corpus", name
        corpora.append(Corpus(name, path, conf.block_size, args.eval_tokens, encoding))
    for c in corpora:
        print(f"corpus {c.describe()}")

    # Two runs of one checkpoint against different tables are different experiments, and the shrink
    # ladder runs exactly that: without the tables in the ledger name the fourth rung silently
    # overwrites the first three.
    explicit_tables = [d for d in list(args.count_table) + list(args.count_domain_table) if d]
    table_tag = ("_" + "_".join(os.path.basename(d.rstrip('/')) for d in explicit_tables)
                 if explicit_tables else "")

    if not args.count_table and not args.count_domain_table and trained_count['count']:
        recorded = ckpt.get('config', {}).get('count_table', '')
        if recorded:
            args.count_table = [recorded]
            print(f"using the table the checkpoint was trained with: {recorded}")

    arms = build_arms(args, trained_jk, has_ngram=conf.ngram_orders >= 2)
    if not arms:
        print("no arms: pass --lambdas/--fusion/--count-table, or drop --no-standard")
        return 2

    from count.table import CountTable, set_threads
    from count.generate import count_batch
    from count.late_fusion import fuse_logits

    set_threads(8)
    table_cache = {}

    def tables_for(dirs):
        out = []
        for d in dirs:
            if d not in table_cache:
                table_cache[d] = CountTable(d)
                print(f"table {table_cache[d].describe()}")
            out.append(table_cache[d])
        return out

    # One pass per (arm, corpus). Reopening the tables per corpus would re-do the mmap for nothing.
    results = []
    for arm in arms:
        entry = {"arm": arm['name'], "kind": arm['kind'], "corpora": {}}
        tables, weights = [], [1.0]
        if arm['kind'] in ("count", "fusion"):
            tables = tables_for(arm['tables'])
            weights = arm['weights']
            model.attach_count_tables(tables, weights=weights, j=arm['j'], k=arm['k'],
                                      max_order=arm.get('max_order'),
                                      backoff_depth=args.backoff_depth, score_dtype=torch.float32,
                                      allow_jk_override=arm.get('override', False))
        use_count = arm['kind'] == 'count'
        model.ngram_enabled = not arm.get('ngram_off', False)
        # 'standard' and every late-fusion arm score the model's own distribution; a COUNT checkpoint
        # keeps its amplification LayerNorm either way (that is Figure 17's configuration), only f_t
        # is withdrawn.
        model.config.count = use_count

        for c in corpora:
            nll = 0.0
            n_pos = 0
            coverage = 0
            cand_pos = 0.0
            hits = 0
            mass = 0.0
            t0 = time.time()
            b = args.batch_windows
            starts = c.starts
            for w0 in range(0, len(starts), b):
                idxs, tgts = [], []
                for off in starts[w0:w0 + b]:
                    idx, tgt = c.window(off)
                    idxs.append(idx)
                    tgts.append(tgt)
                idx = torch.from_numpy(np.stack(idxs)).to(device)
                tgt = torch.from_numpy(np.stack(tgts)).to(device)
                tg_flat = tgt.reshape(-1)

                count = None
                cand_flat = score_flat = None
                if tables:
                    # Late fusion reads only h_{t,1}: J=1 (arm['j']); COUNT arms read their J orders.
                    res = model._count_lookup(idx)
                    cand, sc = res['candidates'], res['scores']
                    coverage += int((res['n_found'] > 0).sum().item())
                    cand_pos += float((cand >= 0).sum().item())
                    # Diagnostics are per *position*: a COUNT position has J candidate sets, a
                    # late-fusion position exactly one (its longest match).
                    mass += float(sc.masked_fill(cand < 0, 0.0).sum().item())
                    match = (cand >= 0) & (cand == tgt.view(tgt.shape[0], tgt.shape[1], 1, 1))
                    hits += int(match.any(dim=(2, 3)).sum().item())
                    if use_count:
                        count = {'candidates': cand, 'scores': sc, 'n_found': res['n_found']}
                    else:
                        assert cand.size(2) == 1, "late fusion looks up the longest order only (J=1)"
                        kk = cand.size(3)
                        cand_flat = cand[:, :, 0, :].reshape(-1, kk)
                        score_flat = sc[:, :, 0, :].reshape(-1, kk).float()

                with ctx if ctx is not None else _null():
                    gen = model.logits_rows(idx, count=count, row_chunk=args.row_chunk)
                    for s, e, logp in gen:
                        if arm['kind'] == 'fusion':
                            logp = fuse_logits(logp, cand_flat[s:e], score_flat[s:e],
                                               arm['mode'], arm['weight'])
                        nll += float(-logp.gather(1, tg_flat[s:e, None]).sum().item())
                        n_pos += e - s

            secs = time.time() - t0
            nats = nll / max(1, n_pos)
            entry["corpora"][c.name] = {
                "nll_nats_per_token": nats,
                "bpb": (nll / math.log(2)) / c.n_bytes,
                "positions": n_pos,
                "bytes": c.n_bytes,
                "coverage": coverage / max(1, n_pos),
                "candidates_per_pos": cand_pos / max(1, n_pos),
                "true_token_hit": hits / max(1, n_pos),
                "mean_candidate_mass": mass / max(1, n_pos),
                "seconds": secs,
                "tokens_per_second": n_pos / max(1e-9, secs),
            }
            m = entry["corpora"][c.name]
            diag = (f"  cov {m['coverage']:.3f} cand/pos {m['candidates_per_pos']:.2f} "
                    f"hit {m['true_token_hit']:.3f} mass {m['mean_candidate_mass']:.3f}"
                    if tables else "")
            print(f"  [{arm['name']}] {c.name}: {nats:.4f} nats/token  BPB {m['bpb']:.4f}  "
                  f"({n_pos / secs:,.0f} tok/s){diag}")
        results.append(entry)

    baseline = next((e for e in results if e['kind'] == 'none'), results[0])
    print("\n== LM loss (nats/token), delta against "
          f"'{baseline['arm']}' ==")
    names = [c.name for c in corpora]
    header = f"{'arm':<28}" + "".join(f"{n:>13}" for n in names) + f"{'mean':>13}"
    print(header)
    for e in results:
        vals = [e['corpora'][n]['nll_nats_per_token'] for n in names]
        deltas = [v - baseline['corpora'][n]['nll_nats_per_token'] for v, n in zip(vals, names)]
        row = f"{e['arm']:<28}"
        for v, d in zip(vals, deltas):
            row += f"{v:>7.4f}{d:>+6.3f}" if e is not baseline else f"{v:>13.4f}"
        mean = sum(vals) / len(vals)
        bmean = sum(baseline['corpora'][n]['nll_nats_per_token'] for n in names) / len(names)
        row += f"{mean:>7.4f}{mean - bmean:>+6.3f}" if e is not baseline else f"{mean:>13.4f}"
        print(row)

    print(f"\n== Task loss (bits per byte), delta against '{baseline['arm']}' ==")
    print(header)
    for e in results:
        vals = [e['corpora'][n]['bpb'] for n in names]
        deltas = [v - baseline['corpora'][n]['bpb'] for v, n in zip(vals, names)]
        row = f"{e['arm']:<28}"
        for v, d in zip(vals, deltas):
            row += f"{v:>7.4f}{d:>+6.3f}" if e is not baseline else f"{v:>13.4f}"
        mean = sum(vals) / len(vals)
        bmean = sum(baseline['corpora'][n]['bpb'] for n in names) / len(names)
        row += f"{mean:>7.4f}{mean - bmean:>+6.3f}" if e is not baseline else f"{mean:>13.4f}"
        print(row)

    out = args.out or os.path.join(
        "eval_results",
        f"lm_{os.path.basename(os.path.dirname(args.ckpt))}{table_tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ledger = {
        "checkpoint": args.ckpt,
        "iter_num": ckpt.get('iter_num'),
        "params": sum(p.numel() for p in model.parameters()),
        "trained_count": trained_count,
        "dtype": args.dtype,
        "device": str(device),
        "block_size": conf.block_size,
        "vocab_size": conf.vocab_size,
        "eval_tokens_per_corpus": args.eval_tokens,
        "corpora": {c.name: {"path": c.path, "positions": c.n_positions, "bytes": c.n_bytes}
                    for c in corpora},
        "arms": [{**{k: v for k, v in a.items()}, "results": e["corpora"]}
                 for a, e in zip(arms, results)],
        "tables": {d: {"total_rows": int(t.total_rows()), "host_bytes": int(t.host_bytes()),
                       "corpus_tokens": int(t.corpus_tokens),
                       "max_order": int(t.max_order), "k_store": int(t.k_store),
                       "vocab_size": int(t.vocab_size)}
                   for d, t in table_cache.items()},
    }
    with open(out, 'w') as f:
        json.dump(ledger, f, indent=2)
    print(f"\nwrote {out}")

    for t in table_cache.values():
        t.close()
    return 0


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    sys.exit(main())
