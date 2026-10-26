"""
Predict how much VRAM a training step needs, without running a training step.

Why measuring on CPU says anything about a GPU: what decides whether a micro-batch fits is mostly
*which tensors autograd decides to keep for the backward pass*. That decision is made by the
autograd graph, not by the device, so the saved_tensors_hooks below count real bytes for the real
model.py at the real 12L/768/T=1024/vocab-50304 shape. pack() adds bytes and unpack() takes them
back, so the number that matters -- the high water mark of simultaneously live saved tensors --
is measured, not assumed, and it is fitted over three micro-batch sizes instead of extrapolated
from one.

Three things a CPU measurement cannot see, each handled as an explicit named term rather than a
hidden fudge factor:

  1. SDPA backend. On CPU torch uses the math path, which materializes (B, heads, T, T) and saves
     it. With AOTriton flash enabled (amd_opt.py sets TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1)
     it saves the output plus one logsumexp per (batch, head, position) instead. The measured math
     term is subtracted and the flash term added.
  2. The lm_head / cross_entropy transient. Autocast runs cross_entropy in fp32, so at the top of
     the step there are three simultaneous (rows, 50304) tensors: the bf16 logits, the fp32 upcast
     whose log_softmax is what backward saves, and the fp32 gradient. Only the middle one is a
     saved tensor, so the other two are added as their own term -- which is also the term
     --chunked-head-rows shrinks, and why that lever is measured with a chunk small enough for the
     CPU probes to actually trigger it.
  3. Allocator fragmentation, driver scratch pools, Inductor scratch. Not modelable from here. It
     is calibrated against the one hardware fact measured on this box (micro-batch 8 -> 12 cost
     +2.5 GiB on the 16 GB card) and carried as a printed margin.

Nothing in this file touches a GPU: no cuda import, no device string. Run it with the GPU hidden:

    NANOAMD=0 HIP_VISIBLE_DEVICES= venv/bin/python tools/vram_model.py
    NANOAMD=0 HIP_VISIBLE_DEVICES= venv/bin/python tools/vram_model.py --json   # also writes JSON
"""
import argparse
import json
import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from model import GPT, GPTConfig, STOCK_ARCH # noqa: E402

# the shape every run on this box actually uses (config/train_fineweb_edu_amd.py)
N_LAYER, N_HEAD, N_EMBD, BLOCK, VOCAB = 12, 12, 768, 1024, 50304
HEAD_DIM = N_EMBD // N_HEAD
TOKENS_PER_ITER = 262144 # 8 micro-batches x 32 accumulations x 1024: the pinned comparison volume
GIB = 2 ** 30

# Measured on this box (AMD_NOTES.md). The 16 GB card clamps to 8 GiB while its DP-3 output drives
# a monitor; the 24 GB card is headless and gets no clamp.
CARDS = {
    'gfx1100': dict(name='RX 7900 XTX', gib=24.0, clamp=None, note='device 0, headless, no clamp applied'),
    'gfx1101': dict(name='RX 7800 XT', gib=16.0, clamp=8.0, note='device 1, clamped to 8 GiB while DP-3 is connected'),
}
MEASURED_KNEE = dict(card='gfx1101', from_bs=8, to_bs=12, delta_gib=2.5,
                     note='mb 8 -> 12 bought 49.5k -> 49.9k tok/s and +2.5 GiB; mb 16 did not fit 8 GiB')

# The only absolute peaks ever measured on this box: gfx1101, stock block, compile=True,
# T=1024, vocab 50304, bf16 (AMD_NOTES.md "Micro-batch size"). Two points fit a line, and the
# third (bs=16, OOM after reaching 11.59 GiB) is a bound the fit has to respect.
HW_ANCHORS = dict(card='gfx1101', points=[(8, 7.18), (12, 9.62)], oom_at=(16, 11.59),
                  note='compile=True, eager cost +1.4 GiB at bs=8 and more at larger micro-batch')

# cross_entropy in fp32 on bf16 logits: bf16 logits + fp32 log_softmax (saved for backward) + fp32 grad
HEAD_BYTES_PER_ELEMENT = 2 + 4 + 4
GPU_CHUNK_ROWS = 8192 # what a real run would pass to --chunked-head-rows; CPU probes use a smaller one


def build(arch_on, chunked_head_rows=0, checkpoint=False):
    cfg = dict(n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD, block_size=BLOCK,
               vocab_size=VOCAB, bias=False, dropout=0.0)
    if not arch_on:
        cfg.update(STOCK_ARCH) # spelled out: the dataclass defaults are now the arch-on block
    return GPT(GPTConfig(**cfg, chunked_head_rows=chunked_head_rows, checkpoint=checkpoint))


def classify(shape, numel, B):
    """Name a saved tensor by what it is, so the total reads as parts and not one number."""
    if numel == B * BLOCK * VOCAB:
        return 'vocab projection (fp32 logits kept by cross_entropy)'
    if len(shape) == 4 and shape[0] == B and shape[1] == N_HEAD and shape[3] == HEAD_DIM:
        return 'attention q/k/v/o, per block'
    if len(shape) == 4 and shape[2] == BLOCK and shape[3] == BLOCK:
        return 'SDPA math-path (B,heads,T,T) -- CPU only'
    if len(shape) == 3 and shape[0] == B and shape[1] == BLOCK and shape[2] == N_EMBD:
        return 'residual stream (block inputs, norm/attn/mlp outputs)'
    if len(shape) == 3 and shape[0] == B and shape[1] == BLOCK:
        return 'MLP inner activation'
    if len(shape) == 2:
        return 'fp32 weights re-saved per matmul (constant in B)'
    return 'other'


def measure(model, B):
    """Forward + backward with counting-only hooks.

    Returns (peak_bytes, categories, n_events, seconds). peak_bytes is the maximum number of
    bytes simultaneously live in the saved-tensor set, which is what an allocator must hold, and
    which is why --activation-checkpoint shows up as a saving even though its recomputed tensors
    are still counted in the per-category totals.
    """
    cats = {}
    live = {'alive': 0, 'peak': 0, 'events': 0}

    def pack(t):
        nbytes = t.numel() * t.element_size()
        live['alive'] += nbytes
        live['peak'] = max(live['peak'], live['alive'])
        live['events'] += 1
        kind = classify(tuple(t.shape), t.numel(), B)
        cats[kind] = cats.get(kind, 0) + nbytes
        return (t, nbytes)

    def unpack(pair):
        t, nbytes = pair
        live['alive'] -= nbytes
        return t

    X = torch.randint(0, VOCAB, (B, BLOCK))
    Y = torch.randint(0, VOCAB, (B, BLOCK))
    model.train()
    for p in model.parameters():
        p.grad = None
    t0 = time.time()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            _, loss = model(X, Y)
            loss.backward()
    peak = live['peak']
    del loss
    model.zero_grad(set_to_none=True)
    return peak, cats, live['events'], time.time() - t0


def fit_linear(points):
    """peak(B) = const + coef*B, solved from the two extreme probes. The third probe is not used
    in the fit; its deviation from the line is reported, because 'scales linearly in B' is a claim
    and the cheap way to keep it honest is to check it."""
    (b1, t1), (b2, t2) = points[0], points[-1]
    coef = (t2 - t1) / (b2 - b1)
    const = t1 - coef * b1
    worst = max(abs((const + coef * b - t) / t) for b, t in points)
    return const, coef, worst


def main():
    ap = argparse.ArgumentParser(description='CPU-measured VRAM model for nanoGPT on this box')
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--json-out', default=os.path.join(REPO, 'tools', 'vram_model.json'),
                    help='where --json writes; stdout stays the human report')
    ap.add_argument('--cards', default='gfx1100,gfx1101')
    ap.add_argument('--probes', default='1,2,4', help='micro-batch sizes to measure')
    ap.add_argument('--chunk-rows', type=int, default=2048,
                    help='chunk size for the chunked-head variant; 2048 so the CPU probes exceed it')
    args = ap.parse_args()
    probes = [int(b) for b in args.probes.split(',')]

    variants = [
        ('stock (layernorm/gelu/no-qk/no-rope)', dict(arch_on=False)),
        ('default (rmsnorm/swiglu/qk/rope)', dict(arch_on=True)),
        (f'default + chunked_head_rows={args.chunk_rows}',
         dict(arch_on=True, chunked_head_rows=args.chunk_rows)),
        ('default + activation_checkpoint', dict(arch_on=True, checkpoint=True)),
    ]

    print(f"model: {N_LAYER} layers / {N_EMBD} wide / {N_HEAD} heads, block {BLOCK}, vocab {VOCAB}")
    print("bf16 autocast, fp32 master weights. Measured on CPU; nothing was run on a GPU.\n")

    out = {'probes': probes, 'variants': {}}
    for label, kw in variants:
        model = build(**kw)
        n_params = model.get_num_params()
        chunk = kw.get('chunked_head_rows', 0)
        points, cats = [], None
        for B in probes:
            peak, c, events, dt = measure(model, B)
            points.append((B, peak))
            if B == probes[-1]:
                cats, n_events, secs = c, events, dt
        const, coef, err = fit_linear(points)
        math_unit = cats.pop('SDPA math-path (B,heads,T,T) -- CPU only', 0) / probes[-1]
        flash_unit = BLOCK * N_HEAD * (HEAD_DIM * 2 + 4) # bf16 O + fp32 logsumexp, per batch element
        head_unit = cats.get('vocab projection (fp32 logits kept by cross_entropy)', 0) / probes[-1]
        # the head is handled by its own term below, so it leaves the linear coefficient
        trunk_unit = coef - math_unit - head_unit + flash_unit
        state = n_params * 4 * 4 # fp32 weights + fp32 grads + AdamW exp + exp_avg_sq
        out['variants'][label] = dict(n_params=n_params, peak_points={str(b): p for b, p in points},
                                      const_bytes=const, trunk_unit_bytes=trunk_unit,
                                      head_unit_bytes=head_unit, math_unit_bytes=math_unit,
                                      flash_unit_bytes=flash_unit, fit_max_rel_err=err,
                                      state_bytes=state, chunk_rows=chunk,
                                      categories=cats)
        print(label)
        print("  live-saved-tensor high water mark: " +
              "  ".join(f"B={b} {p/GIB:6.3f} GiB" for b, p in points) +
              f"   linear-fit miss {100*err:5.2f}%")
        print(f"  {n_events} saved tensors at B={probes[-1]}, {secs:.1f}s to measure")
        for k, v in sorted(cats.items(), key=lambda kv: -kv[1]):
            print(f"    {k:<56s} {v/GIB:7.3f} GiB counted at B={probes[-1]}")
        print(f"  model: constant {const/GIB:.3f} GiB + {trunk_unit/GIB:.4f} GiB per unit of "
              f"micro-batch (CPU coef {coef/GIB:.4f}, minus SDPA math {math_unit/GIB:.4f}, "
              f"plus flash {flash_unit/GIB:.4f}, head separated out)")
        print(f"  state: {state/GIB:.3f} GiB constant ({n_params:,} params x 16 bytes)")
        if chunk:
            print(f"  note: kept for the category breakdown only -- the vocab-projection line disappears "
                  f"against the {label.split('=')[0]} row, but this probe's slope is not used in the table")
        print()
        del model

    print("-- calibration against what this box has actually measured --")
    stock_label = 'stock (layernorm/gelu/no-qk/no-rope)'
    stock = out['variants'][stock_label]
    cpu_slope = (stock['trunk_unit_bytes'] + stock['head_unit_bytes']) / GIB
    (b0, p0), (b1, p1) = HW_ANCHORS['points']
    hw_coef = (p1 - p0) / (b1 - b0)
    hw_intercept = p0 - hw_coef * b0
    ratio = hw_coef / cpu_slope
    print(f"  CPU stock slope {cpu_slope:.3f} GiB per unit of micro-batch")
    print(f"  {HW_ANCHORS['card']} measured {p0} GiB at bs={b0}, {p1} GiB at bs={b1} "
          f"-> hardware slope {hw_coef:.3f} GiB/unit, constant {hw_intercept:.2f} GiB "
          f"({HW_ANCHORS['note']})")
    print(f"  the CPU model is {100*(1-ratio):.0f}% too heavy per micro-batch unit; trunk deltas are "
          f"scaled by {ratio:.2f} to match, absolute levels are taken from the card, not from CPU")
    ob, og = HW_ANCHORS['oom_at']
    print(f"  bound check: bs={ob} reached {og} GiB and OOM'd; this model says "
          f"{hw_intercept + hw_coef*ob:.2f} GiB for the stock block on that card")
    k = MEASURED_KNEE
    print(f"  knee check: mb {k['from_bs']} -> {k['to_bs']} predicted "
          f"+{hw_coef*(k['to_bs']-k['from_bs']):.2f} GiB, measured +{k['delta_gib']:.1f} GiB")
    margin = 0.75
    print(f"  carried margin for fragmentation + driver/Inductor scratch: {margin:.2f} GiB")

    def head_model(bs, chunk_rows):
        rows = bs * BLOCK
        return min(rows, chunk_rows) * VOCAB * HEAD_BYTES_PER_ELEMENT if chunk_rows else rows * VOCAB * HEAD_BYTES_PER_ELEMENT

    def predict(label, bs):
        v = out['variants'][label]
        # the hardware anchors already contain the eager head working set at the same batch sizes,
        # so only the trunk *delta* against the stock block is added on top of them
        delta = (v['trunk_unit_bytes'] - stock['trunk_unit_bytes']) * ratio * bs / GIB
        return hw_intercept + hw_coef * bs + delta + margin

    def predict_head_only(bs, chunk_rows):
        """Analytic: what chunking does to the lm_head / cross_entropy working set, on its own."""
        eager = head_model(bs, 0)
        chunked = head_model(bs, chunk_rows)
        return eager, chunked

    TABLE_SKIP = 'chunked'
    print(f"\n-- predicted peak vs micro-batch, tokens/iter pinned at {TOKENS_PER_ITER:,} --")
    table = {}
    for card_key in args.cards.split(','):
        card = CARDS[card_key]
        cap = card['clamp'] or card['gib']
        print(f"\n{card['name']} ({card_key}): {card['gib']:.0f} GiB"
              + (f", clamped to {cap:.0f} GiB" if card['clamp'] else '') + f" -- {card['note']}")
        for label in out['variants']:
            if TABLE_SKIP in label:
                continue # measured but deliberately not extrapolated; see the head block below
            cells = []
            for bs in (4, 8, 12, 16, 24, 32, 48):
                peak = predict(label, bs)
                accum = max(1, round(TOKENS_PER_ITER / (bs * BLOCK)))
                verdict = 'fits' if peak <= cap else 'no'
                if peak <= cap and cap - peak < 2.0:
                    verdict = 'tight'
                cells.append(f"bs{bs:<3d} {peak:5.2f}G x{accum:<3d} {verdict:>4s}")
                table[f"{card_key}|{label}|bs{bs}"] = round(peak, 3)
            print(f"  {label:<38s} " + ' '.join(cells))

    print("\n-- the --chunked-head-rows lever, analytically and not through the fit --")
    print("   The probe measured it at chunk=2048 with B<=4, where the chunk count and the")
    print("   per-chunk weight re-saves do not scale like a real run's, so its slope is not used.")
    print("   What chunking does do is bound the (rows, 50304) working set, and that part is arithmetic:")
    for bs in (8, 16, 24, 32):
        eager, chunked = predict_head_only(bs, GPU_CHUNK_ROWS)
        nchunks = -(-bs * BLOCK // GPU_CHUNK_ROWS)
        print(f"   bs={bs:<3d} rows={bs*BLOCK:>6d} eager {eager/GIB:6.2f} GiB -> "
              f"{nchunks} chunks of {GPU_CHUNK_ROWS}: {chunked/GIB:5.2f} GiB, "
              f"saves {(eager-chunked)/GIB:5.2f} GiB")
    out['head_gib'] = {f"bs{bs}": [head_model(bs, 0) / GIB, head_model(bs, GPU_CHUNK_ROWS) / GIB]
                       for bs in (8, 16, 24, 32)}

    print("\n-- the same line against block_size, because 'bs=32 used about 7 GB' is a T question --")
    print("   everything in the slope is per token (activations + logits); only the "
          f"{hw_intercept:.2f} GiB constant is not:")
    for T in (1024, 512, 256, 128):
        cells = '  '.join(f"bs{bs:<3d} {hw_intercept + hw_coef * bs * T / BLOCK + margin:5.2f}G"
                          for bs in (8, 16, 32, 64))
        print(f"   block_size {T:<5d} {cells}")
    out['table_gib'] = table
    out['margin_gib'] = margin
    out['hw_coef_gib'] = hw_coef
    out['hw_intercept_gib'] = hw_intercept
    out['cpu_to_hw_slope_ratio'] = ratio

    if args.json:
        path = args.json_out
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote the machine-readable version (including the per-category saved-tensor "
              f"breakdown) to {path}. The human report above went to stdout, so this file is pure JSON.")
    else:
        print("\nPeaks are predictions with a +/-"
              f"{margin:.2f} GiB band on top of a slope borrowed from a 16 GB card, not "
              "measurements. The first micro-batch that crosses the cap is where an OOM lands; "
              "tools/bench_train.py --batch_size N --synthetic reports the real number the moment "
              "a card is free.")


if __name__ == '__main__':
    main()
