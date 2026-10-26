"""Where does the COUNT fold actually spend its time on the GPU?

The paper reports COUNT training at 93-98% of a standard transformer's tokens/second. The lookup is
host work the prefetcher hides (a 1.661 GB table searched over 6,144 positions costs about 2 ms
against 135 ms of matmul), so whatever COUNT costs on the GPU is model.count_embedding: gathering
J*K unembedding rows per position and adding them up. This measures that one function at the shapes
training uses, both implementations, and checks that they agree.

Knobs it sweeps:

  count_dedup    True: factor B[w] out of Eq. 1 and gather each unique token once with its scores
                 summed. False: sum slot by slot, the way Eq. 1 is written.
  count_chunk    groups (dedup) or positions (naive) folded per step -- the transient is
                 chunk * (J*K or 1) * d * 4 bytes, which is what OOMs a VRAM-clamped card at K=32
  J and K        the paper's own J=1 ablation; J*K is the number of rows gathered
  count_norm     'j' divides by a constant, 'nfound' by a per-position tensor

Run:  venv/bin/python tools/bench_count_fold.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import amd_opt
from model import GPT, GPTConfig

B, T = 6, 1024
SHAPES = ((6, 8), (1, 8), (6, 32))
CHUNKS = (256, 1024, 4096)


def timeit(fn, warm=10, reps=30):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def main():
    amd_opt.configure(dtype='bfloat16', tunable=False, strict_flash=False)
    torch.manual_seed(0)
    cfg = GPTConfig(n_layer=12, n_head=12, n_embd=768, block_size=T, vocab_size=50304,
                    bias=False, dropout=0.0, count=True, count_j=6, count_k=8)
    model = GPT(cfg).to('cuda')
    weight = model.lm_head.weight

    print(f"GPT-2 124M on {torch.cuda.get_device_name(0)}, batch {B} x block {T}, "
          f"unembedding {tuple(weight.shape)} {weight.dtype}")
    print(f"J=6 K=8 gathers B*T*J*K = {B * T * 48:,} embedding rows per micro-batch "
          f"= {B * T * 48 * 768 * 2 / 2**20:.0f} MiB of bf16 reads\n")

    outputs = {}
    for dedup in (True, False):
        model.config.count_dedup = dedup
        torch.cuda.reset_peak_memory_stats()
        for (j, k) in SHAPES:
            cands = torch.randint(0, 50304, (B, T, j, k), device='cuda', dtype=torch.long)
            scores = torch.rand(B, T, j, k, device='cuda', dtype=torch.bfloat16) * 0.4
            nfound = torch.randint(1, j + 1, (B, T), device='cuda', dtype=torch.long)
            for chunk in CHUNKS:
                model.config.count_chunk = chunk
                model.config.count_j = j
                try:
                    dt = timeit(lambda: model.count_embedding(cands, scores, nfound))
                    out = model.count_embedding(cands, scores, nfound)
                except torch.OutOfMemoryError:
                    # the naive path materializes chunk*J*K fp32 rows at once; at K=32 that is the
                    # allocation that OOMs an 8 GiB-clamped card. The comparison still matters, so
                    # record it and keep going.
                    print(f"{'dedup' if dedup else 'naive':5s} J={j} K={k} chunk={chunk:5d} "
                          f"OOM (chunk*J*K*d*4 = {chunk * j * k * 768 * 4 / 2**20:.0f} MiB fp32)")
                    outputs.setdefault((dedup, j, k, chunk), model.count_embedding(
                        cands[:, :16], scores[:, :16], nfound[:, :16]))  # small slice, for agreement
                    outputs[(dedup, j, k, chunk)] = outputs[(dedup, j, k, chunk)]
                    torch.cuda.empty_cache()
                    continue
                outputs[(dedup, j, k, chunk)] = out
                peak = torch.cuda.max_memory_allocated() / 2**20
                print(f"{'dedup' if dedup else 'naive':5s} J={j} K={k} chunk={chunk:5d} "
                      f"{dt * 1000:7.2f} ms  ({B * T / dt / 1000:7.1f}k positions/s, "
                      f"peak {peak:7.0f} MiB)")
            torch.cuda.reset_peak_memory_stats()
        model.config.count_chunk = 1024

    print("\nboth paths, same input (Eq. 1 factored vs summed slot by slot):")
    for (j, k) in SHAPES:
        a = outputs[(True, j, k, 1024)].float()
        b = outputs[(False, j, k, 1024)].float()
        n = min(a.size(0), b.size(0))
        a, b = a[:n], b[:n]
        diff = (a - b).abs()
        rel = diff.max().item() / max(1e-9, b.abs().max().item())
        print(f"  J={j} K={k}: max abs diff {diff.max().item():.3e}, "
              f"max |f| {b.abs().max().item():.3e}, relative {rel:.3e}")
        assert rel < 1e-3, "the two implementations of Eq. 1 disagree"

    # how much of a training step is that?
    j, k = 6, 8
    idx = torch.randint(0, 50304, (B, T), device='cuda', dtype=torch.long)
    targets = torch.randint(0, 50304, (B, T), device='cuda', dtype=torch.long)
    count = {"candidates": torch.randint(0, 50304, (B, T, j, k), device='cuda', dtype=torch.long),
             "scores": torch.rand(B, T, j, k, device='cuda', dtype=torch.bfloat16) * 0.4,
             "n_found": torch.randint(1, j + 1, (B, T), device='cuda', dtype=torch.long)}
    for dedup in (True, False):
        model.config.count_dedup, model.config.count_j, model.config.count_k = dedup, 6, 8
        model.config.count_chunk = 1024

        def step():
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, loss = model(idx, targets, count=count)
            loss.backward()
            model.zero_grad(set_to_none=True)

        dt = timeit(step, warm=5, reps=15)
        print(f"forward+backward {'dedup' if dedup else 'naive':5s} {dt * 1000:7.2f} ms/iter "
              f"({B * T / dt / 1000:6.1f}k tok/s)")

    model.config.count = False

    def step_plain():
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, loss = model(idx, targets)
        loss.backward()
        model.zero_grad(set_to_none=True)

    dt = timeit(step_plain, warm=5, reps=15)
    print(f"forward+backward no COUNT {dt * 1000:7.2f} ms/iter ({B * T / dt / 1000:6.1f}k tok/s)")
    print(f"peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, "
          f"reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB")
    return 0


if __name__ == '__main__':
    sys.exit(main())
