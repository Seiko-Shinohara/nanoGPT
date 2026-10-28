"""
Quick memory + timing probe for a training config (no eval, no logging).

Reproduces the teacher's exact micro-batch loop (autocast bf16, activation
checkpointing on, fused AdamW) for a few iterations and reports:
  * peak allocated / reserved GPU memory
  * per-iteration wall time (to spot allocator stutters)

Usage:
  HIP_VISIBLE_DEVICES=1 python byte_distill/probe_mem.py \
      --n_layer=10 --n_head=10 --n_embd=640 --block_size=2048 \
      --batch_size=64 --grad_accum=2 --iters=4
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llama import Llama, LlamaConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_layer", type=int, default=10)
    ap.add_argument("--n_head", type=int, default=10)
    ap.add_argument("--n_embd", type=int, default=640)
    ap.add_argument("--block_size", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--vocab_size", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--data", default="data/fineweb-small/train.bin")
    args = ap.parse_args()

    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")
    conf = LlamaConfig(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                       block_size=args.block_size, vocab_size=args.vocab_size,
                       dropout=0.0, qk_norm=False, rope_base=500000.0, tie_weights=False)
    model = Llama(conf).to(device)
    opt = model.configure_optimizers(1e-4, 2e-3, (0.9, 0.95), "cuda")
    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    L = args.block_size
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    def get_batch(bs):
        ix = torch.randint(len(data) - L, (bs,))
        x = torch.stack([torch.from_numpy(data[i:i + L].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + L].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)

    model.train()
    torch.cuda.synchronize()
    t0 = time.time()
    for it in range(args.iters):
        loss = None
        for _ in range(args.grad_accum):
            x, y = get_batch(args.batch_size)
            with ctx:
                _, loss = model(x, y)
            (loss / args.grad_accum).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.time() - t0
        t0 = time.time()
        print(f"iter {it}: {dt * 1000:7.0f} ms  "
              f"peak_alloc={torch.cuda.max_memory_allocated() / 1e9:6.2f} GB  "
              f"peak_reserved={torch.cuda.max_memory_reserved() / 1e9:6.2f} GB")
    print(f"TOTAL peak_alloc={torch.cuda.max_memory_allocated() / 1e9:.2f} GB  "
          f"peak_reserved={torch.cuda.max_memory_reserved() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()