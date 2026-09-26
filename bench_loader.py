"""Benchmark training steps fed by the streaming parquet loader
(parquet_stream.py). Same effective batch (480 seqs of 1024) as the
train config, so tokens/step and tok/s are comparable to a real run.
"""
import os, time
import torch
from contextlib import nullcontext

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT, GPTConfig
from parquet_stream import ParquetStream, plan_parquet_splits

torch.manual_seed(0)
dev = 'cuda'
ptdtype = torch.bfloat16
ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)

# ---- model: GPT-2 small ----
conf = GPTConfig(n_layer=12, n_head=12, n_embd=768, block_size=1024,
                 bias=False, vocab_size=50304, dropout=0.0)
model = GPT(conf).to(dev)
model = torch.compile(model)

# ---- data: stream the parquet shards, tokenize on the fly ----
data_dir = 'data/fineweb-edu'
block_size = 1024
train_files, val_files, _ = plan_parquet_splits(data_dir, 'text', 'gpt2',
                                                train_tokens=500_000_000,
                                                val_tokens=1_000_000)
stream = ParquetStream(train_files, 'text', 'gpt2', token_budget=500_000_000,
                       block_size=block_size, batch_rows=20_000,
                       buffer_tokens=256_000_000, workers=4, label='bench')
stream.start()

def get_batch(batch_size):
    x, y = stream.get_batch(batch_size)
    return (torch.from_numpy(x).pin_memory().to(dev, non_blocking=True),
            torch.from_numpy(y).pin_memory().to(dev, non_blocking=True))

def train_step(fetch, micro, accum, optim):
    optim.zero_grad(set_to_none=True)
    for _ in range(accum):
        X, Y = fetch(micro)
        with ctx:
            loss, _ = model(X, Y)
        (loss / accum).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()

def bench(name, micro, accum, steps=8, warmup=4):
    optim = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9,0.95), weight_decay=0.1)
    for _ in range(warmup):
        train_step(get_batch, micro, accum, optim)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(steps):
        train_step(get_batch, micro, accum, optim)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / steps
    tokens_per_step = micro * accum * block_size
    print(f"{name:34s} {dt*1000:8.1f} ms/step   {tokens_per_step/dt:12,.0f} tok/s")
    return dt

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"effective batch = 480 seqs (491,520 tokens)\n")

bench("parquet stream (mb24 x20, 4 threads)", 24, 20)
