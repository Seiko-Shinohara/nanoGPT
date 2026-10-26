"""
End-to-end nanoGPT training-step benchmark (device 1 only).

Replicates train.py's inner loop faithfully -- memmap batch fetch, autocast,
the full gradient_accumulation_steps loop, grad clip, AdamW step -- and reports
steady-state throughput AND peak VRAM, because on a 16 GB card VRAM is the
binding constraint.

Pin the device first:  HIP_VISIBLE_DEVICES=1
Enable AOTriton flash: TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
"""
import os, sys, time, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import amd_opt                      # must precede torch: sets ROCm env vars + VRAM cap
import numpy as np
import torch

from model import GPTConfig, GPT, STOCK_ARCH
from torch.nn.attention import sdpa_kernel, SDPBackend

BACKEND_MAP = {'default': None, 'flash': SDPBackend.FLASH_ATTENTION,
               'efficient': SDPBackend.EFFICIENT_ATTENTION, 'math': SDPBackend.MATH}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='shakespeare')
    p.add_argument('--batch_size', type=int, default=12)
    p.add_argument('--block_size', type=int, default=1024)
    p.add_argument('--accum', type=int, default=8)
    p.add_argument('--n_layer', type=int, default=12)
    p.add_argument('--n_head', type=int, default=12)
    p.add_argument('--n_embd', type=int, default=768)
    p.add_argument('--vocab_size', type=int, default=50304)
    p.add_argument('--dtype', default='bf16', choices=['bf16', 'fp16', 'fp32'])
    p.add_argument('--compile', default='default', choices=['off', 'default', 'reduce-overhead', 'max-autotune'])
    p.add_argument('--sdpa', default='default', choices=list(BACKEND_MAP))
    p.add_argument('--tunable', action='store_true')
    p.add_argument('--synthetic', action='store_true',
                   help='random token ids generated on the GPU: no memmap, no page faults, '
                        'and any --vocab_size is safe (needed for the lm_head ablation)')
    # the two GPTConfig VRAM levers from model.py; both default to the stock path
    p.add_argument('--chunked-head-rows', type=int, default=0, metavar='ROWS',
                   help='run lm_head + cross_entropy ROWS rows at a time (0 = stock). '
                        'Cuts the forward peak of the vocab projection; costs a few %% time')
    p.add_argument('--activation-checkpoint', action='store_true',
                   help='recompute each transformer block during backward: less VRAM, ~33%% more compute')
    # the four architecture levers from model.py; defaults are the stock GPT-2 block.
    # Measure them here for their throughput/VRAM cost and in train.py for their loss cost:
    # a change that costs 5%% tokens/s has to buy more than that back to be worth running.
    p.add_argument('--norm', default=None, choices=['layernorm', 'rmsnorm'])
    p.add_argument('--ffn', default=None, choices=['gelu', 'swiglu'],
                   help='swiglu uses hidden 8/3*n_embd rounded to 128 (2048 at 768): same '
                        'parameters and same FLOPs as gelu\'s 4x MLP, narrower activation')
    p.add_argument('--qk-norm', action=argparse.BooleanOptionalAction, default=None,
                   help='RMSNorm over head_dim on q and k (--qk-norm / --no-qk-norm)')
    p.add_argument('--rope', action=argparse.BooleanOptionalAction, default=None,
                   help='rotary q/k positions, no learned wpe (--rope / --no-rope)')
    p.add_argument('--rope-base', type=float, default=None)
    p.add_argument('--stock-arch', action='store_true',
                   help='the pre-flag block in one word: layernorm + gelu + no qk-norm + no rope. '
                        'This is what every checkpoint written before these flags was trained with, '
                        'so use it when benchmarking a run you intend to resume')
    # COUNT (The COUNT). Throughput is the paper's headline claim -- a COUNT training step should
    # cost ~93-98% of a standard transformer step -- so the comparison has to run in this harness,
    # against the same block, the same batch shape and the same fused-attention backend. Without
    # --count-table this script is unchanged.
    p.add_argument('--count-table', action='append', default=[], metavar='DIR',
                   help='count table directory (count_native/count_build --out-dir). Repeatable.')
    p.add_argument('--count-domain-table', default='', metavar='DIR',
                   help='domain table mixed into --count-table at --count-lambda')
    p.add_argument('--count-lambda', type=float, default=0.0)
    p.add_argument('--count-j', type=int, default=6, help='suffix orders summed into f_t')
    p.add_argument('--count-k', type=int, default=8, help='candidates per order')
    p.add_argument('--count-orders', type=int, default=0, help='0 = every order the table stores')
    p.add_argument('--count-backoff-depth', type=int, default=-1)
    p.add_argument('--count-workers', type=int, default=8, help='CPU threads inside the lookup')
    p.add_argument('--count-prefetch-depth', type=int, default=2,
                   help='batches of lookups kept in flight. 0 = the lookup serializes in front of '
                        'the GPU, which is the fastest way to see how much COUNT really costs')
    p.add_argument('--count-chunk', type=int, default=1024,
                   help='positions folded into f_t at a time. The transient is '
                        'chunk*J*K*n_embd, so on a VRAM-clamped card this is the lever')
    p.add_argument('--count-amplify', action='store_true',
                   help='sqrt(d) LayerNorm over the combined input (Appendix B.6)')
    p.add_argument('--warmup', type=int, default=6)
    p.add_argument('--iters', type=int, default=10)
    p.add_argument('--tag', default='run')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(1337)
    dtype_name = {'bf16': 'bfloat16', 'fp16': 'float16', 'fp32': 'float32'}[args.dtype]
    amd_opt.configure(dtype=dtype_name, tunable=args.tunable, strict_flash=False)

    dtypes = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}
    ptdtype = dtypes[args.dtype]

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(here, 'data', args.dataset)

    if args.synthetic:
        data = None
        get_batch = lambda: (
            torch.randint(0, args.vocab_size, (args.batch_size, args.block_size),
                          device='cuda', dtype=torch.long),
            torch.randint(0, args.vocab_size, (args.batch_size, args.block_size),
                          device='cuda', dtype=torch.long))
    else:
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')

        def get_batch():
            ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
            x = torch.stack([torch.from_numpy((data[i:i+args.block_size]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((data[i+1:i+1+args.block_size]).astype(np.int64)) for i in ix])
            return x.pin_memory().to('cuda', non_blocking=True), y.pin_memory().to('cuda', non_blocking=True)

    # Architecture: use what was asked for and fall back to GPTConfig's own defaults, so a bare
    # bench_train.py measures the same block a bare train.py run would build. --stock-arch is the
    # one-word way to measure the block the *existing* checkpoints use.
    asked = dict(STOCK_ARCH) if args.stock_arch else {}
    for k in ('norm', 'ffn', 'qk_norm', 'rope', 'rope_base'):
        v = getattr(args, k)
        if v is not None:
            asked[k] = v
    arch = {**{k: GPTConfig.__dataclass_fields__[k].default for k in STOCK_ARCH}, **asked}

    count_dirs = list(args.count_table) + ([args.count_domain_table] if args.count_domain_table else [])
    if args.synthetic and count_dirs:
        raise SystemExit("--synthetic generates random tokens: there is no n-gram structure for a "
                         "count table to match. Drop --synthetic to benchmark COUNT.")

    pipeline = None
    if count_dirs:
        sys.path.insert(0, here)
        from count.data import CountDataPipeline, open_tables, table_weights
        from count.table import set_threads

        set_threads(args.count_workers)
        tables = open_tables(count_dirs)
        weights = table_weights(args.count_lambda, len(tables))
        pipeline = CountDataPipeline(
            tables, weights, [os.path.join(data_dir, 'train.bin')], args.block_size, args.batch_size,
            j=args.count_j, k=args.count_k,
            max_order=(args.count_orders if args.count_orders > 0 else None),
            backoff_depth=args.count_backoff_depth, device='cuda', score_dtype=ptdtype,
            workers=args.count_workers, prefetch_depth=args.count_prefetch_depth, seed=1337)
        for t in tables:
            print(t.describe())

    def get_batch_count():
        b = pipeline.next_batch()
        return b['idx'], b['targets'], b

    model = GPT(GPTConfig(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                          block_size=args.block_size, bias=False, vocab_size=args.vocab_size, dropout=0.0,
                          chunked_head_rows=args.chunked_head_rows, checkpoint=args.activation_checkpoint,
                          count=bool(count_dirs), count_j=args.count_j, count_k=args.count_k,
                          count_chunk=args.count_chunk, count_amplify=args.count_amplify,
                          **arch))
    model.to('cuda')
    n_params = model.get_num_params()
    arch_str = f"{arch['norm']}/{arch['ffn']}{'/qk' if arch['qk_norm'] else ''}{'/rope' if arch['rope'] else ''}"
    optimizer = model.configure_optimizers(1e-1, 6e-4, (0.9, 0.95), 'cuda')

    if args.compile != 'off':
        model = torch.compile(model, mode=None if args.compile == 'default' else args.compile)

    backend = BACKEND_MAP[args.sdpa]
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'fp16'))

    def one_micro(X, Y, C=None):
        ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
        with ctx:
            logits, loss = model(X, Y, count=C)
            loss = loss / args.accum
        return loss

    def full_step(X, Y, C=None):
        losses = []
        for _ in range(args.accum):
            with _maybe_backend(backend):
                loss = one_micro(X, Y, C)
            # the next batch is fetched while the GPU still owns the current one, exactly as
            # train.py does it -- with COUNT that is where the table lookup gets hidden
            Xn, Yn, Cn = get_batch()
            with _maybe_backend(backend):
                scaler.scale(loss).backward()
            X, Y, C = Xn, Yn, Cn
            losses.append(loss)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        return X, Y, losses[-1]

    if pipeline is not None:
        get_batch = get_batch_count
    else:
        _get_batch2 = get_batch

        def get_batch(): # one more slot in the tuple so the step loop has a single shape
            X, Y = _get_batch2()
            return X, Y, None

    X, Y, C = get_batch()
    torch.cuda.reset_peak_memory_stats()

    try:
        t_compile = time.time()
        for _ in range(args.warmup):
            X, Y, _ = full_step(X, Y, C)
        torch.cuda.synchronize()
        t_compile = time.time() - t_compile
        torch.cuda.reset_peak_memory_stats()

        times = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            X, Y, last_loss = full_step(X, Y, C)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    except torch.OutOfMemoryError:
        pa = torch.cuda.max_memory_allocated()/2**30
        pr = torch.cuda.max_memory_reserved()/2**30
        tot = torch.cuda.get_device_properties(0).total_memory/2**30
        print(f"[{args.tag:>26s}] sdpa={args.sdpa:<9s} {args.dtype} compile={args.compile:<14s} "
              f"bs={args.batch_size:<3d} accum={args.accum:<3d} arch={arch_str:<28s} "
              f"| OOM  peak {pa:5.2f}/{pr:5.2f} GiB of {tot:.1f}  <-- does not fit")
        return

    med = sorted(times)[len(times)//2]
    worst = max(times)
    tokens_per_iter = args.accum * args.batch_size * args.block_size
    L, H, Q, T = args.n_layer, args.n_head, args.n_embd//args.n_head, args.block_size
    flops_per_iter = (6*n_params + 12*L*H*Q*T) * T * args.accum * args.batch_size
    tflops = flops_per_iter / med / 1e12
    peak_alloc = torch.cuda.max_memory_allocated()/2**30
    peak_resv = torch.cuda.max_memory_reserved()/2**30
    total_gib = torch.cuda.get_device_properties(0).total_memory/2**30

    print(f"[{args.tag:>26s}] sdpa={args.sdpa:<9s} {args.dtype} compile={args.compile:<14s} "
          f"bs={args.batch_size:<3d} accum={args.accum:<3d} tun={'Y' if args.tunable else 'N'} "
          f"arch={arch_str:<28s} "
          f"| {med*1000:8.2f} ms/iter (max {worst*1000:7.2f}) | {tokens_per_iter/med/1000:8.1f}k tok/s "
          f"| {tflops:6.2f} TFLOP/s | peak {peak_alloc:5.2f}/{peak_resv:5.2f} GiB of {total_gib:.1f}")
    print(f"{'':27s} sanity: final loss {last_loss.item():.4f}  (params {n_params/1e6:.1f}M)")
    if pipeline is not None:
        print(f"{'':27s} COUNT: J={args.count_j} K={args.count_k} orders="
              f"{'all' if args.count_orders == 0 else args.count_orders} "
              f"prefetch={args.count_prefetch_depth} workers={args.count_workers} "
              f"tables={'+'.join(os.path.basename(d) for d in count_dirs)} "
              f"host={sum(t.host_bytes() for t in tables)/1e9:.3f} GB "
              f"amplify={args.count_amplify}")
        pipeline.close()


from contextlib import nullcontext, contextmanager

@contextmanager
def _maybe_backend(backend):
    if backend is None:
        with nullcontext():
            yield
    else:
        with sdpa_kernel([backend]):
            yield


if __name__ == '__main__':
    main()
