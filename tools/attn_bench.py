"""Micro-benchmark SDPA attention backends on the pinned HIP device."""
import os, time, argparse, torch
from torch.nn.functional import scaled_dot_product_attention as sdpa
from torch.nn.attention import sdpa_kernel, SDPBackend

BACKENDS = {
    'FLASH': SDPBackend.FLASH_ATTENTION,
    'EFFICIENT': SDPBackend.EFFICIENT_ATTENTION,
    'MATH': SDPBackend.MATH,
}

def timeit(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--B', type=int, default=12)
    p.add_argument('--T', type=int, default=1024)
    p.add_argument('--nh', type=int, default=12)
    p.add_argument('--dt', default='bf16,bf16')
    args = p.parse_args()
    dtypes = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}
    hs = 64
    print(f"device={torch.cuda.get_device_properties(0).name} B={args.B} nh={args.nh} T={args.T} hs={hs}")
    print("AOTRITON_EXPERIMENTAL=" + os.environ.get('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', '0'))
    for dname in ['fp16', 'bf16', 'fp32']:
        dt = dtypes[dname]
        q = torch.randn(args.B, args.nh, args.T, hs, device='cuda', dtype=dt, requires_grad=True)
        results = {}
        for bname, ctx in BACKENDS.items():
            try:
                with sdpa_kernel(ctx):
                    fwd = timeit(lambda: sdpa(q, q, q, is_causal=True))
                    # fwd+bwd
                    def fb():
                        o = sdpa(q, q, q, is_causal=True)
                        o.backward(torch.ones_like(o))
                    fb = timeit(fb)
                    results[bname] = (fwd * 1000, fb * 1000)
            except Exception as e:
                results[bname] = None
        print(f"\n-- {dname} --")
        for bname, r in results.items():
            if r is None:
                print(f"  {bname:10s} unavailable")
            else:
                print(f"  {bname:10s} fwd {r[0]:8.3f} ms   fwd+bwd {r[1]:8.3f} ms")

if __name__ == '__main__':
    main()
