"""
Measure this GPU's practical peak throughput (square matmuls) so TFLOP/s
numbers reported by the benchmarks can be read against a real denominator
instead of a datasheet guess.
"""
import time, argparse, torch

def bench(M, N, K, dtype, iters=20):
    a = torch.randn(M, K, device='cuda', dtype=dtype)
    b = torch.randn(K, N, device='cuda', dtype=dtype)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return 2.0 * M * N * K / dt / 1e12

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=8192)
    args = ap.parse_args()
    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name} ({p.gcnArchName})  {p.total_memory/2**30:.1f} GiB  {p.multi_processor_count} compute units")
    print(f"square matmul {args.size}^3:\n")
    for name, dt in [('fp32', torch.float32), ('tf32', torch.float32), ('fp16', torch.float16), ('bf16', torch.bfloat16)]:
        tf32 = (name == 'tf32')
        torch.backends.cuda.matmul.allow_tf32 = tf32
        try:
            t = bench(args.size, args.size, args.size, dt)
            print(f"  {name:5s} : {t:7.2f} TFLOP/s")
        except Exception as e:
            print(f"  {name:5s} : failed ({e})")
        torch.backends.cuda.matmul.allow_tf32 = False
    # shapes that actually occur in GPT-2 124M: (B*T, 768) x (768, 2304/3072) and (B*T,768)x(768,50304)
    print(f"\nGPT-2 124M-shaped GEMMs (bf16, M=B*T):")
    for M in (8192, 12288, 24576):
        for (K, N) in ((768, 2304), (768, 3072), (768, 50304)):
            t = bench(M, N, K, torch.bfloat16)
            print(f"  M={M:6d} K={K:4d} N={N:5d} : {t:7.2f} TFLOP/s")

if __name__ == '__main__':
    main()
