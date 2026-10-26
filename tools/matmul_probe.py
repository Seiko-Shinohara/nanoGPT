"""
Contention-aware matmul probe.

Absolute timings on this GPU are unreliable: device 1 also drives the desktop, and
GPU work gets preempted/evicted (amd-smi reports EVICTED_TIME on these cards). So
instead of measuring fp32 then bf16 in sequence, this probe INTERLEAVES them round
by round. If the machine hiccups, it hits both dtypes in the same round, and the
median across rounds is unaffected while the spread tells us how noisy the machine
was being.

Low memory on purpose (~1 GiB peak) so it cannot disturb the compositor.
"""
import argparse, statistics, torch

def timed(a, b, rounds, inner):
    """Return list of per-round TFLOP/s measured with HIP events."""
    M, K, N = a.shape[0], a.shape[1], b.shape[1]
    flops = 2.0 * M * K * N
    out = []
    for _ in range(rounds):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        # warm the queue, then time a batch of back-to-back GEMMs
        a @ b
        torch.cuda.synchronize()
        s.record()
        for _ in range(inner):
            a @ b
        e.record()
        torch.cuda.synchronize()
        dt = s.elapsed_time(e) / inner / 1000.0
        out.append(flops / dt / 1e12)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=4096)
    ap.add_argument('--rounds', type=int, default=9)
    ap.add_argument('--inner', type=int, default=12)
    args = ap.parse_args()
    S = args.size
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name} ({p.gcnArchName})  {p.total_memory/2**30:.1f} GiB")
    print(f"interleaved {S}x{S}x{S} GEMMs, {args.rounds} rounds x {args.inner} inner ops\n")

    specs = [('fp32', torch.float32, False), ('fp32-tf32flag', torch.float32, True),
             ('fp16', torch.float16, False), ('bf16', torch.bfloat16, False)]
    tensors = {}
    for name, dt, tf32 in specs:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        tensors[name] = (torch.randn(S, S, device='cuda', dtype=dt),
                         torch.randn(S, S, device='cuda', dtype=dt))
    torch.backends.cuda.matmul.allow_tf32 = True

    results = {name: [] for name, _, _ in specs}
    for r in range(args.rounds):
        for name, dt, tf32 in specs:
            torch.backends.cuda.matmul.allow_tf32 = tf32
            a, b = tensors[name]
            results[name].append(timed(a, b, 1, args.inner)[0])

    print(f"{'dtype':14s} {'median':>8s} {'min':>8s} {'max':>8s} {'spread%':>8s}")
    ref = None
    for name, _, _ in specs:
        v = results[name]
        med = statistics.median(v)
        spread = 100 * (max(v) - min(v)) / med
        if name == 'bf16':
            ref = med
        print(f"{name:14s} {med:8.2f} {min(v):8.2f} {max(v):8.2f} {spread:8.1f}")
        print(f"{'':14s} per-round: " + " ".join(f"{x:.1f}" for x in v))
    if ref:
        print(f"\nrelative to bf16 ({ref:.1f} TFLOP/s):")
        for name, _, _ in specs:
            print(f"  {name:14s} {statistics.median(results[name])/ref*100:6.1f}% of bf16")
    print("\nA spread > ~15% means the machine was contended and no absolute number here is trustworthy.")

if __name__ == '__main__':
    main()
