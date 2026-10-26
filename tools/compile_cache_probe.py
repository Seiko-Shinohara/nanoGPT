"""
Does inductor's FX graph cache survive between processes on this box?

nanoGPT calls torch.compile(model) in a fresh process every run, so the only thing
that can make run #2 faster than run #1 is the on-disk inductor cache. This probe
spawns N identical child processes, each compiling the same small module, and prints
per-child compile time. Run #1 should be slow, #2+ fast. If they are all slow, the
cache is either not being written or not being found -- and on this machine /tmp is
tmpfs, so the default cache dir (/tmp/torchinductor_$USER) is deleted at every boot.

    python tools/compile_cache_probe.py --device cpu --runs 3
    python tools/compile_cache_probe.py --device cuda --runs 2     # real thing
    python tools/compile_cache_probe.py --cache-dir /tmp/scratch   # prove a wipe hurts

The module compiled here is a 3-block attention-ish MLP: enough matmul shapes to be
realistic, small enough to compile in seconds. It is NOT nanoGPT, so treat the
absolute seconds as a probe, not as your training startup.
"""
import argparse, os, subprocess, sys, time


class Block(__import__('torch').nn.Module):
    def __init__(self, torch, d=768, h=3072):
        super().__init__()
        self.a = torch.nn.Linear(d, h, bias=False)
        self.b = torch.nn.Linear(h, d, bias=False)
        self.c = torch.nn.Linear(d, d, bias=False)
        self.g = torch.nn.GELU()

    def forward(self, x):
        return self.c(self.g(self.b(self.a(x))))


def child_main(args):
    import torch
    torch.manual_seed(0)
    dev = torch.device(args.device)
    model = torch.nn.Sequential(*[Block(torch) for _ in range(3)]).to(dev)
    model = torch.compile(model)
    x = torch.randn(args.batch, args.tokens, 768, device=dev, requires_grad=True)
    y = torch.randn(args.batch, args.tokens, 768, device=dev)

    t0 = time.time()
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    torch.cuda.synchronize() if dev.type == 'cuda' else None
    first = time.time() - t0

    t0 = time.time()
    for _ in range(args.steps):
        model.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
    torch.cuda.synchronize() if dev.type == 'cuda' else None
    steady = (time.time() - t0) / args.steps

    print(f"CHILD first_iter={first*1000:.0f}ms steady={steady*1000:.1f}ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--steps', type=int, default=5)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--tokens', type=int, default=1024)
    ap.add_argument('--cache-dir', default=None,
                    help='override TORCHINDUCTOR_CACHE_DIR for this probe')
    ap.add_argument('--wipe', action='store_true', help='delete the cache dir first')
    ap.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        child_main(args)
        return

    env = dict(os.environ)
    cache = args.cache_dir or env.get('TORCHINDUCTOR_CACHE_DIR') \
        or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'cache', 'torchinductor')
    env['TORCHINDUCTOR_CACHE_DIR'] = cache
    env.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '4')
    if args.wipe:
        subprocess.run(['rm', '-rf', cache], check=False)
    os.makedirs(cache, exist_ok=True)

    print(f"cache dir: {cache}")
    for i in range(args.runs):
        t0 = time.time()
        res = subprocess.run([sys.executable, os.path.abspath(__file__),
                              '--child', '--device', args.device, '--steps', str(args.steps),
                              '--batch', str(args.batch), '--tokens', str(args.tokens)],
                             env=env, capture_output=True, text=True)
        wall = time.time() - t0
        line = [l for l in res.stdout.splitlines() if l.startswith('CHILD')]
        print(f"run {i+1}: process wall {wall:6.1f}s  {line[0] if line else res.stderr[-200:]}")

    import glob
    n_files = sum(1 for _ in glob.glob(os.path.join(cache, '**', '*'), recursive=True))
    print(f"cache now holds {n_files} entries under {cache}")


if __name__ == '__main__':
    main()
