"""
CPU-only self-check for the three things in this repo that are easy to get wrong and hard to
notice: the VRAM-cap decision in amd_opt.py, the chunked-head / activation-checkpoint path in
model.py, and the four architecture flags (norm / ffn / qk_norm / rope).

    venv/bin/python tools/selfcheck.py        # no GPU needed, no compile, a few seconds

Why it exists: the safety cap decides what a run is allowed to allocate, and it branches on
information that is sometimes missing (no amd-smi, no readable sysfs, a display live on the
*other* card). The last time one of those branches was wrong, it was wrong silently. Every
branch is checked here against a fake sysfs tree and a fake torch, so this runs on a machine
with no AMD GPU at all -- including before you trust amd_opt on a card you have not measured.

The model half proves the chunked head and activation checkpointing compute the same loss and
the same gradients as the stock forward, on a real (tiny) GPT, with a chunk size that does not
divide the row count evenly.

The architecture half proves the algebra each flag claims -- SwiGLU really is parameter-matched
to the 4x GELU MLP, RoPE really is an isometry that depends on relative position only, QK-norm
really bounds the attention logits -- and that a typo or a bad default type cannot reach a
training run. It proves none of them improves loss; only GPU runs can do that.
"""
import ast
import copy
import gc
import math
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import types

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch

import amd_opt
from model import GPT, GPTConfig, LayerNorm, MLP, RMSNorm, CausalSelfAttention, make_norm, STOCK_ARCH

fails = []


def check(name, got, want):
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'}  {name:54s} got {got!r}")
    if not ok:
        fails.append(name)


# ---------------------------------------------------------------- amd_opt._vram_cap
# Pure decision: (bdf, connector, displays, requested, ceiling, override) -> (GiB, reason)
# 0.0 GiB means "no cap"; requested=None means the user set nothing. configure() only ever
# passes a connector when it resolved this GPU's BDF, so a case with a connector and a None
# displays dict would be an impossible input and is not tested.
CEIL, DP, NONE = 8.0, {'0000:06:00.0': 'DP-3'}, {}

cases = [
    ("display attached, nothing requested -> clamp",
     '0000:06:00.0', 'DP-3', DP, None, CEIL, False, (8.0, 'display')),
    ("display attached, smaller request honoured",
     '0000:06:00.0', 'DP-3', DP, 4.0, CEIL, False, (4.0, 'display')),
    ("display attached, larger request refused",
     '0000:06:00.0', 'DP-3', DP, 12.0, CEIL, False, (8.0, 'display_clamped')),
    ("display attached + ALLOW_DISPLAY_OVERRIDE obeyed",
     '0000:06:00.0', 'DP-3', DP, 12.0, CEIL, True, (12.0, 'display_override')),
    ("display attached, MAX_VRAM_GIB=0 refused (uncap needs OVERRIDE)",
     '0000:06:00.0', 'DP-3', DP, 0.0, CEIL, False, (8.0, 'display_clamped')),
    ("display attached, MAX_VRAM_GIB=0 + OVERRIDE obeyed",
     '0000:06:00.0', 'DP-3', DP, 0.0, CEIL, True, (0.0, 'display_override')),
    ("headless, nothing requested -> no cap",
     '0000:03:00.0', '', NONE, None, CEIL, False, (0.0, 'headless')),
    ("headless + explicit request -> that cap",
     '0000:03:00.0', '', NONE, 6.0, CEIL, False, (6.0, 'headless')),
    ("headless + MAX_VRAM_GIB=0 -> uncapped",
     '0000:03:00.0', '', NONE, 0.0, CEIL, False, (0.0, 'headless')),
    ("headless while a display is live on the other card",
     '0000:03:00.0', '', {'0000:06:00.0': 'DP-3'}, None, CEIL, False, (0.0, 'headless')),
    ("no amd-smi (bdf None) while a display is live elsewhere",
     None, None, DP, None, CEIL, False, (8.0, 'unknown')),
    ("sysfs unreadable (displays is None) -> assume the worst",
     '0000:06:00.0', None, None, None, CEIL, False, (8.0, 'unknown')),
]
for name, bdf, conn, displays, req, ceil, ovr, want in cases:
    check(name, amd_opt._vram_cap(bdf, conn, displays, req, ceil, ovr), want)


# ---------------------------------------------------------------- fake /sys/class/drm
def fake_drm(connectors, bare_cards=(), extra_dirs=()):
    """Build a directory that looks like /sys/class/drm to _connected_display_outputs()."""
    root = tempfile.mkdtemp()
    for d in extra_dirs:
        os.makedirs(os.path.join(root, d))
    for card in bare_cards:
        os.makedirs(os.path.join(root, card, 'device'))
    for card, connector, status in connectors:
        os.makedirs(os.path.join(root, f'{card}-{connector}'))
        os.makedirs(os.path.join(root, card, 'device'))
        with open(os.path.join(root, f'{card}-{connector}', 'status'), 'w') as f:
            f.write(status + '\n')
    return root


# None vs {} is the distinction that matters: None means "cannot tell, cap anyway",
# {} means "DRM cards visible and none of them is drawing anything, do not cap".
d = fake_drm([], extra_dirs=('renderD128', 'controlD64'))
check("drm dir with only renderD* -> None (unknown)", amd_opt._connected_display_outputs(d), None)
check("unreadable drm dir -> None (unknown)", amd_opt._connected_display_outputs(d + '/absent'), None)

d = fake_drm([], bare_cards=('card0', 'card1'))
check("bare card dirs, no connectors -> {} (headless)", amd_opt._connected_display_outputs(d), {})

d = fake_drm([('card0', 'DP-3', 'connected'), ('card1', 'HDMI-A-1', 'disconnected'),
              ('card2', 'Writeback-1', 'connected')])
check("connected connector only, disconnected+Writeback skipped",
      amd_opt._connected_display_outputs(d), {'device': 'DP-3'})

# The same question against the real sysfs, so the fake tree is not the only thing being
# tested. Skipped where there is no sysfs or no amd-smi to resolve a BDF with.
real = amd_opt._connected_display_outputs()
bdf1 = amd_opt._gpu_identity(amd_opt._visible_physical_index())[0]
if real is None or bdf1 is None:
    print(f"skip  real sysfs check (sysfs: {real}, amd-smi BDF: {bdf1})")
else:
    print(f"info  real sysfs: {real}; this process's GPU {bdf1} -> "
          f"{real.get(bdf1, 'no display attached')}")
    check("real /sys/class/drm gives an answer here, not None", isinstance(real, dict), True)

# ---------------------------------------------------------------- configure() opt-outs
class FakeProps:
    name = 'NVIDIA A100-SXM4-80GB'
    gcnArchName = 'NVIDIA A100-SXM4-80GB' # ROCm-only attribute; absent on NVIDIA builds
    total_memory = 80 * 2**30


fake_torch = types.SimpleNamespace(
    version=types.SimpleNamespace(hip=None),
    cuda=types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_properties=lambda i: FakeProps(),
        mem_get_info=lambda: (60 * 2**30, 80 * 2**30),
        set_per_process_memory_fraction=lambda frac: fails.append(
            'configure() clamped an NVIDIA card with set_per_process_memory_fraction'),
    ),
)


def configure_with_fake_torch(**kw):
    """configure() imports torch inside itself, so sys.modules is the injection point."""
    saved = sys.modules['torch']
    sys.modules['torch'] = fake_torch
    try:
        return amd_opt.configure(**kw)
    finally:
        sys.modules['torch'] = saved


print("\n-- an NVIDIA box must not inherit the amdgpu rules --")
saved_enable = amd_opt._ENABLE
amd_opt._ENABLE = True # emulate the default env so the ROCm check, not NANOAMD, is what stops it
check("non-ROCm build: configure() returns 'cuda'", configure_with_fake_torch(dtype='bfloat16'), 'cuda')

print("-- NANOAMD=0 has to gate the cap and the SDPA lockdown, not just the env vars --")
amd_opt._ENABLE = False
check("NANOAMD=0: configure() opts out", configure_with_fake_torch(dtype='bfloat16'), 'cuda')
amd_opt._ENABLE = saved_enable

# ---------------------------------------------------------------- model.py head path
print("\n-- the two VRAM levers compute the stock loss --")
cfg = dict(n_layer=3, n_head=4, n_embd=64, block_size=64, vocab_size=512, bias=False, dropout=0.0,
           **STOCK_ARCH) # "stock loss" here means the stock block, spelled out
torch.manual_seed(0)
base = GPT(GPTConfig(**cfg))
base.train()
X = torch.randint(0, 512, (4, 64))
Y = torch.randint(0, 512, (4, 64))
Y[:, -5:] = -1 # ignore_index has to survive chunking, including mid-chunk


def run(chunk, ckpt):
    m = copy.deepcopy(base)
    m.config.chunked_head_rows = chunk
    m.config.checkpoint = ckpt
    logits, loss = m(X.clone(), Y.clone())
    loss.backward() # train.py calls backward on this tensor; it must still be in the graph
    return loss, logits, {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}


l0, lg0, g0 = run(0, False)  # stock
l1, lg1, g1 = run(97, True)   # 97 does not divide 256: the ragged last chunk is the point
l2, lg2, g2 = run(256, False) # chunked head alone

check("chunked head returns a tensor loss, not a float", torch.is_tensor(l1), True)
check("chunked head loss is still attached to the graph", l1.grad_fn is not None, True)
check("chunked head returns logits=None", (lg1 is None) and (lg0 is not None), True)
check("every parameter still gets a gradient", len(g1), len(g0))
check("chunked head: loss matches stock", abs(l2.item() - l0.item()) < 1e-5, True)
check("chunked head: gradients match stock exactly",
      max(float((g2[n] - g0[n]).abs().max()) for n in g0) == 0.0, True)
check("chunked head + checkpoint: loss matches stock", abs(l1.item() - l0.item()) < 1e-5, True)
check("chunked head + checkpoint: gradients match stock",
      max(float((g1[n] - g0[n]).abs().max()) for n in g0) < 1e-5, True)

# ---------------------------------------------------------------- architecture flags
# The four levers in model.py/GPTConfig, all verified on CPU at small sizes plus one
# parameter-count check at the real 12L/768 size. None of these measure loss: that takes
# GPU runs. What they do prove is the algebra each lever claims, and that a typo cannot
# reach a training run.
print("\n-- what a bare train.py builds now, and what STOCK_ARCH is for --")
defaults = {k: GPTConfig.__dataclass_fields__[k].default for k in STOCK_ARCH}
check("GPTConfig defaults: all four levers on",
      {k: defaults[k] for k in ('norm', 'ffn', 'qk_norm', 'rope')},
      {'norm': 'rmsnorm', 'ffn': 'swiglu', 'qk_norm': True, 'rope': True})
check("STOCK_ARCH is the block these flags replaced", dict(STOCK_ARCH),
      {'norm': 'layernorm', 'ffn': 'gelu', 'qk_norm': False, 'rope': False, 'rope_base': 10000.0})
check("defaults and STOCK_ARCH disagree on all four levers, so nothing here is implicit",
      [k for k in ('norm', 'ffn', 'qk_norm', 'rope') if defaults[k] == STOCK_ARCH[k]], [])
check("rope_base is the one value they share", defaults['rope_base'], STOCK_ARCH['rope_base'])
# The stock block has to stay constructible from one constant: it is what every checkpoint
# written before these flags exists, and it is what GPT.from_pretrained has to build.
toy = dict(n_layer=1, n_head=2, n_embd=32, block_size=16, vocab_size=64, bias=False, dropout=0.0)
g_stock = GPT(GPTConfig(**toy, **STOCK_ARCH))
g_default = GPT(GPTConfig(**toy))
check("GPTConfig(**STOCK_ARCH): LayerNorms, a 4x GELU MLP, learned positions",
      (type(g_stock.transformer.ln_f).__name__, hasattr(g_stock.transformer.h[0].mlp, 'c_fc'),
       g_stock.transformer.wpe is not None, hasattr(g_stock.transformer.h[0].attn, 'rope_cos')),
      ('LayerNorm', True, True, False))
check("GPTConfig() with no arguments: RMSNorms, SwiGLU, no learned positions, rope tables",
      (type(g_default.transformer.ln_f).__name__, hasattr(g_default.transformer.h[0].mlp, 'w_gate'),
       g_default.transformer.wpe is None, hasattr(g_default.transformer.h[0].attn, 'rope_cos')),
      ('RMSNorm', True, True, True))
check("GPTConfig(**STOCK_ARCH) has no qk-norm weights to load",
      any('q_norm_weight' in n for n, _ in g_stock.named_parameters()), False)
check("the default block adds exactly 2 qk-norm weights (of size head_dim) per layer",
      sum('q_norm_weight' in n or 'k_norm_weight' in n for n, _ in g_default.named_parameters()),
      2 * toy['n_layer'])
# (No parameter-count comparison at this toy size on purpose: SwiGLU's hidden width is
# round(8/3*d/128)*128, which is only the param-matched 8/3*d when d is a multiple of 128. At
# d=32 it rounds to 384 instead of 85 and the comparison is meaningless. The exact accounting
# happens at the real 12L/768 size below.)

print("-- SwiGLU is parameter- and flop-matched to the GELU MLP --")
D = 768

def mlp(**kw):
    # **{**STOCK_ARCH, **kw}: the gelu arm has to be spelled out now that the default is swiglu,
    # and a plain **STOCK_ARCH, **kw would be a duplicate-keyword TypeError when kw sets ffn.
    return MLP(GPTConfig(n_layer=12, n_head=12, n_embd=D, block_size=16, vocab_size=1024,
                         bias=False, dropout=0.0, **{**STOCK_ARCH, **kw}))

def npar(m):
    return sum(p.numel() for p in m.parameters())

m_gelu, m_swiglu = mlp(), mlp(ffn='swiglu')
check("SwiGLU hidden width at d=768 is 2048", m_swiglu.c_proj.in_features, 2048)
check("GELU MLP params at d=768 (2*768*3072)", npar(m_gelu), 4718592)
check("SwiGLU MLP params == GELU MLP params (3*768*2048)", npar(m_swiglu), npar(m_gelu))
check("SwiGLU backward needs the narrower activation", m_swiglu.w_gate.out_features < m_gelu.c_fc.out_features, True)

xa = torch.randn(2, 8, D)
ym = m_swiglu(xa)
check("SwiGLU forward shape", tuple(ym.shape), (2, 8, D))
check("SwiGLU forward is finite", bool(torch.isfinite(ym).all()), True)
ym.pow(2).mean().backward()
check("SwiGLU: gate, up and down all get gradients",
      all(getattr(m_swiglu, n.split('.')[0]).weight.grad is not None
          for n in ['w_gate.weight', 'w_up.weight', 'c_proj.weight']), True)
try:
    mlp(ffn='mish'); bad_ffn = 'silently accepted'
except ValueError:
    bad_ffn = 'ValueError'
check("a typo'd --ffn is refused loudly", bad_ffn, 'ValueError')
try:
    make_norm(8, 'layar_norm', False); bad_norm = 'silently accepted'
except ValueError:
    bad_norm = 'ValueError'
check("a typo'd --norm is refused loudly", bad_norm, 'ValueError')

print("-- RMSNorm matches the reference, is scale invariant, and costs the same --")
xr = torch.randn(3, 4, 64)
rn = RMSNorm(64)
ref = xr * torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + 1e-5)
check("RMSNorm == x*rsqrt(mean(x^2)+eps) with weight=1", float((rn(xr) - ref).abs().max()) < 1e-6, True)
check("RMSNorm has no bias by default (like LayerNorm(bias=False))", rn.bias is None, True)
check("RMSNorm params == LayerNorm(bias=False) params", npar(rn), npar(LayerNorm(64, bias=False)))
# What the two actually differ on: both ignore a uniform *scale* of the input (that is the
# property QK-norm is buying), but LayerNorm also divides out the mean, so it ignores an added
# constant and RMSNorm does not.
check("RMSNorm output ignores a scale change", float((rn(xr * 100) - rn(xr)).abs().max()) < 1e-3, True)
check("LayerNorm output ignores a scale change too",
      float((LayerNorm(64, bias=False)(xr * 100) - LayerNorm(64, bias=False)(xr)).abs().max()) < 1e-3, True)
check("LayerNorm output ignores an added constant",
      float((LayerNorm(64, bias=False)(xr + 5.0) - LayerNorm(64, bias=False)(xr)).abs().max()) < 1e-4, True)
check("RMSNorm output does not ignore an added constant", float((rn(xr + 5.0) - rn(xr)).abs().max()) > 0.5, True)

small = dict(n_layer=3, n_head=4, n_embd=64, block_size=32, vocab_size=256, bias=False, dropout=0.0,
             **STOCK_ARCH) # every "stock" arm below says so; the arch arms override single keys
g_ln = GPT(GPTConfig(**small))
g_rms = GPT(GPTConfig(**dict(small, norm='rmsnorm')))
check("layernorm -> rmsnorm: same parameter count", npar(g_rms), npar(g_ln))
check("layernorm -> rmsnorm: reaches ln_f too", type(g_rms.transformer.ln_f).__name__, 'RMSNorm')
# The reason train.py checks these flags on resume instead of trusting load_state_dict:
# LayerNorm(bias=False) and RMSNorm expose the identical state_dict, so a swapped
# architecture loads cleanly and silently trains a different function.
check("layernorm -> rmsnorm: state_dict keys are identical",
      sorted(g_rms.state_dict()) == sorted(g_ln.state_dict()), True)

print("-- QK-norm: the decisive test is what happens when q and k grow --")
acfg = dict(n_layer=2, n_head=4, n_embd=64, block_size=64, vocab_size=128, bias=False, dropout=0.0,
            **STOCK_ARCH)
att = CausalSelfAttention(GPTConfig(**acfg)).eval()
att_qk = CausalSelfAttention(GPTConfig(**dict(acfg, qk_norm=True))).eval()
sd = att.state_dict()
for name, prm in att_qk.named_parameters():
    if name in sd:
        prm.data.copy_(sd[name]) # identical q/k/v/out projections; the new norm weights stay 1
HEAD_DIM = acfg['n_embd'] // acfg['n_head']
q_in = torch.randn(2, 16, acfg['n_embd'])
y_plain = att(q_in)
y_qk, y_qk100 = att_qk(q_in), att_qk(q_in * 100)
rel_qk = float((y_qk100 - y_qk * 100).abs().max() / y_qk100.abs().max())
check("qk_norm=True: out(100x) == 100*out(x)", rel_qk < 1e-3, True)
y_plain100 = att(q_in * 100)
rel_plain = float((y_plain100 - y_plain * 100).abs().max() / y_plain100.abs().max())
check("qk_norm=False: the same identity fails", rel_plain > 0.05, True)
print(f"info  relative gap of that identity: qk_norm on {rel_qk:.2e}, off {rel_plain:.2e}")

orig_sdpa = torch.nn.functional.scaled_dot_product_attention
seen = {}
def spy(q, k, v, **kw): # model.py calls SDPA through the module attribute, so this is observable
    seen['qk'] = (q.detach().clone(), k.detach().clone())
    return orig_sdpa(q, k, v, **kw)
torch.nn.functional.scaled_dot_product_attention = spy
try:
    att_qk(q_in * 100); q1, k1 = seen['qk']
    att(q_in * 100); q2, k2 = seen['qk']
finally:
    torch.nn.functional.scaled_dot_product_attention = orig_sdpa
l_qk = float((q1 @ k1.transpose(-2, -1)).abs().max() / math.sqrt(HEAD_DIM))
l_plain = float((q2 @ k2.transpose(-2, -1)).abs().max() / math.sqrt(HEAD_DIM))
print(f"info  max |scaled logit| at 100x input: qk_norm on {l_qk:.3f} (bound sqrt(head_dim)="
      f"{math.sqrt(HEAD_DIM):.1f}), off {l_plain:.1f}")
check("qk_norm bounds the logits by sqrt(head_dim)", l_qk <= math.sqrt(HEAD_DIM) + 1e-2, True)
check("without qk_norm there is no such bound", l_plain > math.sqrt(HEAD_DIM), True)

print("-- RoPE: isometry, unit cos/sin, dependence on relative position only --")
att_r = CausalSelfAttention(GPTConfig(**dict(acfg, rope=True))).eval()
rot = torch.randn(2, 4, 12, HEAD_DIM)
check("RoPE preserves every vector's norm",
      float((att_r._apply_rope(rot).norm(dim=-1) - rot.norm(dim=-1)).abs().max()) < 1e-5, True)
check("cos^2 + sin^2 == 1 across the whole table",
      float((att_r.rope_cos ** 2 + att_r.rope_sin ** 2 - 1).abs().max()) < 1e-6, True)
check("tables are fp32 constants, not parameters", (
      att_r.rope_cos.dtype == torch.float32 and not any(
          'rope' in n for n, _ in att_r.named_parameters())), True)
check("tables are not in the state_dict (persistent=False)",
      any('rope' in k for k in att_r.state_dict()), False)
qb, kb = torch.randn(1, 4, 10, HEAD_DIM), torch.randn(1, 4, 10, HEAD_DIM)
p, s = 3, 4 # same q and k, shifted by s in absolute position: relative offsets unchanged
logits_a = att_r._apply_rope(qb, p) @ att_r._apply_rope(kb, 0).transpose(-2, -1)
logits_b = att_r._apply_rope(qb, p + s) @ att_r._apply_rope(kb, s).transpose(-2, -1)
check("logits depend on relative offsets only", float((logits_a - logits_b).abs().max()) < 1e-4, True)
try:
    att_r._apply_rope(qb, start=acfg['block_size'])
    past_end = 'no error'
except AssertionError:
    past_end = 'AssertionError'
check("reading the rope table past block_size is caught", past_end, 'AssertionError')

g_rope = GPT(GPTConfig(**dict(small, rope=True))) # rope on, everything else stock
check("rope=True drops the learned position embedding", g_rope.transformer.wpe is None, True)
check("rope=True: get_num_params still works", g_rope.get_num_params() > 0, True)
Xs = torch.randint(0, small['vocab_size'], (2, small['block_size']))
Ys = torch.randint(0, small['vocab_size'], (2, small['block_size']))
lg, ls = g_rope(Xs, Ys)
check("rope=True forward produces a loss", ls is not None and bool(torch.isfinite(ls)), True)

print("-- all four together: a real forward/backward, and the exact parameter delta --")
g_arch = GPT(GPTConfig(**dict(small, norm='rmsnorm', ffn='swiglu', qk_norm=True, rope=True)))
g_arch.train()
lg, ls = g_arch(Xs, Ys)
ls.backward()
check("all four flags: loss is finite", bool(torch.isfinite(ls)), True)
check("all four flags: every parameter gets a gradient",
      sum(p.grad is not None for p in g_arch.parameters()), len(list(g_arch.parameters())))
check("all four flags: no learned wpe, rmsnorm in the blocks",
      (g_arch.transformer.wpe, type(g_arch.transformer.h[0].ln_1).__name__,
       type(g_arch.transformer.h[0].mlp).__name__), (None, 'RMSNorm', 'MLP'))
check("all four flags: the block MLP is the SwiGLU branch",
      hasattr(g_arch.transformer.h[0].mlp, 'w_gate'), True)
g_arch.crop_block_size(16)
check("crop_block_size shrinks the rope tables", g_arch.transformer.h[0].attn.rope_cos.shape[0], 16)
check("crop_block_size leaves no wpe to shrink", g_arch.transformer.wpe, None)
# .contiguous() here is the test being polite, not the model needing help: stock model.py
# calls targets.view(-1), which needs a contiguous targets, and train.py always gives it one.
lg2, ls2 = g_arch(Xs[:, :16], Ys[:, :16].contiguous())
check("crop_block_size: a shorter forward still works", bool(torch.isfinite(ls2)), True)

big = dict(n_layer=12, n_head=12, n_embd=768, block_size=1024, vocab_size=50304, bias=False,
           **STOCK_ARCH) # the real 124M block: stock, because these counts are the published ones
def big_params(**kw):
    m = GPT(GPTConfig(**{**big, **kw}))
    n = m.get_num_params()
    del m
    gc.collect()
    return n

n_stock, n_arch = big_params(), big_params(norm='rmsnorm', ffn='swiglu', qk_norm=True, rope=True)
# 50304*768 (tied wte == lm_head) + 12*(c_attn + c_proj + c_fc + c_proj + 2 norms) + ln_f.
# wpe (1024*768) is excluded by get_num_params, and the rope model has no wpe at all.
check("12L/768 stock parameter count", n_stock, 38633472 + 12 * 7079424 + 768)
check("12L/768 stock parameter count is 123,587,328", n_stock, 123587328)
check("all four flags add exactly 2*head_dim*n_layer parameters", n_arch - n_stock, 2 * 64 * 12)
print(f"info  12L/768: stock {n_stock:,} params -> all four flags {n_arch:,} "
      f"(+{100 * (n_arch - n_stock) / n_stock:.4f}%)")

print("-- resume refuses a checkpoint that was trained with different architecture flags --")
# End to end, against the real train.py, on CPU with a throwaway dataset: the guard is the
# main foot-gun of these flags, because load_state_dict would accept most of the swaps.
tiny_dir = os.path.join(REPO, 'data', 'selfcheck_tiny')
tmp_root = tempfile.mkdtemp()
try:
    os.makedirs(tiny_dir, exist_ok=True)
    tokens = (np.arange(2048) % 64).astype(np.uint16)
    tokens.tofile(os.path.join(tiny_dir, 'train.bin'))
    tokens.tofile(os.path.join(tiny_dir, 'val.bin'))
    with open(os.path.join(tiny_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump({'vocab_size': 64}, f)

    def build_ckpt(name, old_format=False, **arch):
        # a fixture checkpoint for the resume tests: the stock block unless asked otherwise,
        # because the checkpoints that actually exist on this box were all trained that way
        cfg = GPTConfig(n_layer=1, n_head=2, n_embd=32, block_size=16, vocab_size=64,
                        bias=False, dropout=0.0, **{**STOCK_ARCH, **arch})
        model = GPT(cfg)
        out_dir = os.path.join(tmp_root, name)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
            pickle.dump({'vocab_size': 64}, f)
        saved_args = {k: v for k, v in cfg.__dict__.items()
                      if k not in ('chunked_head_rows', 'checkpoint')}
        if old_format:
            for k in ('norm', 'ffn', 'qk_norm', 'rope', 'rope_base'):
                saved_args.pop(k, None) # a checkpoint written before these flags existed
        torch.save(dict(model=model.state_dict(), iter_num=0, best_val_loss=float('inf'),
                        optimizer=model.configure_optimizers(0.1, 6e-4, (0.9, 0.95), 'cpu').state_dict(),
                        model_args=saved_args), os.path.join(out_dir, 'ckpt.pt'))
        return out_dir

    refused = resumed = 0

    def run_train(out_dir, extra_args, init='resume', config_file=None):
        global refused, resumed # only to report how many cases trained and how many refused
        env = dict(os.environ, NANOAMD='0', HIP_VISIBLE_DEVICES='', CUDA_VISIBLE_DEVICES='')
        # --block_size is not optional here, and that is a pre-existing nanoGPT rule, not
        # something these flags introduced: resume puts the checkpoint's block_size into the
        # model, but get_batch() slices using the module-level block_size variable. Pass a
        # bigger one and the first forward dies on `t <= config.block_size`. lr_decay_iters
        # must also exceed warmup_iters, or get_lr() divides by zero on the first step.
        # eval_interval=1 is what makes train.py reach its checkpoint-save branch at all:
        # with max_iters=1 and the stock 1000 it trains one step and saves nothing, so any
        # claim about "what the new checkpoint recorded" would be reading the wrong file.
        argv = [sys.executable, 'train.py']
        if config_file:
            argv.append(config_file) # configurator.py applies arguments left to right
        argv += [f'--init_from={init}', '--device=cpu', '--compile=False',
                 '--dataset=selfcheck_tiny', f'--out_dir={out_dir}', '--max_iters=1',
                 '--log_interval=1', '--eval_interval=1', '--eval_iters=1', '--batch_size=2',
                 '--block_size=16', '--gradient_accumulation_steps=1', '--warmup_iters=1',
                 '--lr_decay_iters=4', '--learning_rate=6e-4', '--min_lr=6e-5'] + extra_args
        proc = subprocess.run(argv, cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
        if proc.returncode == 0:
            resumed += 1
        else:
            refused += 1
        return proc.returncode, proc.stdout + proc.stderr

    out = build_ckpt('gelu')
    code, log = run_train(out, ['--ffn=swiglu'])
    check("resume with --ffn=swiglu against a gelu ckpt: refused",
          (code != 0, 'architecture flags differ' in log), (True, True))
    check("  ...and says which flag", 'ffn' in log and "'gelu'" in log and "'swiglu'" in log, True)
    check("  ...and refuses before building any model or training any step",
          ('Resuming training from' in log, 'number of parameters' in log, 'iter 0:' in log),
          (True, False, False))

    # The whole point of STOCK_ARCH. A checkpoint that never mentioned these keys was trained on
    # the stock block; train.py's defaults are now the opposite block, so an unqualified resume
    # of it has to stop rather than load GELU weights into SwiGLU-shaped matrices.
    STOCK_FLAGS = ['--norm=layernorm', '--ffn=gelu', '--qk_norm=False', '--rope=False',
                   '--rope_base=10000.0']
    out = build_ckpt('gelu_pre_flag', old_format=True)
    before = torch.load(os.path.join(out, 'ckpt.pt'), map_location='cpu', weights_only=True)['model_args']
    check("  ...that fixture really had no architecture keys",
          [k for k in before if k in ('norm', 'ffn', 'qk_norm', 'rope')], [])
    code, log = run_train(out, [])
    check("resume of a pre-flag ckpt with today's defaults: refused", code != 0, True)
    check("  ...and the refusal prints the flags that would make it work",
          all(flag in log for flag in STOCK_FLAGS), True)
    code, log = run_train(out, STOCK_FLAGS)
    check("the same pre-flag ckpt resumes once the stock flags are spelled out", code == 0, True)
    if code == 0:
        rearmed = torch.load(os.path.join(out, 'ckpt.pt'), map_location='cpu', weights_only=True)
        check("  ...train.py's new checkpoint records the architecture it actually used",
              {k: rearmed['model_args'][k] for k in ('norm', 'ffn', 'qk_norm', 'rope')},
              {k: STOCK_ARCH[k] for k in ('norm', 'ffn', 'qk_norm', 'rope')})
        code2, log2 = run_train(out, ['--ffn=swiglu'])
        check("  ...and refusing happens on the *second* resume too", code2 != 0, True)

    # ...and the new default round-trips: an arch-ON checkpoint resumes with no flags at all.
    out = build_ckpt('arch_all', norm='rmsnorm', ffn='swiglu', qk_norm=True, rope=True)
    code, log = run_train(out, [])
    check("a checkpoint built from today's defaults resumes with no flags", code == 0, True)
    code, log = run_train(out, ['--rope=False'])
    check("resume with --rope=False against a rope ckpt: refused",
          (code != 0, 'architecture flags differ' in log), (True, True))
    check("  ...and says which flag", 'rope' in log, True)

    out = build_ckpt('rms', norm='rmsnorm')
    code, log = run_train(out, ['--norm=rmsnorm', '--ffn=gelu', '--qk_norm=False', '--rope=False'])
    check("resume with matching flags: allowed", code == 0, True)

    out = build_ckpt('old', old_format=True)
    code, log = run_train(out, ['--norm=rmsnorm'])
    check("a pre-flag ckpt cannot be silently continued as rmsnorm", code != 0, True)

    # The other direction: drive train.py with the real arch config file, shrunk to a toy
    # model so it costs a second on CPU. Proves the config file is accepted and that its
    # values reach GPTConfig and the checkpoint, which is what a 7600-iteration run depends on.
    out = os.path.join(tmp_root, 'scratch_arch')
    code, log = run_train(out, ['--n_layer=2', '--n_head=4', '--n_embd=64'],
                          init='scratch', config_file='config/train_fineweb_edu_arch.py')
    check("config/train_fineweb_edu_arch.py runs end to end (scratch, CPU, toy size)", code == 0, True)
    if code == 0:
        saved = torch.load(os.path.join(out, 'ckpt.pt'), map_location='cpu', weights_only=True)
        check("  ...the config file's flags reached GPTConfig and the checkpoint",
              {k: saved['model_args'][k] for k in ('norm', 'ffn', 'qk_norm', 'rope')},
              {'norm': 'rmsnorm', 'ffn': 'swiglu', 'qk_norm': True, 'rope': True})
        check("  ...a rope run saved no wpe in the state_dict",
              any('wpe' in k for k in saved['model']), False)

    # The same for the baseline config: now that the defaults are the new block, that file is
    # only a baseline because it spells 'stock' out. If someone deletes those five lines the
    # baseline arm quietly becomes the arch arm and the A/B measures nothing.
    out = os.path.join(tmp_root, 'scratch_baseline')
    code, log = run_train(out, ['--n_layer=2', '--n_head=4', '--n_embd=64'],
                          init='scratch', config_file='config/train_fineweb_edu_amd.py')
    check("config/train_fineweb_edu_amd.py runs end to end (scratch, CPU, toy size)", code == 0, True)
    if code == 0:
        saved = torch.load(os.path.join(out, 'ckpt.pt'), map_location='cpu', weights_only=True)
        check("  ...the baseline config pinned the stock block, not the defaults",
              {k: saved['model_args'][k] for k in ('norm', 'ffn', 'qk_norm', 'rope')},
              {k: STOCK_ARCH[k] for k in ('norm', 'ffn', 'qk_norm', 'rope')})
        check("  ...and a stock run saved its learned position embeddings",
              any('wpe' in k for k in saved['model']), True)

    print(f"info  resume cases run against the real train.py on CPU: {resumed} resumed, "
          f"{refused} refused (data/selfcheck_tiny is deleted below)")
finally:
    shutil.rmtree(tiny_dir, ignore_errors=True)
    shutil.rmtree(tmp_root, ignore_errors=True)

print("-- train.py exposes the flags in the form configurator.py demands --")
src = open(os.path.join(REPO, 'train.py')).read()
tree = ast.parse(src)
exec_line = min(node.lineno for node in ast.walk(tree)
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and getattr(node.value.func, 'id', '') == 'exec')
assigned = {}
for node in tree.body:
    if isinstance(node, ast.Assign) and node.lineno < exec_line:
        for t in node.targets:
            if isinstance(t, ast.Name):
                assigned[t.id] = node.value
want = {'norm': str, 'ffn': str, 'qk_norm': bool, 'rope': bool, 'rope_base': float}
for k, typ in want.items():
    node = assigned.get(k)
    val = node.value if isinstance(node, ast.Constant) else None
    field = GPTConfig.__dataclass_fields__[k].default
    # configurator.py:42 asserts type(attempt) == type(globals()[key]), with no message.
    # A bool written as 0 or a float written as 76 fails there, so the default types matter.
    check(f"train.py:{k} is assigned before configurator.py runs", type(val) is typ, True)
    check(f"train.py:{k} type matches GPTConfig's field type", type(val) is type(field), True)
    # ...and the value matches too. If train.py's copy of the default drifted from GPTConfig's,
    # a config file that names neither would build a different model than model.py promises.
    check(f"train.py:{k} equals GPTConfig's default", val == field, True)
ma = next(n for n in tree.body if isinstance(n, ast.Assign)
          and getattr(n.targets[0], 'id', '') == 'model_args')
ma_keys = {kw.arg for kw in ma.value.keywords}
check("model_args carries all five architecture flags", set(want) <= ma_keys, True)
check("the VRAM levers stay out of model_args (old checkpoints stay resumable)",
      bool({'chunked_head_rows', 'checkpoint'} & ma_keys), False)
check("train.py refuses to resume a checkpoint with different architecture flags",
      'arch_keys' in src and 'architecture flags differ' in src, True)
# A checkpoint that predates the keys must mean STOCK_ARCH. If the guard instead filled the
# gaps with GPTConfig's defaults, every pre-flag checkpoint on this box would resume as the new
# block, and the failure mode is the silent kind: wrong math, right tensor shapes.
check("the resume guard fills absent arch keys from STOCK_ARCH, not from GPTConfig defaults",
      'checkpoint_model_args.get(k, STOCK_ARCH[k])' in src, True)
check("the resume refusal prints a command that would actually work",
      'replay' in src and 'python train.py --init_from=resume' in src, True)
model_src = open(os.path.join(REPO, 'model.py')).read()
check("GPT.from_pretrained pins the block to STOCK_ARCH (OpenAI weights have no SwiGLU/rope tensors)",
      'config_args.update(STOCK_ARCH)' in model_src, True)
sample_src = open(os.path.join(REPO, 'sample.py')).read()
check("sample.py fills a pre-flag checkpoint's model_args with STOCK_ARCH",
      '{**STOCK_ARCH, **checkpoint' in sample_src, True)
check("train.py adopts from_pretrained's forced block instead of trusting its own defaults",
      'forces the stock GPT-2 block' in src, True)

# Every key in every config file has to be a train.py global that exists *before*
# configurator.py runs, with the same type. That is the whole difference between a config
# file that works and a bare AssertionError at <string> line 42.
for cfg_file in ['config/train_fineweb_edu_amd.py', 'config/train_fineweb_edu_arch.py']:
    cfg_tree = ast.parse(open(os.path.join(REPO, cfg_file)).read())
    cfg_keys = {n.targets[0].id: n.value for n in cfg_tree.body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    unknown = sorted(k for k in cfg_keys if k not in assigned)
    mismatched = sorted(k for k, v in cfg_keys.items()
                        if k in assigned and isinstance(v, ast.Constant)
                        and isinstance(assigned[k], ast.Constant)
                        and type(v.value) is not type(assigned[k].value))
    check(f"{cfg_file}: {len(cfg_keys)} keys, all known to configurator.py", unknown, [])
    check(f"{cfg_file}: no type that configurator.py would assert on", mismatched, [])
    set_arch = sorted(k for k in want if k in cfg_keys)
    pinned = {k: cfg_keys[k].value for k in want if k in cfg_keys}
    check(f"{cfg_file}: sets all five architecture keys out in the open", set_arch, sorted(want))
    if cfg_file.endswith('_arch.py'):
        check(f"{cfg_file}: pinned to the current defaults (it restates them so the arm is explicit)",
              pinned, defaults)
    else:
        check(f"{cfg_file}: pinned to STOCK_ARCH -- that is what makes it the baseline arm, and what "
              f"makes out_fineweb_edu_amd/ckpt.pt resumable", pinned, dict(STOCK_ARCH))

print()
print('ALL OK' if not fails else f'{len(fails)} FAILED: {fails}')
sys.exit(1 if fails else 0)