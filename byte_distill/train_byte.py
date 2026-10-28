"""
Training script for the byte-distillation study (arXiv:2609.12303).

Supports the six configurations of the paper's Table 1:

  tokenization   objective      loss
  -----------   -----------   -------------------------------------------
  token        cross-entropy   CE over BPE tokens
  token        distillation    alpha * KL(top-k teacher) + (1-alpha) * CE
  byte         cross-entropy   CE over bytes (Marginalize-It stream)
  byte         distillation    alpha * KL(byte teacher) + (1-alpha) * CE
  byte_eot     cross-entropy   CE over bytes + <eot> (End-Of-Token stream)
  byte_eot     distillation    alpha * KL(byte teacher) + (1-alpha) * CE

Loss (paper eq. 3-5):  L_KD = alpha * L_KL + (1 - alpha) * L_CE
  L_KL = (1/n) sum_t sum_c p_T(c|x_<t) log( p_T(c|x_<t) / p_S(c|x_<t) )
  L_CE = -(1/n) sum_t log p_S(y_t | x_<t)
Distilled runs use alpha = 1 (pure distillation); supervised runs alpha = 0.

Evaluation uses BPB (bits per byte, paper eq. 6): tokenizer-independent,
special tokens (DOC/EOT/PAD/EOS/BOS) are excluded from both the loss sum and
the byte count. Training FLOPs follow the paper's Table 9 formula:
  F_total = 3 * (F_attn + F_qkvo + F_ffn + F_deemb)
  F_attn = 2*n*d*(l+0.5), F_qkvo = 8*n*d^2, F_ffn = 4*n*d*d_ffn, F_deemb = 2*d*V
and are logged to <out_dir>/scaling.jsonl together with BPB for the
scaling-law analysis (analyze.py).

Usage (byte distillation example):
  HIP_VISIBLE_DEVICES=1 python byte_distill/train_byte.py \
      --tokenization=byte_eot --objective=distill --alpha=1.0 \
      --data_dir=data/fineweb-small --byte_dir=out/byte_distill_study/byte_streams \
      --teacher_dir=out/byte_distill_study/teacher_dumps \
      --n_layer=6 --n_head=6 --n_embd=384 --block_size=11264 --batch_size=64 \
      --learning_rate=2e-3 --max_iters=3000 --out_dir=out/byte_distill_study/eot_distill
"""

import os
import sys
import time
import math
import json
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llama import Llama, LlamaConfig, default_d_ffn  # noqa: E402
from tokenizer import VOCAB_BYTE, VOCAB_BYTE_EOT  # noqa: E402

# -----------------------------------------------------------------------------
# default config values
# I/O
out_dir = 'out'
eval_interval = 250
log_interval = 10
eval_iters = 200
eval_batch_size = 0            # 0 -> use batch_size (eval runs without checkpointing)
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'   # 'scratch' or 'resume'
# data
data_dir = 'data/fineweb-small'      # BPE token stream (train.bin/val.bin/vocab.json)
byte_dir = ''                        # dir with byte_{mi,eot}_{train,val}.bin
teacher_dir = ''                     # dir with teacher dumps (distillation only)
tokenization = 'token'               # 'token' | 'byte' | 'byte_eot'
objective = 'ce'                     # 'ce' | 'distill'
alpha = 1.0                          # distillation weight (1 = pure distillation)
topk = 600                           # top-k for token distillation (paper: 600)
# model
n_layer = 6
n_head = 6
n_embd = 384
d_ffn = 0                            # 0 -> Llama default
block_size = 2048
batch_size = 64                      # micro-batch (sequences per forward)
grad_accum = 1                       # gradient accumulation (global batch = batch_size * grad_accum)
dropout = 0.0
qk_norm = False                      # paper A.4: used for byte-model stability
rope_base = 500000.0
# adamw optimizer (paper A.3)
learning_rate = 2e-3
max_iters = 3000
weight_decay = 1e-4
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# learning rate schedule (paper A.3: 10% warmup, cosine)
warmup_frac = 0.1
min_lr = 0.0                         # 0 -> learning_rate / 10
# system
device = 'cuda:0'
dtype = 'bfloat16'
compile = True
seed = 1337
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'configurator.py')).read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

assert tokenization in ('token', 'byte', 'byte_eot')
assert objective in ('ce', 'distill')
if objective == 'distill':
    assert teacher_dir, "distillation requires --teacher_dir"
if tokenization in ('byte', 'byte_eot'):
    assert byte_dir, "byte training requires --byte_dir"

torch.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = torch.device(device)
device_type = 'cuda' if 'cuda' in device.type else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# data
# -----------------------------------------------------------------------------
vocab_size = None
if tokenization == 'token':
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta_vocab_size = pickle.load(f)['vocab_size']
        vocab_size = meta_vocab_size
    else:
        vocab_size = 4096
    train_path, val_path = os.path.join(data_dir, 'train.bin'), os.path.join(data_dir, 'val.bin')
    teacher_kind = 'topk'
elif tokenization == 'byte':
    vocab_size = VOCAB_BYTE
    train_path = os.path.join(byte_dir, 'byte_mi_train.bin')
    val_path = os.path.join(byte_dir, 'byte_mi_val.bin')
    teacher_kind = 'mi'
else:
    vocab_size = VOCAB_BYTE_EOT
    train_path = os.path.join(byte_dir, 'byte_eot_train.bin')
    val_path = os.path.join(byte_dir, 'byte_eot_val.bin')
    teacher_kind = 'eot'

train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
print(f"tokenization={tokenization}, objective={objective}, alpha={alpha}")
print(f"train: {len(train_data):,} units, val: {len(val_data):,} units, vocab_size={vocab_size}")

# per-token byte lengths for BPB on token models (marker decodes to "\n")
token_byte_lengths = None
if tokenization == 'token':
    with open(os.path.join(data_dir, 'vocab.json')) as f:
        vj = json.load(f)
    V = vj['vocab_size']
    marker_id = vj['marker_id']
    tbl = np.ones(V, dtype=np.int64)
    for k, piece in vj['id_to_piece'].items():
        t = int(k)
        if t != marker_id:
            tbl[t] = len(piece.encode('utf-8'))
    token_byte_lengths = torch.tensor(tbl, dtype=torch.int64, device=device)

# teacher dumps (distillation only)
teacher_dist = None
topk_probs_mm = None
topk_idx_mm = None
if objective == 'distill':
    if teacher_kind == 'topk':
        n_tok = len(train_data)
        topk_probs_mm = np.memmap(os.path.join(teacher_dir, 'topk_probs_train.fp16'), dtype=np.float16, mode='r', shape=(n_tok, topk))
        topk_idx_mm = np.memmap(os.path.join(teacher_dir, 'topk_idx_train.int16'), dtype=np.int16, mode='r', shape=(n_tok, topk))
        print(f"top-k teacher dump: {topk} entries per position")
    else:
        n_units = len(train_data)
        teacher_dist = np.memmap(os.path.join(teacher_dir, f"tdist_{teacher_kind}_train.fp16"),
                                 dtype=np.float16, mode='r', shape=(n_units, vocab_size))
        print(f"byte teacher dump: {vocab_size}-way distributions")

# -----------------------------------------------------------------------------
# model
# -----------------------------------------------------------------------------
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, d_ffn=d_ffn,
                      block_size=block_size, vocab_size=vocab_size, dropout=dropout,
                      qk_norm=qk_norm, rope_base=rope_base, tie_weights=False)
    gptconf = LlamaConfig(**model_args)
    model = Llama(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device, weights_only=False)
    model_args = ckpt['model_args']
    gptconf = LlamaConfig(**model_args)
    model = Llama(gptconf)
    state_dict = ckpt['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
else:
    raise ValueError(init_from)
model.to(device)

iter_num = 0
best_val_bpb = 1e9
if init_from == 'resume':
    iter_num = ckpt['iter_num']
    best_val_bpb = ckpt['best_val_bpb']

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(ckpt['optimizer'])
    ckpt = None

if compile:
    print("compiling the model...")
    unoptimized_model = model
    model = torch.compile(model)
    raw_model = unoptimized_model
else:
    raw_model = model

# -----------------------------------------------------------------------------
# data loading
# -----------------------------------------------------------------------------
def get_batch(split, bs=None):
    data = train_data if split == 'train' else val_data
    bs = batch_size if bs is None else bs
    ix = torch.randint(len(data) - block_size, (bs,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def get_teacher(ix):
    """Teacher distributions for targets y = data[ix+1 : ix+1+L].
    Convention (dump_teacher.py): TopkProbs[t] / Tdist[p] are the teacher
    distributions for the target AT position t / p (i.e. predicting token t
    / byte p). So for a window starting at s, target y[j] = data[s+1+j]
    uses teacher row s+1+j."""
    L = block_size
    ix_np = ix.cpu().numpy().astype(np.int64)
    idx = ix_np[:, None] + 1 + np.arange(L)[None, :]
    if teacher_kind == 'topk':
        p = torch.from_numpy(topk_probs_mm[idx].astype(np.float32))
        i = torch.from_numpy(topk_idx_mm[idx].astype(np.int64))
        return p.to(device, non_blocking=True), i.to(device, non_blocking=True)
    t = torch.from_numpy(teacher_dist[idx].astype(np.float32))
    return t.to(device, non_blocking=True), None


# -----------------------------------------------------------------------------
# loss
# -----------------------------------------------------------------------------
def compute_loss(x, y, teacher=None, topk_idx=None):
    logits, _ = model(x, y)
    if objective == 'ce':
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        return loss
    logp_s = F.log_softmax(logits.float(), dim=-1)
    ce = -logp_s.gather(-1, y.unsqueeze(-1)).squeeze(-1)  # (B, L) nats
    if teacher_kind == 'topk':
        p_t = teacher  # (B, L, k)
        logp_s_t = logp_s.gather(-1, topk_idx)
        kl = (p_t * (torch.log(p_t.clamp_min(1e-12)) - logp_s_t)).sum(-1)  # (B, L)
        loss = (alpha * kl + (1 - alpha) * ce).mean()
    else:
        p_t = teacher  # (B, L, V_byte)
        kl = (p_t * (torch.log(p_t.clamp_min(1e-12)) - logp_s)).sum(-1)  # (B, L)
        valid = (p_t.sum(-1) > 0).float()  # first token of stream has no teacher dist
        loss = (valid * (alpha * kl + (1 - alpha) * ce)).sum() / valid.sum().clamp_min(1.0)
    return loss


# -----------------------------------------------------------------------------
# BPB evaluation (paper eq. 6)
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_bpb():
    model.eval()
    eval_bs = batch_size if eval_batch_size == 0 else eval_batch_size
    total_nats = 0.0
    total_bytes = 0
    total_ce = 0.0
    for k in range(eval_iters):
        x, y = get_batch('val', bs=eval_bs)
        logits, _ = model(x, y)  # pass targets to get logits at all positions
        logp = F.log_softmax(logits.float(), dim=-1)
        nll = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)  # (B, L) nats
        if tokenization == 'token':
            mask = torch.ones_like(y, dtype=torch.bool)
            nbytes = token_byte_lengths[y].sum()
        else:
            mask = (y < 256)
            nbytes = mask.sum()
        total_nats += (nll * mask).sum().item()
        total_bytes += int(nbytes.item())
        total_ce += nll.sum().item()
    model.train()
    bpb = total_nats * math.log2(math.e) / total_bytes
    return bpb, total_ce / (eval_iters * eval_bs * block_size)


# -----------------------------------------------------------------------------
# FLOPs accounting (paper Table 9)
# -----------------------------------------------------------------------------
def flops_per_unit():
    n, d, l = model_args['n_layer'], model_args['n_embd'], model_args['block_size']
    d_ffn = model_args.get('d_ffn') or default_d_ffn(d)
    V = model_args['vocab_size']
    f_attn = 2 * n * d * (l + 0.5)
    f_qkvo = 8 * n * d * d
    f_ffn = 4 * n * d * d_ffn
    f_deemb = 2 * d * V
    return 3 * (f_attn + f_qkvo + f_ffn + f_deemb)


FLOPS_PER_UNIT = flops_per_unit()
UNITS_PER_ITER = batch_size * block_size * grad_accum
print(f"FLOPs per unit (paper Table 9 formula): {FLOPS_PER_UNIT:.4e}")
print(f"units per iteration (global batch): {UNITS_PER_ITER:,}")


def total_flops(it):
    return FLOPS_PER_UNIT * it * UNITS_PER_ITER


# -----------------------------------------------------------------------------
# LR schedule (paper A.3: linear warmup 10%, cosine to min_lr)
# -----------------------------------------------------------------------------
warmup_iters = max(1, int(warmup_frac * max_iters))
if min_lr == 0.0:
    min_lr = learning_rate / 10


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, (max_iters - warmup_iters))
    decay_ratio = min(1.0, max(0.0, decay_ratio))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# -----------------------------------------------------------------------------
# logging
# -----------------------------------------------------------------------------
os.makedirs(out_dir, exist_ok=True)
scaling_log_path = os.path.join(out_dir, 'scaling.jsonl')
with open(os.path.join(out_dir, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)


def log_scaling(iter_, bpb, val_ce, lr):
    entry = dict(iter=iter_, flops=total_flops(iter_), bpb=bpb, val_ce=val_ce, lr=lr)
    with open(scaling_log_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')


# -----------------------------------------------------------------------------
# training loop
# -----------------------------------------------------------------------------
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
while True:
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0:
        bpb, val_ce = estimate_bpb()
        print(f"step {iter_num}: val BPB {bpb:.4f}, val CE {val_ce:.4f} nats/unit, "
              f"FLOPs {total_flops(iter_num):.3e}")
        log_scaling(iter_num, bpb, val_ce, lr)
        if bpb < best_val_bpb or always_save_checkpoint:
            best_val_bpb = bpb
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_bpb': best_val_bpb,
                    'config': config,
                }
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                print(f"saving checkpoint to {out_dir}")
    if iter_num == 0 and eval_only:
        break

    # forward / backward with optional gradient accumulation
    loss = None
    for micro_step in range(grad_accum):
        with ctx:
            if objective == 'distill':
                T, TK = get_teacher(X)
                loss = compute_loss(X, Y, teacher=T, topk_idx=TK)
            else:
                loss = compute_loss(X, Y)
        loss = loss / grad_accum
        # prefetch next micro-batch while the GPU is busy
        X, Y = get_batch('train')
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        print(f"iter {iter_num}: loss {loss.item():.4f}, time {dt * 1000:.0f}ms, lr {lr:.2e}")
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

# final eval + checkpoint
bpb, val_ce = estimate_bpb()
print(f"step {iter_num}: val BPB {bpb:.4f}, val CE {val_ce:.4f} nats/unit")
log_scaling(iter_num, bpb, val_ce, get_lr(iter_num))
checkpoint = {
    'model': raw_model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': model_args,
    'iter_num': iter_num,
    'best_val_bpb': best_val_bpb,
    'config': config,
}
torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
print(f"final checkpoint saved to {out_dir}")