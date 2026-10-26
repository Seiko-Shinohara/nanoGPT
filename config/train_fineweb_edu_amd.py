"""
nanoGPT config: GPT-2 124M on FineWeb-EDU, tuned for this AMD box.

Run it with:
    python train.py config/train_fineweb_edu_amd.py
(amd_opt.py pins the process to device 1 itself; no env vars needed.)

Why these numbers (all measured on this machine, see AMD_NOTES.md):

- The 7800 XT (gfx1101) has ~47 TFLOP/s of bf16/fp16 matmul throughput, but only
  ~1.8 TFLOP/s of fp32 matmul throughput on this ROCm build. fp32 is 26x slower.
  dtype is therefore left to amd_opt/nanoGPT's default (bfloat16) -- never set
  dtype='float32' here.
- Attention needs TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 to get a flash kernel;
  amd_opt.py sets that from train.py's import and forbids the MATH fallback.
- block_size 1024 makes the lm_head logits the single biggest allocation
  (batch_size x 1024 x 50304 x 2 bytes = 0.82 GiB at bs=8). This is what OOMs a
  16 GB card, not attention.
- device 1 is also the GPU driving your desktop (0000:06:00.0, card2-DP-3 connected),
  so the working set is deliberately kept under the 8 GiB clamp amd_opt enforces,
  leaving the rest of the 16 GiB for the compositor.
"""

dataset = 'fineweb-edu'
out_dir = 'out_fineweb_edu_amd'

# --- model / data -------------------------------------------------------------
block_size = 1024
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# --- architecture: pinned to the stock GPT-2 block --------------------------------
# train.py/GPTConfig now default to rmsnorm + swiglu + qk_norm + rope, so a baseline arm has to
# say 'stock' out loud. Two reasons this is not cosmetic:
#   1. this file is the baseline of the A/B; if it followed the defaults both arms would be
#      the same architecture and the comparison would measure nothing;
#   2. out_fineweb_edu_amd/ckpt.pt was trained before the flags existed, and train.py maps a
#      checkpoint that silent-about-architecture onto STOCK_ARCH and refuses a resume that asks
#      for anything else. These five lines are what make that checkpoint resumable.
norm = 'layernorm'
ffn = 'gelu'
qk_norm = False
rope = False
rope_base = 10000.0

# --- throughput ---------------------------------------------------------------
# Measured on this card (tools/sweep5.sh): micro-batch 8 is the knee. 8 -> 12 bought
# no throughput (49.5k -> 49.9k tok/s) and cost 2.5 GiB; 16 did not fit at all.
batch_size = 8
gradient_accumulation_steps = 32   # 8 x 32 x 1024 = 262144 tokens per iteration
compile = True
tunable = True                     # TunableOp: 780 -> 650 ms/iter, a measured 18% gain

# --- schedule sized to the prepared dataset (2.0B tokens) --------------------
# 2.0e9 tokens / 262144 tokens per iter ~ 7600 iterations
max_iters = 7600
lr_decay_iters = 7600
warmup_iters = 400
min_lr = 6e-5
learning_rate = 6e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True

# --- eval cost: stock eval_iters=200 evaluates 2 x 200 x 8 x 1024 = 3.3M tokens
# every eval_interval, which is a large slice of a short run
eval_interval = 1000
eval_iters = 20
log_interval = 1
always_save_checkpoint = True
init_from = 'scratch'

wandb_log = False
wandb_project = 'fineweb-edu'
wandb_run_name = 'gpt2-124m-amd'

# --- when you move to the 7900 XTX (device 0, 24 GB) after stopping the LLM ---
# NANOAMD_DEVICE=0 python train.py config/train_fineweb_edu_amd.py \
#     --batch_size=16 --gradient_accumulation_steps=16
# Device 0 has no connected display outputs, so amd_opt applies no VRAM cap there.
# (16 x 16 x 1024 = 262144 tokens/iter -> the same 7600 iterations cover the corpus.)
# Measured on that card: 3527 ms/iter = 63.5 TFLOP/s. amd_opt's peak table now holds
# 76.0 TFLOP/s for gfx1100 (your matmul_probe number), so the printed mfu is already
# honest; --peak_tflops only needs setting if you re-measure with --size 8192.
