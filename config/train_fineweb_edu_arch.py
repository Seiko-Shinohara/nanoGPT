"""
nanoGPT config: the architecture arm of the A/B. Same data, same tokens per iteration,
same schedule as config/train_fineweb_edu_amd.py -- only the block itself differs.

Run the pair like this (device 0, nothing competing with it):

    NANOAMD_DEVICE=0 python train.py config/train_fineweb_edu_amd.py
    NANOAMD_DEVICE=0 python train.py config/train_fineweb_edu_arch.py

and compare val loss at iter 1000 / 2000 / 4000, i.e. at 262M / 524M / 1.05B tokens.
Single-lever arms come from the command line, because configurator.py applies arguments
left to right:

    python train.py config/train_fineweb_edu_arch.py --ffn=gelu        # drop just SwiGLU
    python train.py config/train_fineweb_edu_arch.py --qk_norm=False   # drop just QK-norm
    python train.py config/train_fineweb_edu_arch.py --rope=False      # drop just RoPE
    python train.py config/train_fineweb_edu_arch.py --norm=layernorm  # drop just RMSNorm

Everything below the "architecture" line is an expectation from the literature, not a
measurement on this box. See AMD_NOTES.md "Architecture flags" for what is measured and
what is not.
"""

dataset = 'fineweb-edu'
out_dir = 'out_arch_all'

# --- model / data: identical to the baseline on purpose ----------------------
block_size = 1024
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# --- architecture: the four levers, all on ----------------------------------
# norm      RMSNorm everywhere. Same parameter count as LayerNorm(bias=False), one reduction
#           instead of two. Small effect expected; the reason to include it is that QK-norm
#           wants the same op anyway.
# ffn       SwiGLU with hidden 8/3*768 -> 2048. Exactly 4,718,592 parameters per block, the
#           same as GELU's 4x MLP, so 123,587,328 parameters overall (verified in
#           tools/selfcheck.py). Narrower activation, so slightly less saved for backward.
# qk_norm   RMSNorm over the 64-dim head on q and k. Adds 2*64*12 = 1536 parameters and
#           bounds every attention logit by sqrt(head_dim)=8. This is the lever that makes a
#           higher learning_rate safe, which is where its loss benefit should come from.
# rope      rotary q/k instead of learned absolute wpe. Frees 1024*768 = 786,432 parameters
#           from the optimizer (they are not counted by get_num_params either way).
norm = 'rmsnorm'
ffn = 'swiglu'
qk_norm = True
rope = True
rope_base = 10000.0

# --- throughput: deliberately unchanged so the comparison is about loss ------
# tokens per iteration stays 8 x 32 x 1024 = 262144, the same number the baseline run is
# spending. If an arm is slower per iteration, it is *behind* on tokens at a given iter, and
# the comparison has to be read off tokens, not iterations.
batch_size = 8
gradient_accumulation_steps = 32
compile = True
tunable = True

# --- schedule: same shape, sized to the same 2.0B token corpus --------------
max_iters = 7600
lr_decay_iters = 7600
warmup_iters = 400
min_lr = 6e-5
learning_rate = 6e-4      # not raised here on purpose; QK-norm's headroom is a later step
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True

# --- eval: same cadence, so both arms print val loss at the same token counts
eval_interval = 1000
eval_iters = 20
log_interval = 1
always_save_checkpoint = True
init_from = 'scratch'     # these flags have no pretrained counterpart; see AMD_NOTES.md

wandb_log = False
wandb_project = 'fineweb-edu'
wandb_run_name = 'gpt2-124m-arch-all'

# --- for the cheap first pass, shorten both arms equally --------------------
# python train.py config/train_fineweb_edu_amd.py --max_iters=2000 --lr_decay_iters=2000
# python train.py config/train_fineweb_edu_arch.py --max_iters=2000 --lr_decay_iters=2000
# 2000 iters x 262144 = 524M tokens per arm. Keep --lr_decay_iters > --warmup_iters or
# get_lr() divides by zero.
