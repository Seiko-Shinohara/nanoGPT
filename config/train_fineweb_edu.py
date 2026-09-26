"""
Config: train a GPT-2 (124M) from scratch on the FineWeb-Edu parquet shards
in data/fineweb-edu/, streamed and tokenized on the fly (no .bin files).

  tokens/iter = batch_size * gradient_accumulation_steps * block_size
              = 24 * 20 * 1024 = 491,520

Set how many tokens to train on with `train_tokens`; max_iters is then
derived automatically as train_tokens / tokens_per_iter. val is streamed
from the last file(s), after the train cutoff, so there is no overlap.

Run:
  python train.py config/train_fineweb_edu.py
"""

# I/O
out_dir = 'out_fineweb_edu'
eval_interval = 100
eval_iters = 100  # ~1.23M tokens ~= one pass over the val budget
always_save_checkpoint = False  # only checkpoint on val-loss improvement
init_from = 'scratch'
wandb_log = False

# data: how many tokens to train on (-1 = all files)
train_tokens = 100_000_000
val_tokens = 1_000_000        # val tokens, streamed from the last file(s)

# effective batch: 480 seqs * 1024 = 491,520 tokens/iter
gradient_accumulation_steps = 20
batch_size = 24
block_size = 1024

# model (GPT-2 / 124M)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# optimizer
learning_rate = 6e-4
max_iters = -1  # auto: train_tokens / tokens_per_iter
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate schedule (cosine decay to min_lr over the run)
decay_lr = True
warmup_iters = 10
lr_decay_iters = -1  # auto: = max_iters

# system
device = 'cuda'
dtype = 'bfloat16'
compile = True
