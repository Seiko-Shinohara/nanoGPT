"""
COUNT smoke config: does the whole COUNT path run end to end, in minutes, on this box?

Run it with:
    python train.py config/count_smoke.py
then check the checkpoint actually carries COUNT:
    python sample.py config/count_smoke.py --num_samples 4 --max_to_generate 64 --out_file smoke.txt

This is not a training run in any scientific sense: 3.7M parameters, 64 iterations, a 150M-token
general table and a 22M-token math table. It exists because COUNT touches five different places
(data pipeline -> prefetch -> model input fold -> optimizer exemptions -> sampler) and a bug in any
of them is cheap to find at this size and expensive to find at 124M x 4700 iterations.

What it validates, in order:
  1. table loading and Sec. 5.1 mixing at lambda=0.25 (general + domain, disjoint corpora);
  2. the prefetcher feeding the training loop (count_wait should print close to 0 ms);
  3. Eq. 1 folding into the first block, including the sqrt(d) LayerNorm amplification;
  4. the resume guard: re-running with --count_j=8 must be refused, not silently retrained;
  5. sample.py picking the tables up from the checkpoint without being told them.

CPU-only fallback (no GPU): --device=cpu --batch_size=2 --max_iters=5 --eval_interval=5.
"""

dataset = 'fineweb-edu'
out_dir = 'out_count_smoke'

# --- model: small, but with the same COUNT knobs the real run uses --------------
block_size = 512
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.0
bias = False

# --- COUNT --------------------------------------------------------------------
# general150m: train.bin tokens 0..150M. math_s47: shards 4:8 of the corpus, which those 150M
# tokens never saw. So the domain table contributes information the general table does not have,
# and lambda > 0 is doing something real rather than reweighting a duplicate.
count_table = 'count_tables/general150m'
count_domain_table = 'count_tables/math_s47'
count_lambda = 0.25
count_j = 6
count_k = 8
count_amplify = True            # sqrt(d) LayerNorm over (f_t + e_t), Figure 17 / Appendix B.6
count_norm = 'j'                # Eq. 1 divides by J
count_chunk = 256               # keeps the f_t transient small; count_chunk >= batch*block is exact
count_workers = 8
count_prefetch_depth = 2        # lookups for the next micro-batch run while the GPU works

# --- schedule: WSD-S, as the paper's Table 2 uses ------------------------------
lr_schedule = 'wsd'
wsd_decay_fraction = 0.2
learning_rate = 2.4e-4          # 0.00235 * (N / 108e6)^(-1/3) at N = 3.7M, floored for a smoke run
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
warmup_iters = 10
lr_decay_iters = 64
max_iters = 64
min_lr = 2.4e-5
decay_lr = True

# --- data / eval ---------------------------------------------------------------
batch_size = 8
gradient_accumulation_steps = 2       # 8 x 2 x 512 = 8192 tokens per iteration
eval_interval = 16
eval_iters = 3
log_interval = 1
always_save_checkpoint = True
init_from = 'scratch'

compile = False                 # torch.compile is not needed to test COUNT plumbing
wandb_log = False
