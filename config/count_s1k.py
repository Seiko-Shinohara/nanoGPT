"""
COUNT arms at 1024 iterations: still small, but long enough that the arms separate.

    venv/bin/python train.py config/count_s1k.py                       # COUNT + amplification
    venv/bin/python train.py config/count_s1k.py --count_amplify=False # COUNT, no amplification
    venv/bin/python train.py config/count_s1k.py --count_table= --count_domain_table= \
        --count_lambda=0.0 --count_amplify=False                        # standard transformer

15.90M parameters (4 layers x 4 heads x 256), 1024 iterations x 8192 tokens = 8.4M tokens, so
D/N ~= 0.53. The paper's ladder runs at D/N = 10-40, so these arms are early in their own training
curves -- the loss is still falling steeply at the last iteration -- and they answer "does the COUNT
path train at all, and in which direction does it move at matched budget on this box", not "does COUNT
win at scale". Amplification on/off is Figure 16's ablation, at 15.9M instead of 190M.

Everything the three arms share: architecture, batch, token budget, schedule, learning rate, seed.
Only the COUNT switches differ.
"""

dataset = 'fineweb-edu'
out_dir = 'out_count_s1k'

block_size = 512
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.0
bias = False

# --- COUNT --------------------------------------------------------------------
count_table = 'count_tables/general150m'
count_domain_table = 'count_tables/math_s47'
count_lambda = 0.25
count_j = 6
count_k = 8
count_amplify = True
count_norm = 'j'
count_chunk = 256
count_workers = 8
count_prefetch_depth = 2
# 0 = draw train windows from the whole 2.0B-token train.bin. general150m describes only its first
# 150M tokens, so a window drawn from elsewhere resolves fewer candidates (measured: 21.0 candidates
# per position and 49.6% of next tokens present among them, against 23.0 and 69.9% for a window drawn
# inside the table's own span, and 21.1 / 50.9% on held-out val.bin). Leaving this at 0 keeps the
# train and eval distributions at the same coverage; setting it to 150000000 would train the model
# on a table that is ~30% more informative than the one it is evaluated with.
count_train_tokens = 0

# --- schedule: same shape as the smoke run, stretched to 1024 iterations -------
lr_schedule = 'wsd'
wsd_decay_fraction = 0.2
learning_rate = 2.4e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
warmup_iters = 32
lr_decay_iters = 1024
max_iters = 1024
min_lr = 2.4e-5
decay_lr = True

# --- data / eval ---------------------------------------------------------------
batch_size = 8
gradient_accumulation_steps = 2       # 8 x 2 x 512 = 8192 tokens per iteration
eval_interval = 128
eval_iters = 4
log_interval = 32
always_save_checkpoint = True
init_from = 'scratch'

compile = False
wandb_log = False
