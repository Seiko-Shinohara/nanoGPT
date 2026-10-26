"""
The Sec. 3.3 baseline at the same budget as config/count_s1k.py: learned n-gram embeddings instead of
COUNT's host-resident score tables. Over-Encoding as LongCat deploys it, Eq. 3.

    venv/bin/python train.py config/count_ngram_s1k.py                     # e_t + q_t, sparse rows
    venv/bin/python train.py config/count_ngram_s1k.py --ngram_host_tables=True
    venv/bin/python train.py config/count_ngram_s1k.py --ngram_sparse=False  # full-table updates
    venv/bin/python train.py config/count_s1k.py                            # the COUNT arm to compare to
    venv/bin/python train.py config/count_s1k.py --count_table= --count_domain_table= \
        --count_lambda=0.0 --count_amplify=False                            # the standard arm

Architecture, batch, token budget, schedule, learning rate and seed are the same as count_s1k.py, and
the tables are sized by the paper's own rule rather than by hand: ngram_rows=0 means "make the table
parameters equal the parameters excluding the input embeddings" (Sec. 3.3). At n_embd=256 that is
3,026,304 table parameters plus a 65,536-parameter projection, in 8 tables totalling 94,572 rows of
width 32 -- measured by train.py's [mem] report as 11.5 MiB of tables and 144.9 MiB of optimizer state
on the device for a 18.99M-parameter arm. That is the cost the paper is contrasting with COUNT's
zero-parameter tables, and it is the number this repo can measure exactly rather than estimate.

ngram_width_div=8 gives rows of d/8=32 lifted to 256 by a projection, which is LongCat's shape.
count_amplify=True is LongCat's embedding amplification (Appendix B.6) applied to e_t + q_t instead of
e_t + f_t; the knob is shared because it is the same LayerNorm in the same place.

What this arm is not: the paper's runs are 60M-1B parameters at D/N >= 10 with |V| = 100,278. This is
15.90M backbone parameters, 18.99M with the tables, 8.4M tokens, D/N ~= 0.53, tiktoken gpt2 at 50,304. It checks that the baseline trains,
that the sparse row updates actually work on this box, and what the throughput and memory cost is next
to the COUNT arm -- not that the baseline loses by the paper's margin.
"""

dataset = 'fineweb-edu'
out_dir = 'out_ngram_s1k'

block_size = 512
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.0
bias = False

# --- learned n-grams (Eq. 3) ---------------------------------------------------
ngram_orders = 3        # orders 2 and 3: bigrams and trigrams, the paper's NL = 3
ngram_tables = 4        # R = 4 hash tables per order
ngram_rows = 0          # derive M_{n,r} from the 1:1 ratio below
ngram_ratio = 1.0       # Sec. 3.3: table params : everything except the input embeddings
ngram_width_div = 8     # rows of d/8, projected back to d
ngram_sparse = True     # retrieved rows only. The paper measured full-table updates at 4.4% of throughput
ngram_host_tables = False # False keeps the 11.5 MiB of tables and their Adam moments on the GPU

# --- no COUNT ------------------------------------------------------------------
count_table = ''
count_domain_table = ''
count_lambda = 0.0
count_amplify = True    # sqrt(d) LayerNorm on h_t^(0) = e_t + q_t, as LongCat applies it

# --- schedule: identical to config/count_s1k.py --------------------------------
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
