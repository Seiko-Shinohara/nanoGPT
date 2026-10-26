"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import amd_opt  # AMD/ROCm setup: MUST be imported before torch (it sets env vars)
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT, STOCK_ARCH, NO_COUNT, NO_NGRAM, LayerNorm, make_norm
# The Sec. 3.3 baseline's optimizer plumbing: SparseAdam for the table rows, and dense_grad_params,
# because clip_grad_norm_ cannot reduce a sparse COO gradient and has to be shown only the dense ones.
from count_ngram.optim import dense_grad_params

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# Architecture levers (see AMD_NOTES.md "Architecture flags"). Unlike the two VRAM levers
# below these DO go into the checkpoint's model_args: they change what the network is, so
# train.py refuses to resume a checkpoint that was trained with different values.
# These four default ON, matching GPTConfig: at n_embd=768 they add no parameters and no
# matmul FLOPs. The consequence is that a checkpoint trained with the *stock* block -- which
# includes everything this repo ran before these flags existed, and every OpenAI GPT-2 weight --
# is now a non-default architecture, so it must be resumed with STOCK_ARCH spelled out.
# config/train_fineweb_edu_amd.py does exactly that. The expected loss effects are
# literature-based estimates; none of them has been measured on this box yet.
norm = 'rmsnorm' # 'layernorm' | 'rmsnorm': one reduction instead of two, same param count
ffn = 'swiglu' # 'gelu' | 'swiglu': SwiGLU at hidden 8/3*d (2048 at d=768) is param- and flop-matched
qk_norm = True # RMSNorm over head_dim on q,k: buys headroom to raise learning_rate, near-free
rope = True # rotary q/k positions, replaces wpe outright (a rope run cannot resume a wpe ckpt)
rope_base = 10000.0 # rotary base, only read when rope=True
tie_embeddings = True # False: lm_head gets its own matrix instead of sharing wte. Eq. 1 weights the
                      # *unembedding* rows, and a tied model's unembedding is its input embedding --
                      # the token-embedding basis Appendix B.7 / Figure 18 measures against the paper's
                      # choice and finds worse. Untying buys a COUNT arm the paper's basis for
                      # vocab_size*n_embd extra parameters, so it is off by default here too.
# VRAM levers for fitting a bigger micro-batch (see AMD_NOTES.md). Both are off by default
# and neither is saved into the checkpoint's model_args, so resuming an old ckpt still works.
chunked_head_rows = 0 # e.g. 4096: lm_head + cross_entropy 4096 rows at a time. Predicted ~40% off the peak of the vocab projection (measured on CPU; not yet measured on this model), costs a few % time
sync_timing = False # True: synchronize the device before the iteration timer stops. nanoGPT's dt is
                    # wall time *around the enqueue*, so an arm whose GPU kernels the CPU outruns is
                    # under-measured: the standard arm timed at 12.92ms/iter (a 133% mfu, impossible
                    # on a 47.2 TFLOP/s card) in a 64-iteration run while the same config measured
                    # 60ms in a longer one. COUNT's dt is not affected the same way because the
                    # prefetched lookup joins the CPU each iteration. Set this True whenever iteration
                    # time is the quantity being measured; leave it False to keep stock logging.
activation_checkpoint = False # recompute each block in backward: ~1 extra forward pass (~33% more compute) instead of storing block activations
# COUNT (The COUNT, Magnusson/Smith/Koh). Read README_COUNT.md before touching these.
# An empty count_table leaves the run a stock transformer, byte-for-byte: no lookup, no extra
# input term, no prefetch thread, no new parameters.
count_table = '' # directory written by ./count_native/count_build (has meta.bin, order*.bin)
count_domain_table = '' # optional second table, mixed in with count_lambda -- Sec. 5.1, no retraining
count_lambda = 0.0 # S_lambda(w|h) = (1-lambda) S_general(w|h) + lambda S_domain(w|h), not renormalized
count_j = 6 # J: stored suffix orders summed into f_t (paper trains with 6, ablates against 1)
count_k = 8 # K: candidates per order the model sums over (the paper stores 32, uses 8)
count_orders = 0 # 0: use every order the table stores. N: never look past order N (Sec. 5.2 knob)
count_backoff_depth = -1 # -1: back off to the unigram row. N: at most N discount steps
count_norm = 'j' # 'j': Eq. 1's fixed 1/J. 'nfound': divide by the rows that actually resolved
count_amplify = False # sqrt(d) LayerNorm(h_t^(0)), Appendix B.6. Off here so stock runs stay stock
count_chunk = 1024 # positions folded into f_t at a time; bounds the gather's transient memory
count_workers = 8 # CPU threads used by the table lookup (the paper's decode config: 8 threads)
count_prefetch_depth = 2 # lookups prepared ahead of the GPU. 0 = synchronous, 1 = nanoGPT-style
count_train_tokens = 0 # 0: draw train windows from all of train.bin. N: only from its first N tokens -- set this to the --tokens the table was built with, so f_t describes contexts the table actually saw
count_no_decay_embeddings = True # Table 2: input embeddings + amplification params skip weight decay
# The baseline COUNT is measured against (Sec. 3.3, Eq. 3): *learned* n-gram embeddings, the
# Over-Encoding method as LongCat deploys it. hash(x_{t-n+1..t}) picks a row out of a learned table of
# width n_embd/8, a projection lifts it to n_embd, and it is added to the token embedding. Unlike
# COUNT's tables these are parameters: they train with the backbone, need optimizer state, and have to
# be re-learned by every run -- which is the comparison the paper is making. Off by default.
ngram_orders = 0 # 0 = off. N >= 2: learn tables for n-gram orders 2..N (the paper uses 3: bigrams+trigrams)
ngram_tables = 4 # R independent hash tables per order
ngram_rows = 0 # rows per table M_{n,r}. 0 = derive from ngram_ratio (see AMD_NOTES/README: at small n_embd the 50k-vocab embedding alone outweighs the blocks and the 1:1 rule has no solution)
ngram_ratio = 1.0 # Sec. 3.3's 1:1 rule: table parameters vs the parameters excluding input embeddings
ngram_width_div = 8 # table row width d_s = n_embd / 8, projected back to n_embd
ngram_sparse = True # update retrieved rows only. False = full-table updates: the paper measured 4.4% of normal throughput
ngram_host_tables = False # keep the tables in host RAM and copy only retrieved rows to the GPU
# learning rate schedule. The paper uses WSD-S: warmup, flat, then a decay over the last 20%.
lr_schedule = 'cosine' # 'cosine' (nanoGPT's default) | 'wsd'
wsd_decay_fraction = 0.2 # fraction of lr_decay_iters spent decaying when lr_schedule='wsd'
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
tunable = False # AMD only: run TunableOp GEMM autotuning, cached to tools/tunable_<dtype>.json
peak_tflops = 0.0 # TFLOPS used for the mfu number; 0 = look it up in amd_opt's per-GPU table
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# AMD/ROCm: unlock AOTriton flash attention, forbid a silent fallback to the slow
# MATH backend, cap this process's VRAM, and (optionally) autotune GEMMs.
if device_type == 'cuda':
    amd_opt.configure(dtype=dtype, tunable=tunable)
    if peak_tflops <= 0:
        peak_tflops = amd_opt.recommended_peak_tflops()
        print(f"reporting mfu against {peak_tflops:.1f} TFLOP/s (set --peak_tflops to override)")

# poor man's data loader
data_dir = os.path.join('data', dataset)

# COUNT: open the host-resident tables once, and the pipelines that turn .bin windows into
# Eq. 1 batches. A table is mmap'd, not loaded: the pages come in as the binary searches touch
# them, which is why a table bigger than RAM still runs (slowly, from the page cache).
count_enabled = bool(count_table)
count_pipelines = {}
if count_enabled:
    from count.data import CountDataPipeline, open_tables, table_weights
    from count.table import set_threads

    if not os.path.isdir(count_table):
        raise SystemExit(f"count_table={count_table!r} is not a directory. Build it with "
                         f"./count_native/count_build --corpus ... --out-dir {count_table}")
    set_threads(count_workers)
    count_dirs = [count_table] + ([count_domain_table] if count_domain_table else [])
    count_tables = open_tables(count_dirs)
    count_weights = table_weights(count_lambda, len(count_tables))
    for t in count_tables:
        print(t.describe())
    print(f"COUNT on: J={count_j} K={count_k} orders={'all' if count_orders == 0 else count_orders}"
          f" norm={count_norm} amplify={count_amplify} lambda={count_lambda}"
          f" backoff_depth={count_backoff_depth} workers={count_workers}"
          f" prefetch_depth={count_prefetch_depth}")


def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout,
                  norm=norm, ffn=ffn, qk_norm=qk_norm, rope=rope, rope_base=rope_base,
                  tie_embeddings=tie_embeddings,
                  count=count_enabled, count_j=count_j, count_k=count_k,
                  count_norm=count_norm, count_amplify=count_amplify,
                  ngram_orders=ngram_orders, ngram_tables=ngram_tables, ngram_rows=ngram_rows,
                  ngram_ratio=ngram_ratio, ngram_width_div=ngram_width_div,
                  ngram_sparse=ngram_sparse, ngram_host_tables=ngram_host_tables) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # The architecture flags are checked, not copied. This is the dangerous one: LayerNorm
    # with bias=False and RMSNorm have identical state_dict keys, so load_state_dict below
    # would happily accept a LayerNorm checkpoint into an RMSNorm model and silently train a
    # different function. A checkpoint that does not mention these keys was written before they
    # existed, so it means STOCK_ARCH -- deliberately NOT GPTConfig's defaults, which since the
    # flags went live default to the opposite block.
    arch_keys = ['norm', 'ffn', 'qk_norm', 'rope', 'rope_base', 'tie_embeddings']
    ckpt_arch = {k: checkpoint_model_args.get(k, STOCK_ARCH[k]) for k in arch_keys}
    mismatch = {k: (ckpt_arch[k], model_args[k]) for k in arch_keys if ckpt_arch[k] != model_args[k]}
    if mismatch:
        for k, (was, now) in mismatch.items():
            print(f"checkpoint {k}={was!r} but this run asks for {k}={now!r}")
        replay = ' '.join(f'--{k}={ckpt_arch[k]}' for k in arch_keys)
        raise SystemExit(f"cannot resume {ckpt_path}: architecture flags differ. That checkpoint is a "
                         f"stock GPT-2 block. Continue it with: python train.py --init_from=resume {replay} "
                         f"--out_dir={out_dir} (and the same --batch_size/--block_size you were using), "
                         f"or start a scratch run in a different --out_dir.")
    for k in arch_keys:
        model_args[k] = ckpt_arch[k] # so the new checkpoint records what the model actually is
    # COUNT keys are checked the same way. J and K change the sum in Eq. 1, 'count' decides whether
    # the first block's input has an f_t at all, and count_amplify decides whether a LayerNorm sits
    # on that input -- a state_dict that loads cleanly is not evidence the model is the same model.
    count_keys = ['count', 'count_j', 'count_k', 'count_norm', 'count_amplify']
    ckpt_count = {k: checkpoint_model_args.get(k, NO_COUNT[k]) for k in count_keys}
    count_mismatch = {k: (ckpt_count[k], model_args[k]) for k in count_keys if ckpt_count[k] != model_args[k]}
    if count_mismatch:
        for k, (was, now) in count_mismatch.items():
            print(f"checkpoint {k}={was!r} but this run asks for {k}={now!r}")
        replay = ' '.join(f'--{k}={ckpt_count[k]}' for k in count_keys if k != 'count')
        want = "--count_table=<the table directory>" if ckpt_count['count'] else "(no --count_table)"
        raise SystemExit(f"cannot resume {ckpt_path}: COUNT settings differ. That checkpoint was "
                         f"trained {'with' if ckpt_count['count'] else 'without'} COUNT. Continue it "
                         f"with: python train.py --init_from=resume {want} {replay} "
                         f"--out_dir={out_dir}, or start a scratch run in a different --out_dir.")
    for k in count_keys:
        model_args[k] = ckpt_count[k]
    # Learned n-gram tables are parameters, so they are architecture: ngram_orders/tables decide which
    # tensors exist at all, ngram_width_div decides their width, and ngram_sparse/ngram_host_tables
    # decide how they are updated and where they live -- a resume that silently switches either one
    # changes the trained model (full-table updates decay rows the sparse path never touched).
    # ngram_rows and ngram_ratio are not checked here: ngram_rows=0 means "derive it from the ratio",
    # which a resume would re-derive, and the real guard is load_state_dict a few lines later -- it is
    # strict, so a table of the wrong size fails loudly on the tensor that is the wrong size.
    ngram_keys = ['ngram_orders', 'ngram_tables', 'ngram_width_div', 'ngram_sparse',
                  'ngram_host_tables']
    ckpt_ngram = {k: checkpoint_model_args.get(k, NO_NGRAM[k]) for k in ngram_keys}
    ngram_mismatch = {k: (ckpt_ngram[k], model_args[k]) for k in ngram_keys
                      if ckpt_ngram[k] != model_args[k]}
    if ngram_mismatch:
        for k, (was, now) in ngram_mismatch.items():
            print(f"checkpoint {k}={was!r} but this run asks for {k}={now!r}")
        replay = ' '.join(f'--{k}={ckpt_ngram[k]}' for k in ngram_keys)
        want = "--ngram_orders=<N>" if ckpt_ngram['ngram_orders'] else "(no --ngram_orders)"
        raise SystemExit(f"cannot resume {ckpt_path}: learned n-gram settings differ. That checkpoint "
                         f"was trained {'with' if ckpt_ngram['ngram_orders'] else 'without'} learned "
                         f"n-gram tables. Continue it with: python train.py --init_from=resume {want} "
                         f"{replay} --out_dir={out_dir}, or start a scratch run in a different --out_dir.")
    for k in ngram_keys:
        model_args[k] = ckpt_ngram[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
    # OpenAI's weights *are* a stock block, and GPT.from_pretrained pins norm/ffn/qk_norm/rope to
    # STOCK_ARCH for exactly that reason (model.py). Adopt those values so the checkpoint records
    # what the model actually is, and say out loud which of this run's flags the checkpoint
    # overrode -- now that the defaults are the new block, `--init_from=gpt2` alone would
    # otherwise ask for SwiGLU/QK-norm/rope weights that do not exist in the OpenAI checkpoint.
    forced = {k: (model_args[k], getattr(model.config, k)) for k in STOCK_ARCH
              if model_args[k] != getattr(model.config, k)}
    for k in STOCK_ARCH:
        model_args[k] = getattr(model.config, k)
    if forced:
        print("init_from=%s forces the stock GPT-2 block; overridden: %s" % (
            init_from, ", ".join(f"{k} {was!r} -> {now!r}" for k, (was, now) in forced.items())))
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
if model.ngram is not None:
    # --ngram_rows=0 means "derive the table size from the 1:1 ratio", which GPT did while building
    # itself. Record what it actually built: the checkpoint then says how many rows the tables really
    # have, and eval_lm.py can rebuild the same shapes from model_args without re-deriving anything.
    model_args['ngram_rows'] = model.ngram.rows_per_table
# apply the VRAM knobs. Deliberately kept out of model_args: they change memory, not
# architecture, so checkpoints written before they existed stay resumable.
model.config.chunked_head_rows = chunked_head_rows
model.config.checkpoint = activation_checkpoint
# COUNT knobs on the model. For scratch and resume the config already carries them -- the guard
# above guarantees they equal what this run asks for. For init_from=gpt2* the weights came from a
# model that never had an f_t term, so the term is bolted on and that is said out loud.
if count_enabled and init_from.startswith('gpt2'):
    print("note: init_from=%s loaded weights trained *without* COUNT. Eq. 1's f_t is being added "
          "to a model that never saw one, which is legal (it adds no parameters) but is not what "
          "the paper trained; expect the first iterations to be worse, not better." % init_from)
model.config.count = count_enabled
model.config.count_j = count_j
model.config.count_k = count_k
model.config.count_norm = count_norm
if count_amplify and model.count_amp is None:
    model.count_amp = LayerNorm(model.config.n_embd, True)
    print("created an untrained embedding-amplification LayerNorm for h_t^(0)")
model.to(device)
if model.ngram is not None and model.ngram.host_tables:
    # model.to(device) moved everything, including the tables this run wants to keep in host RAM.
    # The projections stay on the GPU: what crosses PCIe is the retrieved rows and their gradients.
    model.ngram.to_host()
    rows = model.ngram.table_rows() * model.ngram.width
    print(f"n-gram tables offloaded to host RAM: {rows * 4 / 2**20:.1f} MiB of table weights "
          f"(plus the same again in SparseAdam moments) instead of that much VRAM")

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
# Table 2 of the paper: weight decay 0.1 on everything except input embeddings, embedding
# amplification parameters, and the n-gram tables. COUNT's tables are not in the optimizer at all --
# they are host memory -- so only the embeddings need naming here. See model.py's
# configure_optimizers for why naming wte also spares the tied output head.
# The exemption is a property of Table 2, not of COUNT: it applies to the standard arm too, otherwise
# the arms differ in their weight decay as well as in their n-gram path. (Arms measured before this
# line said `if count_enabled` decayed their input embeddings; see README_COUNT.md.)
no_decay_names = ()
if count_no_decay_embeddings:
    no_decay_names = ('transformer.wte.weight', 'count_amp')
if model.ngram is not None and not model.ngram.sparse and count_no_decay_embeddings:
    # A learned n-gram table is only in the optimizer when it is being updated densely; the sparse
    # path hands those rows to SparseAdam, which has no weight decay to be exempted from.
    no_decay_names = no_decay_names + ('ngram.tables',)
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type,
                                       no_decay_names=no_decay_names)
# Sec. 3.3: the n-gram tables train with sparse row-wise AdamW at zero weight decay, on the backbone's
# schedule and betas. It is a second optimizer object because AdamW cannot accept the sparse gradients
# that nn.Embedding(sparse=True) produces.
ngram_optimizer = model.configure_ngram_optimizer(learning_rate, (beta1, beta2))
if ngram_optimizer is not None and dtype == 'float16':
    raise SystemExit(
        "learned n-gram tables cannot train in float16 here: GradScaler scales the AdamW optimizer's "
        "gradients and SparseAdam's sparse gradients are outside that, so the table rows would be "
        "stepped at the wrong scale. Run --dtype=bfloat16 (or --ngram_sparse=False).")
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
    if ngram_optimizer is not None:
        saved = checkpoint.get('ngram_optimizer')
        if saved is None:
            print("note: this checkpoint has no SparseAdam state (it was written before the n-gram "
                  "tables had one, or by a run that used full-table updates). The table rows restart "
                  "with zero moments.")
        else:
            ngram_optimizer.load_state_dict(saved)
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

if count_enabled:
    count_pipelines['train'] = CountDataPipeline(
        count_tables, count_weights, [os.path.join(data_dir, 'train.bin')], block_size, batch_size,
        j=count_j, k=count_k, max_order=(count_orders if count_orders > 0 else None),
        backoff_depth=count_backoff_depth, device=device, score_dtype=ptdtype,
        workers=count_workers, prefetch_depth=count_prefetch_depth, seed=1337 + seed_offset,
        limit_tokens=(count_train_tokens if count_train_tokens > 0 else None))
    # Eval batches get at least one lookup in flight too; the seed is fixed so two runs of the same
    # config score the same windows.
    count_pipelines['val'] = CountDataPipeline(
        count_tables, count_weights, [os.path.join(data_dir, 'val.bin')], block_size, batch_size,
        j=count_j, k=count_k, max_order=(count_orders if count_orders > 0 else None),
        backoff_depth=count_backoff_depth, device=device, score_dtype=ptdtype,
        workers=count_workers, prefetch_depth=max(1, count_prefetch_depth), seed=991)


def fetch_batch(split):
    """(X, Y, count) for one step. count is None unless this is a COUNT run."""
    if count_enabled:
        b = count_pipelines[split].next_batch()
        if b is None:
            raise RuntimeError("count prefetch pipeline ended early")
        return b['idx'], b['targets'], b
    X, Y = get_batch(split)
    return X, Y, None


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, C = fetch_batch(split)
            with ctx:
                logits, loss = model(X, Y, count=C)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler.
# 'cosine' is nanoGPT's: warm up, then a cosine all the way down.
# 'wsd' is the paper's WSD-S: warm up, hold flat, then decay over only the last slice of training.
# Same endpoints, and the flat middle is where COUNT's benefit is cheapest to see.
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    if lr_schedule == 'wsd':
        # 3a) flat at the max lr until the decay window opens
        decay_start = lr_decay_iters - int(wsd_decay_fraction * (lr_decay_iters - warmup_iters))
        if it < decay_start:
            return learning_rate
        # 3b) a linear ramp down through the window. WSD's published shape is linear or cosine over
        # the last ~20%; linear is the one the paper's WSD-S description implies.
        ramp = (it - decay_start) / max(1, lr_decay_iters - decay_start)
        return learning_rate + (min_lr - learning_rate) * ramp
    # 3c) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y, C = fetch_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
count_wait = 0.0 # seconds spent blocked on a lookup that prefetch did not hide
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if ngram_optimizer is not None:
        # Sec. 3.3: the tables train on "the backbone learning-rate schedule", which means this same
        # ramp, warmup included, not a separate one.
        for param_group in ngram_optimizer.param_groups:
            param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    # the sparse tables' row moments. A resume without them is not fatal, but the
                    # tables then restart from Adam's zero-moment bias correction, which is visible.
                    'ngram_optimizer': (ngram_optimizer.state_dict()
                                        if ngram_optimizer is not None else None),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y, count=C)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU.
        # With COUNT this is the whole point of count_prefetch_depth: the next batch's table search
        # (CPU, OpenMP, host RAM) runs while this batch's backward pass runs on the GPU.
        t_fetch = time.time()
        X, Y, C = fetch_batch('train')
        count_wait += time.time() - t_fetch # how much of the prefetch the GPU actually hid
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        # Only the dense gradients: clip_grad_norm_ reduces them with foreach kernels and a sparse COO
        # gradient in the list is a type error. The n-gram rows are clipped by nothing, which is what
        # their row-wise Adam step already is -- a bounded move of the rows this batch retrieved.
        torch.nn.utils.clip_grad_norm_(dense_grad_params(model), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    if ngram_optimizer is not None:
        # outside the scaler: SparseAdam takes fp32 table weights and sparse gradients, and the fp16
        # path that needs unscale_ cannot reach them (train.py refuses dtype=float16 with them)
        ngram_optimizer.step()
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)
    if ngram_optimizer is not None:
        ngram_optimizer.zero_grad(set_to_none=True)

    # timing and logging
    if sync_timing and device_type == 'cuda':
        # Without this, dt stops as soon as the last kernel is *queued*. Whatever the GPU is still
        # executing spills into the next iteration's dt, which is exactly the arm-to-arm comparison
        # throughput claims rest on.
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    if count_enabled:
        # The paper's throughput claim is that lookup is *hidden*: a count_wait near zero with
        # prefetch_depth >= 1 means the host table search overlapped the GPU instead of serializing
        # in front of it. A large count_wait means the lookup is the bottleneck (table too big for
        # the page cache, too few CPU threads, or prefetch_depth too small).
        count_wait_ms = 1000.0 * count_wait / gradient_accumulation_steps
        count_wait = 0.0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt,
                                         peak_flops=peak_tflops * 1e12)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        if count_enabled:
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, "
                  f"mfu {running_mfu*100:.2f}%, count_wait {count_wait_ms:.2f}ms/micro-batch")
        else:
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

# ---------------------------------------------------------------------------
# Where the bytes actually are. The paper's headline claim is a memory claim, so the
# comparison it rests on has to be reported as numbers: COUNT's n-gram information is a
# read-only table of *scores* in host RAM that the GPU never has to hold, while the Sec. 3.3
# learned n-gram baseline stores learned *vectors* whose gradients and Adam moments have to
# live somewhere. Printed per run, because the two host-residency stories have very different
# costs (a score table is read; a learned table is read, written, and carries 2x its own size
# in optimizer state).
def _optimizer_state_bytes(*optimizers):
    total = 0
    for opt in optimizers:
        if opt is None:
            continue
        for state in opt.state.values():
            for value in state.values() if hasattr(state, 'values') else []:
                if torch.is_tensor(value):
                    total += value.numel() * value.element_size()
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    total += 8 # step counters, betas-as-float and friends
    return total

def _param_tensors(module):
    seen, out = set(), []
    for p in module.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            out.append(p)
    return out

def _param_bytes(module):
    seen, total = set(), 0
    for p in module.parameters():
        if id(p) in seen:
            continue # a tied wte/lm_head is one tensor with two names
        seen.add(id(p))
        total += p.numel() * p.element_size()
    return total

def memory_report():
    mib = 1024 * 1024
    param_bytes = _param_bytes(model)
    ngram_bytes = ngram_host_bytes = 0
    if model.ngram is not None:
        for p in model.ngram.table_weights():
            size = p.numel() * p.element_size()
            if p.device.type == 'cuda':
                ngram_bytes += size
            else:
                ngram_host_bytes += size
    opt_bytes = _optimizer_state_bytes(optimizer, ngram_optimizer)
    n_params = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
    print(f"[mem] device: {param_bytes / mib:.1f} MiB parameters "
          f"({n_params / 1e6:.2f}M weights, {len(_param_tensors(model))} tensors), "
          f"{opt_bytes / mib:.1f} MiB optimizer state, "
          f"{ngram_bytes / mib:.1f} MiB learned n-gram tables")
    if device_type == 'cuda':
        print(f"[mem] device peak: {torch.cuda.max_memory_allocated() / mib:.1f} MiB allocated, "
              f"{torch.cuda.max_memory_reserved() / mib:.1f} MiB reserved by the caching allocator")
    host = 0
    if count_enabled:
        host = sum(t.host_bytes() for t in count_tables)
    if host:
        print(f"[mem] host: {host / mib:.1f} MiB of COUNT score tables (mmap, read-only, searched "
              f"by {count_workers} CPU threads; pages the OS may evict)")
    if ngram_host_bytes:
        print(f"[mem] host: {ngram_host_bytes / mib:.1f} MiB of learned n-gram tables (writable: "
              "their retrieved rows, gradients and Adam moments cross PCIe every step)")

memory_report()

if count_enabled:
    for pipeline in count_pipelines.values():
        pipeline.close()
    for t in count_tables:
        t.close()

if ddp:
    destroy_process_group()
