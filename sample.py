"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT, STOCK_ARCH, NO_COUNT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
# COUNT. Empty count_table means: use whatever the checkpoint says it was trained with (see below).
# A non-empty count_table *overrides* the checkpoint's tables, which is Sec. 5.1's inference-time
# domain swap: same weights, different host-resident n-gram table.
count_table = '' # general table directory
count_domain_table = '' # domain table directory, mixed in at count_lambda
count_lambda = 0.0 # S_lambda = (1-lambda) S_general + lambda S_domain; lambda=1 replaces the table
count_j = 0 # 0 = take J from the checkpoint
count_k = 0 # 0 = take K from the checkpoint
count_orders = 0 # 0 = every order the table stores
count_backoff_depth = -1 # -1 = back off all the way to the unigram row
count_workers = 8
# Late fusion (Appendix B.1): a table prior fused into the next-token distribution at decode time.
# Inference-only, no parameters, and by default it switches Eq. 1's f_t *off* -- the paper's two
# alternatives are the trained f_t and late fusion, not both. '' turns it off.
fusion = '' # 'additive:0.1' = Eq. 5, 'multiplicative:0.1' = Eq. 6/7. The paper fixes 0.1 for both.
fusion_j = 1 # B.1 scores the longest history only, so J=1 even for a checkpoint trained at J=6
fusion_k = 8
fusion_keep_count = False # True runs late fusion on top of a trained f_t. Not a paper arm; measured only
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    # A checkpoint that does not mention the architecture flags was written before they existed,
    # and GPTConfig's defaults no longer describe that block. Fill the gaps with STOCK_ARCH, or
    # load_state_dict below fails with a missing-keys error over a perfectly good checkpoint.
    gptconf = GPTConfig(**{**STOCK_ARCH, **NO_COUNT, **checkpoint['model_args']})
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

# COUNT tables. Resolution order matters: an explicit --count_table always wins (that is the
# domain-swap experiment), otherwise a checkpoint that was trained with COUNT uses the table
# directory recorded in its own config, otherwise the run is a plain transformer.
ckpt_count = {k: checkpoint['model_args'].get(k, NO_COUNT[k]) for k in
              ['count', 'count_j', 'count_k', 'count_norm', 'count_amplify']} if init_from == 'resume' else dict(NO_COUNT)
if init_from == 'resume' and ckpt_count['count'] and not count_table:
    recorded = checkpoint.get('config', {}).get('count_table', '')
    if recorded:
        count_table = recorded
        print(f"checkpoint was trained with COUNT: using its recorded table {count_table}")
    else:
        print("checkpoint was trained with COUNT but records no table path: pass --count_table=<dir>")

count_dirs = [d for d in (count_table, count_domain_table) if d]
if count_dirs:
    from count.table import CountTable, set_threads

    set_threads(count_workers)
    count_tables = [CountTable(d) for d in count_dirs]
    for t in count_tables:
        print(t.describe())
    weights = [1.0 - count_lambda, count_lambda] if len(count_tables) == 2 else [1.0]
    if ckpt_count['count']:
        print(f"COUNT decode: J={count_j or ckpt_count['count_j']} K={count_k or ckpt_count['count_k']}"
              f" lambda={count_lambda} (matching the trained model's J/K unless overridden)")
    else:
        print("WARNING: this checkpoint was trained *without* COUNT. Adding Eq. 1's f_t to a model "
              "that never saw one is legal and adds no parameters, but it is not what Sec. 5.1 does "
              "and the first samples after the switch will look off-distribution.")
    model.attach_count_tables(count_tables, weights=weights,
                              j=(count_j or ckpt_count.get('count_j') or None),
                              k=(count_k or ckpt_count.get('count_k') or None),
                              max_order=(count_orders or None),
                              backoff_depth=count_backoff_depth, score_dtype=ptdtype)

model.eval()
model.to(device)
# Restores host residency for a checkpoint trained with ngram_host_tables=True: model.to(device)
# moves the Sec. 3.3 tables onto the GPU, while their forward() still gathers rows on the CPU.
model.rehost_ngram_tables()
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# Late fusion, applied after any wrapping: torch.compile delegates generate() to the wrapped module,
# so the switch has to live on that module and not on the wrapper.
mode, _, weight = fusion.partition(':')
if mode:
    target = getattr(model, '_orig_mod', model)
    target.set_fusion(mode, weight=float(weight or 0.1), j=fusion_j, k=fusion_k,
                      keep_count=fusion_keep_count)
    print(f"late fusion ({mode}, weight={weight or 0.1}): searching {fusion_j} history slot(s) x "
          f"{fusion_k} candidates at every decoded position; Eq. 1's f_t is "
          f"{'kept' if fusion_keep_count else 'switched off'} for sampling")

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
if load_meta and not ('stoi' in meta and 'itos' in meta):
    # A BPE dataset's meta.pkl carries only vocab_size -- data/fineweb-edu/prepare.py writes exactly
    # that, because its ids come from tiktoken rather than from a per-dataset vocabulary. There is no
    # stoi/itos to decode with, and nanoGPT's original check assumed char/word datasets and KeyError'd
    # here for anyone sampling from a fineweb-edu checkpoint.
    print(f"{meta_path} stores no stoi/itos (this dataset's ids come from tiktoken): "
          f"decoding with the gpt2 encoding instead")
    load_meta = False
if load_meta:
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
