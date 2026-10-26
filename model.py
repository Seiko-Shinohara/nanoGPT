"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class RMSNorm(nn.Module):
    """ LayerNorm without the mean subtraction (and without bias by default).

    One reduction instead of two, no bias read, same parameter count as LayerNorm(bias=False).
    It is also exactly what QK-normalization needs, which is why both live here.
    F.rms_norm is a composite ATen op (torch >= 2.4), so inductor decomposes and fuses it the
    same way it fuses F.layer_norm.
    """

    def __init__(self, ndim, bias=False, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, input):
        # The weight is a fp32 master copy; F.rms_norm refuses its fused kernel when input and
        # weight dtypes disagree ("Mismatch dtype between input and weight ... Cannot dispatch to
        # fused implementation", layer_norm.cpp) and falls back to an fp32 upcast of the whole
        # activation. On RDNA3 an fp32 elementwise pass is ~26x slower than the bf16 one, so the
        # cast is the point of this line, not a cosmetic detail. It also keeps the saved tensors
        # on the backward path bf16 instead of fp32.
        y = F.rms_norm(input, self.weight.shape, self.weight.to(input.dtype), self.eps)
        if self.bias is not None:
            y = y + self.bias
        return y

def make_norm(ndim, kind, bias):
    """ The single place that decides which normalization a GPT uses. """
    if kind == 'layernorm':
        return LayerNorm(ndim, bias)
    if kind == 'rmsnorm':
        return RMSNorm(ndim, bias)
    raise ValueError(f"norm must be 'layernorm' or 'rmsnorm', got {kind!r}")

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # architecture levers, all defaulting to the stock GPT-2 path. Both touch q and k
        # only, and both run before SDPA, so the flash path and its head_dim assumptions
        # are unaffected. See AMD_NOTES.md "Architecture flags".
        self.head_dim = config.n_embd // config.n_head
        self.qk_norm = config.qk_norm
        if self.qk_norm:
            self.q_norm_weight = nn.Parameter(torch.ones(self.head_dim))
            self.k_norm_weight = nn.Parameter(torch.ones(self.head_dim))
        self.rope = config.rope
        if self.rope:
            assert self.head_dim % 2 == 0, "RoPE rotate-half needs an even head_dim"
            # cos/sin tables built once at init in fp32, sliced to the live T in forward and
            # cast to the activation dtype there (casting the *tables*, not q/k, keeps the
            # attention matmuls in bf16: an fp32 table would promote q and k to fp32, and on
            # this ROCm build fp32 attention is ~26x slower than bf16).
            # persistent=False: these are derived constants, not weights, so they never enter
            # a state_dict and a checkpoint written before RoPE existed still loads.
            inv_freq = 1.0 / (config.rope_base ** (torch.arange(0, self.head_dim, 2,
                             dtype=torch.float32) / self.head_dim)) # (head_dim/2,)
            angles = torch.arange(config.block_size, dtype=torch.float32)[:, None] * inv_freq[None, :]
            self.register_buffer("rope_cos", torch.cat((angles.cos(), angles.cos()), dim=-1), persistent=False)
            self.register_buffer("rope_sin", torch.cat((angles.sin(), angles.sin()), dim=-1), persistent=False)
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    @staticmethod
    def _rotate_half(x):
        h = x.size(-1) // 2
        return torch.cat((-x[..., h:], x[..., :h]), dim=-1)

    def _apply_rope(self, x, start=0):
        """ Rotate q/k by absolute position, rotate-half form (LLaMA-style).

        `start` exists for tools/selfcheck.py: it asks for a specific absolute position so
        the relative-position property can be checked directly against this method. The
        training path always uses start=0.
        """
        t = x.size(-2)
        assert start + t <= self.rope_cos.size(0), (
            f"position {start + t} is past this model's block_size {self.rope_cos.size(0)}")
        cos = self.rope_cos[start:start + t].to(x.dtype) # (t, head_dim), broadcasts over (B, nh)
        sin = self.rope_sin[start:start + t].to(x.dtype)
        return x * cos + self._rotate_half(x) * sin

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # RoPE before QK-norm: rotation is a linear isometry, so normalizing afterwards means
        # the normalized q/k are unchanged by the scale of x. That ordering is also what
        # Llama-3-RMSNorm and Chameleon use.
        if self.rope:
            q = self._apply_rope(q)
            k = self._apply_rope(k)
        if self.qk_norm:
            # same dtype rule as RMSNorm.forward: fp32 weight next to bf16 q/k costs a fused
            # kernel and an fp32 copy of every q/k in the graph
            q = F.rms_norm(q, (q.size(-1),), self.q_norm_weight.to(q.dtype), 1e-6)
            k = F.rms_norm(k, (k.size(-1),), self.k_norm_weight.to(k.dtype), 1e-6)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ffn = config.ffn
        if self.ffn == 'swiglu':
            # 8/3*d rounded to a multiple of 128 is the hidden width that makes SwiGLU's three
            # matrices cost what GELU's two 4x matrices cost. At n_embd=768 that is exactly 2048:
            #   3 * 768 * 2048 == 2 * 768 * 3072 == 4,718,592 parameters and the same FLOPs,
            # and the wide activation is 2048 columns instead of 3072, i.e. less saved for backward.
            hidden = round(8 * config.n_embd / 3 / 128) * 128
            self.w_gate  = nn.Linear(config.n_embd, hidden, bias=config.bias)
            self.w_up    = nn.Linear(config.n_embd, hidden, bias=config.bias)
            self.c_proj  = nn.Linear(hidden, config.n_embd, bias=config.bias)
        elif self.ffn == 'gelu':
            self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.gelu    = nn.GELU()
            self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        else:
            raise ValueError(f"ffn must be 'gelu' or 'swiglu', got {self.ffn!r}")
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        if self.ffn == 'swiglu':
            x = self.c_proj(F.silu(self.w_gate(x)) * self.w_up(x))
        else:
            x = self.c_proj(self.gelu(self.c_fc(x)))
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = make_norm(config.n_embd, config.norm, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = make_norm(config.n_embd, config.norm, config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# The architecture that OpenAI's GPT-2 weights, and every nanoGPT checkpoint written before
# the flags below existed, actually describe. It is a separate constant from GPTConfig's
# defaults on purpose: defaults say what a *new* run gets, STOCK_ARCH says what a *checkpoint*
# without those keys means. Anything guessing at an absent value (train.py's resume guard,
# sample.py, GPT.from_pretrained) must consult this, not the defaults.
STOCK_ARCH = dict(norm='layernorm', ffn='gelu', qk_norm=False, rope=False, rope_base=10000.0,
                  tie_embeddings=True) # the OpenAI checkpoint shares wte with the head

# Same reasoning as STOCK_ARCH, for the COUNT keys: a checkpoint written before COUNT existed, or
# written without --count_table, has no count in its model_args, and what it *means* is a plain
# transformer -- not GPTConfig's defaults. A COUNT checkpoint resumed into a COUNT-free model (or
# with a different J/K) is a silently different network, so train.py checks these too.
NO_COUNT = dict(count=False, count_j=6, count_k=8, count_norm='j', count_amplify=False,
                count_dedup=True)
# The learned n-gram baseline of Sec. 3.3, off by default so every stock nanoGPT run is unchanged.
NO_NGRAM = dict(ngram_orders=0, ngram_tables=4, ngram_rows=0, ngram_width_div=8, ngram_ratio=1.0,
                ngram_sparse=True, ngram_host_tables=False)

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    # Architecture levers. Unlike the two VRAM levers below these change what the network *is*,
    # so they belong in a checkpoint's model_args and train.py refuses to resume a checkpoint
    # whose values differ. These four now default ON: at n_embd=768 they cost no parameters and
    # no matmul FLOPs (see AMD_NOTES.md "Architecture flags"), so the block a plain
    # `python train.py` gets is the one worth training. The stock GPT-2 block is STOCK_ARCH above:
    # GPT.from_pretrained('gpt2') forces it, train.py maps a checkpoint that predates these flags
    # onto it, and config/train_fineweb_edu_amd.py pins it explicitly.
    norm: str = 'rmsnorm' # 'layernorm' | 'rmsnorm'
    ffn: str = 'swiglu' # 'gelu' | 'swiglu' (SwiGLU hidden = 8/3*d rounded to 128: param-matched)
    qk_norm: bool = True # RMSNorm over head_dim on q and k, applied after RoPE, before SDPA
    rope: bool = True # rotary q/k positions instead of learned absolute position embeddings
    rope_base: float = 10000.0 # rotary base, only read when rope=True
    # VRAM levers, both off by default so the stock path is byte-for-byte untouched.
    # See AMD_NOTES.md "Fitting a bigger micro-batch".
    chunked_head_rows: int = 0 # >0: run lm_head + cross_entropy this many rows at a time
    checkpoint: bool = False # recompute each block's activations during backward
    # COUNT (paper Sec. 2, Eq. 1): the input to the first block becomes h_t^(0) = f_t + e_t, where
    # f_t is the continuation scores of a host-resident n-gram table applied to the model's own
    # unembedding rows. The tables are not parameters and never touch the optimizer; they are
    # searched on the CPU by count_native/libcount_lookup.so and handed to forward() as tensors.
    # See README_COUNT.md. These change what the network is, so they go in model_args.
    count: bool = False # True: forward() requires count=... batches; False: the stock transformer
    count_j: int = 6 # J in Eq. 1: how many stored suffix orders are averaged into f_t
    count_k: int = 8 # K in Eq. 1: candidates per order the model sums over
    count_norm: str = 'j' # 'j': the fixed 1/J of Eq. 1. 'nfound': divide by rows that resolved
    count_amplify: bool = False # embedding amplification, sqrt(d) LayerNorm(h_t^(0)) (Appendix B.6)
    count_chunk: int = 1024 # groups (dedup) or positions (naive) folded per step: bounds f_t's transient
    count_dedup: bool = True # fold sum_j sum_k into one gather per unique token (see count_embedding)
    # Eq. 1 weights the *unembedding* rows B[w]. nanoGPT ties wte to lm_head, so here the
    # unembedding and the token-embedding basis are the same matrix -- which is exactly the
    # alternative Appendix B.7 measures against the paper's choice (Figure 18: at 1B the
    # unembedding basis gives lower LM loss, lower task loss and higher accuracy than reusing the
    # input embeddings). A tied COUNT arm therefore cannot show the paper's better basis; this flag
    # unties the head so f_t can be built from rows the input embedding is free to disagree with.
    # It costs vocab_size * n_embd parameters, so an untied arm is not parameter-matched to a tied
    # one, and it is off by default so every stock run stays stock.
    tie_embeddings: bool = True # False: lm_head gets its own matrix, and B is that matrix
    # The baseline COUNT is measured against (Sec. 3.3, Eq. 3): learned n-gram embeddings, as Over-Encoding
    # implements them and LongCat deploys them. Where COUNT's n-gram information is a read-only table of
    # scores built offline, here each hashed n-gram indexes a *learned* row of width d/8 that is projected
    # to d and added to the token embedding: h_t^(0) = e_t + sum_{n=2..NL} sum_{r=1..R} P_{n,r} T_{n,r}[key].
    # These are parameters. They cost table memory, optimizer state, and a re-learn per run -- the exact
    # costs COUNT's design is arguing against, which is why they are off by default here too.
    ngram_orders: int = 0 # 0 = off; N >= 2: learn tables for orders 2..N (the paper uses NL=3)
    ngram_tables: int = 4 # R hash tables per order
    ngram_rows: int = 0 # rows per table M_{n,r}; 0 = derive from ngram_ratio at model build
    ngram_ratio: float = 1.0 # Sec. 3.3's 1:1 input-embedding-to-backbone parameter rule
    ngram_width_div: int = 8 # table row width d_s = n_embd / 8, projected back up to n_embd
    ngram_sparse: bool = True # sparse gradients: only retrieved rows are updated (Sec. 3.3)
    ngram_host_tables: bool = False # keep the tables in host RAM, ship only retrieved rows to the GPU

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # RoPE replaces learned absolute positions outright rather than adding on top of
            # them: keeping wpe would leave block_size*n_embd parameters in the optimizer state
            # that receive no gradient but still get weight decay.
            wpe = None if config.rope else nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = make_norm(config.n_embd, config.norm, config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # Tied unless this run asks for a separate head. With tying, Eq. 1's basis B *is* the input
        # embedding matrix, i.e. the token-embedding basis Figure 18 measures against the paper's
        # unembedding basis; --tie_embeddings=False is how a COUNT arm gets the basis the paper uses.
        if config.tie_embeddings:
            self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # COUNT's only learned weights, and only when amplification is asked for. Eq. 1 itself
        # adds nothing trainable: f_t reuses the tied unembedding rows above as the basis B.
        # Appendix B.6 amplification. `LayerNorm(..., True)` even when config.bias is False: the
        # paper's LayerNorm "normalizes across the hidden features with learned gain and bias", and
        # Table 2's weight-decay exemption names "embedding-amplification parameters" -- parameters
        # that would not exist if this normalization were affine-free like the rest of a bias=False
        # model. train.py keeps them out of weight decay alongside the input embeddings.
        self.count_amp = LayerNorm(config.n_embd, True) if config.count_amplify else None
        # Which (J, K) this model was *fit* against, fixed here at construction. attach_count_tables
        # rewrites config.count_j/count_k to whatever shape is being run now, so an inference sweep has
        # to compare each arm against the value training used -- not against whatever the previous arm
        # in the sweep asked for, and not against a config some caller flipped in between. None means
        # this network was built with no f_t on its input at all.
        self._trained_jk = (int(config.count_j), int(config.count_k)) if config.count else None

        # Set by attach_count_tables(): a callable that turns a (b, t) window into the count=...
        # batch. generate() needs it because the decode loop re-runs the forward pass over the whole
        # window each step, so every position needs its f_t again, not just the newest one.
        self._count_lookup = None

        # The Sec. 3.3 baseline, if this run asked for one. Built last because ngram_rows=0 means
        # "derive the table size from the 1:1 parameter rule", which needs the rest of the model to
        # exist first. Unlike COUNT, this is a parameter: it is in the state dict, in the optimizer,
        # and in the checkpoint's model_args.
        from count_ngram.embed import ngram_from_config
        self.ngram = ngram_from_config(
            config,
            other_params=sum(p.numel() for p in self.parameters()),
            input_embedding_params=self.transformer.wte.weight.numel(),
        )
        if self.ngram is not None:
            # Set False to score the backbone without q_t. That ablation is what makes the Sec. 3.3
            # baseline comparable to COUNT's: withdrawing f_t from a COUNT checkpoint and withdrawing
            # q_t from a learned-table checkpoint are the same experiment on the two alternatives.
            self.ngram_enabled = True
            pc = self.ngram.param_counts()
            print(f"learned n-grams (Eq. 3): orders {self.ngram.orders}, R={self.ngram.n_tables} per "
                  f"order, {self.ngram.table_rows():,} rows total of width {self.ngram.width} "
                  f"= {pc['table_params']:,} table params + {pc['projection_params']:,} projection "
                  f"params; tables {'on host' if config.ngram_host_tables else 'on device'}, "
                  f"{'retrieved-row' if config.ngram_sparse else 'full-table'} updates")

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.transformer.wpe is not None:
            n_params -= self.transformer.wpe.weight.numel()
        if non_embedding and not self.config.tie_embeddings:
            # The docstring above subtracts the token embeddings too, and only skips doing so here
            # because tying makes one matrix serve both ends. Untied, the input embedding is not a
            # factor of the output matmul, so it is an embedding by that same definition -- and this
            # is the number the Sec. 3.3 1:1 rule and estimate_mfu should be reading.
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_embedding(self, candidates, scores, n_found=None):
        """Eq. 1: f_t = (1/J) sum_j sum_k S(w_k | h_{t,j}) B[w_k], with B the unembedding matrix.

        candidates: (b, t, J, K) long token ids, -1 where a stored row had fewer than K candidates
        scores:     (b, t, J, K) float scores, 0.0 exactly where candidates is -1
        n_found:    (b, t) int32 rows that resolved, only read when count_norm == 'nfound'

        B[w] does not depend on j, so Eq. 1's double sum factors: a token that the J suffix orders
        propose more than once -- and most of them do, because a backoff chain re-raises the same
        continuations at every order -- only needs its embedding row gathered once, with its scores
        added first:

            sum_j sum_k S(w_k | h_j) B[w_k]  ==  sum_w ( sum over slots holding w of S ) B[w]

        count_dedup=True is exactly that rearrangement, not an approximation: the difference from
        summing slot by slot is the order of floating-point additions. It is worth doing because
        J*K gathers per position is where COUNT's GPU time actually goes; tools/bench_count_fold.py
        measures both paths at the training shapes, and tools/test_count_model.py checks that they
        agree.

        The naive path (count_dedup=False) is kept because it is what Eq. 1 reads as written.

        Padding needs no mask in either path: a padded slot carries score 0.0 and its token id is
        clamped to 0, so it contributes an exact zero, and no data-dependent boolean indexing (which
        would synchronize the device on every call) is needed. Accumulation is fp32: J*K
        score-weighted rows of an embedding whose std is 0.02 is exactly the kind of sum bf16 rounds
        badly. index_add_ is nondeterministic in fp32 add order, which is the price of the fold.
        """
        b, t, j, k = candidates.shape
        weight = self.lm_head.weight
        d = weight.size(1)
        if self.config.count_dedup and weight.size(0) > 0x10000:
            # the (row << 16) | token key has a 16-bit token field, the same one count_common.h
            # gives pair keys. A wider vocab needs COUNT_KEY_BITS widened in both places, not here.
            raise RuntimeError(
                f"count_dedup packs token ids into 16 bits, so vocab {weight.size(0)} does not fit: "
                "set count_dedup=False, or widen COUNT_KEY_BITS in count_native/count_common.h and "
                "the key packing here together")
        rows, jk = b * t, j * k
        cands = candidates.reshape(rows, jk)
        sc = scores.reshape(rows, jk).to(torch.float32)
        out = torch.zeros(rows, d, device=weight.device, dtype=torch.float32)

        if self.config.count_dedup:
            # (row << 16) | token: the same 16-bit token field count_common.h uses for a pair key,
            # so the two share one stated ceiling (vocab <= 65536). Sorting (row, token) puts every
            # position's duplicate proposals next to each other; unique_consecutive turns them into
            # one group each, and index_add_ sums the group's scores before any embedding row is read.
            # arange gets unsqueeze(1) so it broadcasts down the (rows, jk) slot axis: the row index
            # is the high field of every one of that position's jk keys, not a value paired with
            # slot 0 only.
            keys = ((torch.arange(rows, device=weight.device, dtype=torch.int64).unsqueeze(1) << 16)
                    | cands.clamp(min=0).to(torch.int64)).reshape(-1)
            order = torch.argsort(keys)
            grouped = keys[order]
            _, inverse, counts = torch.unique_consecutive(grouped, return_inverse=True,
                                                          return_counts=True)
            n_groups = counts.numel()
            starts = torch.cumsum(counts, 0) - counts     # first slot of each group
            tokens = grouped[starts] & 0xFFFF             # one token per group
            which_row = grouped[starts] >> 16             # the position it belongs to
            grouped_scores = torch.zeros(n_groups, device=weight.device, dtype=torch.float32)
            grouped_scores.index_add_(0, inverse, sc.reshape(-1)[order])
            chunk = max(1, min(int(self.config.count_chunk), n_groups))
            index = torch.arange(chunk, device=weight.device)
            for start in range(0, n_groups, chunk):
                stop = min(start + chunk, n_groups)
                gathered = weight.index_select(0, tokens[start:stop]).to(torch.float32)
                out.index_add_(0, which_row[start:stop],
                               gathered * grouped_scores[start:stop].reshape(-1, 1))
        else:
            chunk = max(1, min(int(self.config.count_chunk), rows))
            index = torch.arange(chunk, device=weight.device).repeat_interleave(jk)
            for start in range(0, rows, chunk):
                tok = cands[start:start + chunk].clamp(min=0).reshape(-1)
                rows_in_chunk = index if start + chunk == rows else torch.arange(
                    tok.numel() // jk, device=weight.device).repeat_interleave(jk)
                gathered = weight.index_select(0, tok).to(torch.float32)
                out.index_add_(0, rows_in_chunk, gathered * sc[start:start + chunk].reshape(-1, 1))

        if self.config.count_norm == 'nfound' and n_found is not None:
            scale = 1.0 / n_found.reshape(rows, 1).clamp(min=1).to(torch.float32)
        else:
            scale = 1.0 / float(self.config.count_j) # Eq. 1 divides by J, resolved rows or not
        return (out * scale).to(weight.dtype).view(b, t, d)

    def forward_features(self, idx, count=None):
        """Everything up to (but not including) the lm_head: Eq. 1's fold, the block stack, ln_f.

        Split out because the output-side experiments in the paper (Appendix B.1's late fusion,
        eval_count-style next-token scoring) need the *whole* logit sequence, not the single position
        forward() computes when it is not given targets. Keeping it a separate method is also what
        lets eval_lm.py run the head in its own row chunks without a second copy of this code.
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        if self.config.count and count is None:
            # A model trained with COUNT and run without it silently scores a different network.
            raise RuntimeError(
                "this model has count=True but forward() got no count=... batch: build one with "
                "count.data.CountDataPipeline (training/eval) or count.generate.count_batch (sampling)")
        if count is not None:
            cj, ck = count['candidates'].shape[2], count['candidates'].shape[3]
            if (cj, ck) != (self.config.count_j, self.config.count_k):
                raise RuntimeError(
                    f"count batch has J={cj}, K={ck} but this model was built for "
                    f"J={self.config.count_j}, K={self.config.count_k}. J and K are training "
                    f"decisions: at inference they change f_t, so they have to match the checkpoint "
                    "(attach_count_tables(..., allow_jk_override=True) is how you ask for an "
                    "Appendix B.4 sweep instead).")

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        combined = tok_emb
        if count is not None:
            # h_t^(0) = f_t + e_t. Order of the two additions is the paper's.
            combined = tok_emb + self.count_embedding(
                count['candidates'], count['scores'], count.get('n_found'))
        if self.ngram is not None and getattr(self, 'ngram_enabled', True):
            # Eq. 3: q_t = sum_n sum_r P_{n,r} T_{n,r}[key_{n,r}(t)], added to the same h_t^(0) that
            # Eq. 1's f_t is added to. COUNT and this baseline are alternatives in the paper -- one
            # reads a score table, the other trains a vector table -- but the model keeps both terms
            # separate so an arm that uses both can be measured, which the paper does not do.
            combined = combined + self.ngram(idx)
        if self.count_amp is not None:
            # Embedding amplification, sqrt(d) LayerNorm(.), applied to h_t^(0) = f_t + e_t itself
            # (Appendix B.6, and LongCat's h_t = e_t + q_t), *before* position enters the stream.
            # With rope=True there is no wpe to add, so the two orders coincide; with learned
            # position embeddings they do not, and this order is the one the papers write.
            combined = math.sqrt(self.config.n_embd) * self.count_amp(combined)
        if self.transformer.wpe is not None:
            pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
            combined = combined + self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(combined) # rope: position lives in the attention, not here
        for block in self.transformer.h:
            if self.config.checkpoint and self.training:
                # throw away this block's saved activations, recompute them in backward
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.transformer.ln_f(x)
        return x

    def forward(self, idx, targets=None, count=None):
        x = self.forward_features(idx, count)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            chunk = self.config.chunked_head_rows
            if chunk > 0:
                # Mathematically the same loss as the two lines below: verified on CPU against
                # this model -- gradients bit-identical (1e-8 with `checkpoint` too, from the
                # recomputation), loss identical (5e-7 relative with `checkpoint`).
                # What it saves is subtler than "the logits", though. cross_entropy stashes
                # its log_softmax for backward, not the logits, so the saved-for-backward
                # total is the same either way; chunking shrinks the transient (the fp32
                # upcast of the logits plus the log_softmax) from all b*t rows at once to
                # `chunk` rows at a time. Measured on CPU at 4096x20000 fp32: forward peak
                # 656 -> 392 MiB. Getting the logits tensor back as well is what the
                # `checkpoint` lever above does, not this one.
                # logits comes back as None: the training loop discards it (train.py:311).
                flat_x = x.reshape(-1, x.size(-1))
                flat_t = targets.reshape(-1)
                total = flat_x.new_zeros((), dtype=torch.float32)
                count = total.detach().clone() # tensor, so no .item() sync per chunk
                for start in range(0, flat_x.size(0), chunk):
                    xc = flat_x[start:start + chunk]
                    tc = flat_t[start:start + chunk]
                    total = total + F.cross_entropy(self.lm_head(xc), tc, ignore_index=-1,
                                                    reduction='sum')
                    count = count + (tc != -1).sum()
                logits, loss = None, total / count # 0/0 -> nan, same as F.cross_entropy
            else:
                logits = self.lm_head(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if self.transformer.wpe is not None:
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
            if hasattr(block.attn, 'rope_cos'): # non-persistent buffers, cropped by name (model.py:88)
                block.attn.rope_cos = block.attn.rope_cos[:block_size]
                block.attn.rope_sin = block.attn.rope_sin[:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # The OpenAI checkpoint *is* a stock block: LayerNorms, a 4x GELU MLP, learned absolute
        # positions. There is no SwiGLU / QK-norm / rotary weight anywhere in it, so loading it
        # into a model built from GPTConfig's current defaults would trip the key-count assert
        # below with a message about mismatched keys. Pin the architecture instead.
        config_args.update(STOCK_ARCH)
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type,
                             no_decay_names=()):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # The n-gram table rows, when they update sparsely, do not belong in AdamW at all: with
        # nn.Embedding(sparse=True) their gradients arrive as sparse COO tensors and AdamW's foreach
        # kernels reject them. Sec. 3.3 asks for row-wise AdamW at zero weight decay, and AdamW at zero
        # decay *is* Adam, so those rows go to SparseAdam instead (configure_ngram_optimizer below) on
        # the same schedule. Tables configured ngram_sparse=False are dense gradients and stay here.
        if self.ngram is not None and self.ngram.sparse:
            sparse_ids = {id(p) for p in self.ngram.table_weights()}
            param_dict = {pn: p for pn, p in param_dict.items() if id(p) not in sparse_ids}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        # no_decay_names pulls named 2D parameters out of the decayed group as well: Table 2 of the
        # COUNT paper decays everything at 0.1 *except* input embeddings and the
        # embedding-amplification parameters. Because nanoGPT ties wte to lm_head, exempting
        # 'transformer.wte.weight' exempts the tied tensor once -- named_parameters() reports a
        # tied weight under one name only, so the output head skips decay too. That is the honest
        # reading of the paper's rule for a tied model; --count_no_decay_embeddings=False restores
        # the stock behaviour for runs that want it.
        skip = tuple(no_decay_names)
        def decayed(name, param):
            if param.dim() < 2:
                return False
            return not (skip and name.startswith(skip))
        decay_params = [p for n, p in param_dict.items() if decayed(n, p)]
        nodecay_params = [p for n, p in param_dict.items() if not decayed(n, p)]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def configure_ngram_optimizer(self, learning_rate, betas):
        """The second optimizer a learned-n-gram run needs, or None.

        SparseAdam over the retrieved rows of T_{n,r}: Sec. 3.3's "sparse optimizer updates for only
        the retrieved embedding rows", on the backbone's learning-rate schedule and betas, with no
        weight decay (SparseAdam has no weight decay at all, which is what Table 2 asks these tables to
        have anyway). train.py keeps it on the same get_lr() ramp and stores its state in the checkpoint.
        """
        from count_ngram.optim import build_sparse_optimizer
        return build_sparse_optimizer(self, learning_rate, betas)

    def estimate_mfu(self, fwdbwd_per_iter, dt, peak_flops=312e12):
        """ estimate model flops utilization (MFU) in units of the given GPU's peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as a ratio of the GPU's practical peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = peak_flops # e.g. 312e12 for an A100; see train.py peak_tflops
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def rehost_ngram_tables(self):
        """Re-apply a checkpoint's host residency after any model.to(device).

        nn.Module.to() moves *every* parameter, including the Sec. 3.3 tables this model was
        configured to keep off the GPU, while LearnedNgramEmbeddings.forward() still gathers those
        rows on the host: the result is a device mismatch, not a slow path. train.py calls
        to_host() itself after model.to(device); the eval and sampling scripts need the same three
        lines and should not have to remember that nn.Module.to is recursive.
        """
        ngram = getattr(self, 'ngram', None)
        if ngram is not None and ngram.host_tables:
            ngram.to_host()
        return self

    def attach_count_tables(self, tables, weights=None, j=None, k=None, max_order=None,
                            backoff_depth=-1, score_dtype=None, allow_jk_override=False):
        """Give this model host-resident COUNT tables to search while it decodes.

        Two very different things both look like this call, and they are not the same experiment:

        * A checkpoint trained with COUNT (config.count already True) resumed with its own tables.
          j/k/max_order must then match what was trained -- forward() enforces J and K.
        * Tables bolted onto a model that was trained without them. That is allowed and adds no
          parameters, but the network never saw an f_t term on its input, so the first tokens it
          emits after the switch will look off-distribution. The paper's inference-time adaptation
          (Sec. 5.1) swaps tables inside an already-COUNT model; it does not add COUNT to a
          standard transformer at inference.

        allow_jk_override is the Appendix B.4 ablation: run a J=6 K=8 checkpoint at K=32 or J=1 and
        report what happens to LM loss and task loss. It changes f_t, so it is refused by default
        and announced when granted -- a sweep result is only meaningful if the reader knows the
        checkpoint was not trained at that shape.
        """
        from count.generate import count_batch

        # The shape this model was fit against, recorded by GPT.__init__ from the config it was built
        # with. attach_count_tables necessarily rewrites config.count_j/count_k for the shape being run
        # now, so without this a sweep's second arm would compare itself against its first arm and
        # report a mismatch that never happened at training time.
        trained = getattr(self, '_trained_jk', None)

        jj = int(j if j is not None else self.config.count_j)
        kk = int(k if k is not None else self.config.count_k)
        w = list(weights) if weights is not None else [1.0] * len(tables)
        if len(w) != len(tables):
            raise ValueError("weights must match tables")
        if trained is not None and (jj, kk) != trained:
            if not allow_jk_override:
                raise RuntimeError(
                    f"this checkpoint was trained at J={trained[0]} K={trained[1]} but the tables are "
                    f"being attached at J={jj} K={kk}. Pass allow_jk_override=True to run that as an "
                    "inference-time ablation (Appendix B.4).")
            print(f"count shape override (Appendix B.4): trained J={trained[0]} "
                  f"K={trained[1]} -> running at J={jj} K={kk}. f_t is a different "
                  "tensor than the one this checkpoint's weights were fit against.")
        elif trained is None and not getattr(self, '_warned_no_count', False):
            # A table bolted onto a network that never had an f_t on its input. Legitimate to measure
            # -- it is how the inference-only cost of COUNT gets compared to a standard transformer --
            # but it is not Sec. 5.1, which swaps tables inside an already-COUNT model.
            self._warned_no_count = True
            print("COUNT is being attached to a checkpoint that was trained without it: this network "
                  "never saw an f_t term, so its output after the switch is off-distribution.")
        if not self.config.count:
            self.config.count = True
        self.config.count_j = jj
        self.config.count_k = kk
        lookup_kwargs = dict(
            max_order=(int(max_order) if max_order else None),
            backoff_depth=int(backoff_depth), score_dtype=score_dtype,
        )
        self._count_tables, self._count_weights = list(tables), list(w)
        self._count_lookup_kwargs = lookup_kwargs
        self._count_lookup = lambda idx: count_batch(tables, w, idx, j=jj, k=kk, **lookup_kwargs)
        # Late fusion (Appendix B.1) reads J=1 -- the longest history only -- even in a checkpoint
        # whose f_t was trained at J=6. One set of tables, two lookup shapes, so the shape is a
        # parameter here instead of attaching the same tables twice.
        self._count_lookup_at = lambda idx, j, k: count_batch(tables, w, idx, j=j, k=k, **lookup_kwargs)
        return self._count_lookup

    def set_fusion(self, mode, weight=0.1, j=1, k=8, keep_count=False):
        """Ask generate() to fuse a table prior into its next-token distribution (Appendix B.1).

        Inference-only, exactly as the paper uses it: no parameters, nothing trained, and h_t^(0) is
        left alone. B.1 reads the longest history only (J=1) with K=8 candidates, and fixes lambda=0.1
        for Eq. 5 and alpha=0.1 for Eq. 6/7, which is why the weight has a default.

        keep_count=False -- the default, and the paper's configuration -- switches config.count off, so
        what gets fused is the *standard transformer's* distribution. Eq. 1's trained f_t and B.1's late
        fusion are the paper's two alternatives; running both at once is neither, and eval_lm.py never
        does. Set it True only to measure that combination anyway.
        """
        if mode in (None, "", "none"):
            self._fusion = None
            return self
        if mode not in ("additive", "multiplicative"):
            raise ValueError(f"unknown fusion mode {mode!r}: expected none|additive|multiplicative")
        if getattr(self, '_count_lookup_at', None) is None:
            raise RuntimeError(
                "late fusion needs host-resident tables attached first: call attach_count_tables(...) "
                "before set_fusion(mode, weight)")
        self._fusion = dict(mode=mode, weight=float(weight), j=int(j), k=int(k))
        if not keep_count:
            self.config.count = False
        return self

    @torch.no_grad()
    def logits_rows(self, idx, count=None, row_chunk=1024):
        """log_softmax over the whole (b, t) window, one row chunk at a time.

        forward() returns logits only for the last position when targets is None, and returns no
        logits at all when chunked_head_rows is on, so the output-side experiments build their own
        view of the head. Yields (start, stop, logp) with logp in fp32; the chunk bound is what
        keeps a 50304-wide vocab affordable (1024 rows is 206 MB of fp32).
        """
        x = self.forward_features(idx, count)
        b, t, d = x.shape
        flat = x.reshape(-1, d)
        n_rows = flat.size(0)
        for start in range(0, n_rows, row_chunk):
            stop = min(start + int(row_chunk), n_rows)
            yield start, stop, F.log_softmax(self.lm_head(flat[start:stop]).float(), dim=-1)

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence.
            # With COUNT the lookup runs over the whole cropped window each step, so the f_t that
            # each position got during training is the f_t it gets here too.
            count = None
            if self.config.count:
                if self._count_lookup is None:
                    raise RuntimeError(
                        "this model has count=True but no tables are attached: call "
                        "model.attach_count_tables([...]) before generate()")
                count = self._count_lookup(idx_cond)
            logits, _ = self(idx_cond, count=count)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            fusion = getattr(self, '_fusion', None)
            if fusion is not None:
                # Appendix B.1, applied at the one place it can be applied while decoding: after the
                # model's own distribution exists, before anything truncates it. Candidates come from
                # slot 0 -- the longest history available, h_{t,1} -- which is what B.1 scores, and
                # fusing before top_k matters because lifting candidates the model had cropped away is
                # the whole point of the arm. fuse_logits returns log-probabilities, so the softmax
                # below is exact rather than an approximation of a renormalised distribution.
                from count.late_fusion import fuse_logits
                res = self._count_lookup_at(idx_cond, fusion['j'], fusion['k'])
                cand = res['candidates'][:, -1, 0, :] # (b, K) for the position being predicted
                score = res['scores'][:, -1, 0, :]
                logits = fuse_logits(torch.log_softmax(logits, dim=-1), cand, score,
                                     fusion['mode'], fusion['weight'])
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
