"""
Full definition of a GPT Language Model, all of it in this single file.
Modernized architecture: RoPE positional embeddings, RMSNorm, SwiGLU MLP.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
 https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
 https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
3) RoPE (rotary position embeddings): https://arxiv.org/abs/2104.09864
4) SwiGLU: https://arxiv.org/abs/2002.05202
5) RMSNorm: https://arxiv.org/abs/1910.07467
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """ Root-Mean-Square LayerNorm (https://arxiv.org/abs/1910.07467).
    Unlike LayerNorm there is no mean subtraction: the input is scaled by its
    RMS and a learnable per-channel weight. """

    def __init__(self, ndim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, input):
        return self.weight * input * torch.rsqrt(input.pow(2).mean(-1, keepdim=True) + 1e-5)

# ---------------------------------------------------------------------------
# RoPE (rotary position embeddings)
# ---------------------------------------------------------------------------
def precompute_freqs_cis(dim: int, end: int, base: float = 10000.0):
    """ Precompute the rotary frequency matrix for positions 0..end-1.
    Returns a complex64 tensor of shape (end, dim // 2). """
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, dtype=torch.float)
    freqs = torch.outer(t, freqs)                        # (end, dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    """ Apply rotary embeddings to query/key tensors of shape (..., dim).
    freqs_cis has shape (T, dim // 2) and broadcasts over the leading dims. """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# ---------------------------------------------------------------------------
# Causal self-attention (with RoPE + flash attention)
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        head_dim = config.n_embd // config.n_head
        assert head_dim % 2 == 0, f"head_dim ({head_dim}) must be even for RoPE"
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
        # RoPE: precomputed rotary frequencies, shape (block_size, head_dim // 2) complex.
        # non-persistent: derived, not learned, so it stays out of checkpoints.
        freqs_cis = precompute_freqs_cis(head_dim, config.block_size)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # apply rotary position embeddings to queries and keys (positions 0..T-1)
        freqs_cis = self.freqs_cis[:T].view(1, 1, T, -1)  # (1, 1, T, hs//2) broadcasts over (B, nh, T, hs//2)
        q, k = apply_rotary_emb(q, k, freqs_cis)

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

# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """ SwiGLU feed-forward (https://arxiv.org/abs/2002.05202):
            y = down( SiLU(gate(x)) * up(x) )
    Hidden width is 8/3 * n_embd (rounded up to a multiple of 256), which gives
    the same parameter count as the classic 4 * n_embd GELU MLP. """

    def __init__(self, config):
        super().__init__()
        hidden_dim = int(8 * config.n_embd / 3)
        hidden_dim = ((hidden_dim + 255) // 256) * 256
        self.gate = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.up   = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.down = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.down(self.dropout(F.silu(self.gate(x)) * self.up(x)))
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linear layers, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('down.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        With RoPE there are no position-embedding parameters to subtract, so this
        is just the full count. The token embeddings are tied to lm_head and are
        used as the output weights, so they are included.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load a checkpoint (block size 1024) but want to use a
        # smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        # truncate the precomputed RoPE frequencies and any causal mask buffer
        for block in self.transformer.h:
            block.attn.freqs_cis = block.attn.freqs_cis[:block_size]
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        raise NotImplementedError(
            "from_pretrained loads OpenAI GPT-2 weights, which use learned "
            "position embeddings, GELU MLPs, and LayerNorm. This model uses "
            "RoPE, SwiGLU, and RMSNorm, so GPT-2 checkpoints are architecturally "
            "incompatible. Train from scratch (init_from='scratch')."
        )

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type,
                             optimizer_name='adamw', muon_lr=None):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and norms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        if optimizer_name == 'adamw':
            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
            print(f"using fused AdamW: {use_fused}")
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
            return [optimizer]

        elif optimizer_name == 'muon':
            # Muon (https://kellerjordan.github.io/posts/muon/) orthogonalizes 2D
            # gradient matrices with Newton-Schulz iterations; it is designed for
            # hidden-layer weight matrices. Embeddings, biases and norms are not
            # suited to it, so they stay on AdamW (as recommended by the Muon paper).
            # Weight tying makes wte.weight identical to lm_head.weight: dedup by identity.
            emb_ids = {id(m.weight) for m in self.modules() if isinstance(m, nn.Embedding)}
            seen = set()
            muon_params = [] # 2D hidden-layer weights -> Muon
            adamw_decay_params = [] # 2D embeddings -> AdamW with weight decay
            adamw_nodecay_params = [] # 1D norms/biases -> AdamW without
            for p in param_dict.values():
                if id(p) in seen:
                    continue
                seen.add(id(p))
                if p.dim() >= 2 and id(p) not in emb_ids:
                    muon_params.append(p)
                elif p.dim() >= 2:
                    adamw_decay_params.append(p)
                else:
                    adamw_nodecay_params.append(p)
            num_muon_params = sum(p.numel() for p in muon_params)
            num_adamw_params = sum(p.numel() for p in adamw_decay_params + adamw_nodecay_params)
            print(f"num Muon parameter tensors: {len(muon_params)}, with {num_muon_params:,} parameters")
            print(f"num AdamW parameter tensors: {len(adamw_decay_params) + len(adamw_nodecay_params)}, with {num_adamw_params:,} parameters")
            # momentum 0.95 + Nesterov are the paper's defaults
            muon_optimizer = torch.optim.Muon(
                [{'params': muon_params, 'weight_decay': weight_decay}],
                lr=muon_lr, momentum=0.95, nesterov=True)
            adamw_optimizer = torch.optim.AdamW(
                [{'params': adamw_decay_params, 'weight_decay': weight_decay},
                 {'params': adamw_nodecay_params, 'weight_decay': 0.0}],
                lr=learning_rate, betas=betas, **extra_args)
            print(f"using fused AdamW: {use_fused}")
            return [muon_optimizer, adamw_optimizer]

        else:
            raise ValueError(f"unknown optimizer: {optimizer_name!r} (expected 'adamw' or 'muon')")

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
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
