"""
Llama-3-style decoder-only transformer, following the model design choices of
arXiv:2609.12303 (Section A.1: "We stick to the Llama-3 implementation for the
model design choices"):

  * RMSNorm (no bias)
  * RoPE positional embeddings (no learned position embeddings)
  * SwiGLU feed-forward with d_ffn = ceil(4/3 * n_embd / 256) * 256
  * no bias in Linear layers
  * untied token embedding / lm_head (paper Table 9 counts d*V twice)
  * optional QK-norm (paper Section A.4: used to stabilize byte-model runs)

The interface mirrors nanoGPT's GPT (forward / configure_optimizers /
estimate_mfu / generate) so it can be swapped in with minimal changes.
"""

import math
import inspect
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from triton_attn import triton_flash_attention, is_available as _triton_attn_available
except Exception:  # pragma: no cover
    def triton_flash_attention(q, k, v, **kw):
        raise RuntimeError("triton attention unavailable")
    def _triton_attn_available():
        return False


# -----------------------------------------------------------------------------
# config

@dataclass
class LlamaConfig:
    block_size: int = 2048
    vocab_size: int = 260
    n_layer: int = 10
    n_head: int = 10
    n_embd: int = 640
    d_ffn: int = 0          # 0 -> Llama default: ceil(4/3 * n_embd / 256) * 256
    dropout: float = 0.0
    rope_base: float = 500000.0
    qk_norm: bool = False   # RMSNorm on per-head q,k before RoPE (paper A.4)
    tie_weights: bool = False
    checkpointing: bool = True  # activation checkpointing per block (memory)
    attn_chunk: int = 256       # query chunk size for chunked attention


def default_d_ffn(n_embd: int) -> int:
    return max(256, math.ceil(4 / 3 * n_embd / 256) * 256)


# -----------------------------------------------------------------------------
# building blocks

class RMSNorm(nn.Module):
    def __init__(self, ndim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps

    def forward(self, x):
        # compute in fp32 (pow/rsqrt are fp32 ops under autocast; fp32 weight
        # x bf16 input would otherwise promote to fp32), return input dtype
        xf = x.to(torch.float32)
        out = self.weight * xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.to(x.dtype)


def precompute_freqs_cis(dim: int, end: int, base: float = 500000.0) -> torch.Tensor:
    """RoPE frequency table, shape (end, dim // 2), complex64."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """xq, xk: (B, nh, T, hs); freqs_cis: (T, hs // 2) complex."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # (B, nh, T, hs//2)
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[: xq_.shape[2]].unsqueeze(0).unsqueeze(1)  # (1, 1, T, hs//2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def chunked_causal_attention(q, k, v, dropout_p=0.0, training=False, chunk=256):
    """Causal self-attention without flash kernels (this ROCm build has no
    flash/mem-efficient SDPA kernels, and the math backend materializes the
    full (B, nh, T, T) matrix). Processes queries in chunks of `chunk` rows:
    memory O(B*nh*chunk*T) instead of O(B*nh*T^2).
    q, k, v: (B, nh, T, hs). Returns (B, nh, T, hs)."""
    B, nh, T, hs = q.shape
    scale = 1.0 / math.sqrt(hs)
    out = torch.empty_like(q)
    t = torch.arange(T, device=q.device)
    for i in range(0, T, chunk):
        qi = q[:, :, i:i + chunk]                        # (B, nh, C, hs)
        s = (qi @ k.transpose(-2, -1)) * scale           # (B, nh, C, T)
        j = torch.arange(i, i + qi.size(2), device=q.device)
        mask = t[None, :] <= j[:, None]                  # (C, T) causal
        s = s.float()
        s.masked_fill_(~mask[None, None, :], float('-inf'))
        p = torch.softmax(s, dim=-1, out=s)              # reuse buffer
        p = p.type_as(q)
        if training and dropout_p > 0.0:
            p = F.dropout(p, dropout_p, training=True)
        out[:, :, i:i + chunk] = p @ v
    return out


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.qk_norm = config.qk_norm
        self.dropout = config.dropout
        self.attn_chunk = config.attn_chunk
        _d = self.head_dim
        self.use_triton = (_triton_attn_available() and _d & (_d - 1) == 0)
        # key, query, value projections for all heads, in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, freqs_cis):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        # causal self-attention. This ROCm build has no SDPA flash kernels, so
        # use the Triton flash-attention kernels (fwd+bwd) when available,
        # otherwise the torch chunked fallback.
        if self.use_triton:
            y = triton_flash_attention(q.contiguous(), k.contiguous(), v.contiguous())
        else:
            y = chunked_causal_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                training=self.training,
                chunk=self.attn_chunk,
            )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class SwiGLU(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.d_ffn, bias=False)
        self.c_gate = nn.Linear(config.n_embd, config.d_ffn, bias=False)
        self.c_proj = nn.Linear(config.d_ffn, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_proj(self.dropout(F.silu(self.c_gate(x)) * self.c_fc(x)))
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.ln_1(x), freqs_cis)
        x = x + self.mlp(self.ln_2(x))
        return x


# -----------------------------------------------------------------------------
# model

class Llama(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        if config.d_ffn == 0:
            config.d_ffn = default_d_ffn(config.n_embd)
        self.config = config

        head_dim = config.n_embd // config.n_head
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_weights:
            self.transformer.wte.weight = self.lm_head.weight

        # RoPE table (non-persistent buffer)
        self.register_buffer(
            'freqs_cis',
            precompute_freqs_cis(head_dim, config.block_size, config.rope_base),
            persistent=False,
        )

        self.apply(self._init_weights)
        # scaled init for residual projections, per GPT-2 / Llama convention
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """Total parameter count. non_embedding subtracts the token embedding
        (it is not shared with lm_head unless tie_weights is set)."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.config.tie_weights:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        freqs_cis = self.freqs_cis  # (block_size, head_dim // 2)

        x = self.transformer.wte(idx)  # (b, t, n_embd)
        for block in self.transformer.h:
            if self.config.checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, freqs_cis, use_reentrant=False)
            else:
                x = block(x, freqs_cis)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])  # keep the time dim
            loss = None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # 2D params (weight matrices) decay; 1D (norm weights) and embeddings don't
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 and not n.endswith('wte.weight')]
        decay_ids = {id(p) for p in decay_params}
        nodecay_params = [p for p in param_dict.values() if id(p) not in decay_ids]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model FLOPs utilization (A100 bf16 peak, like nanoGPT)."""
        N = self.get_num_params(non_embedding=False)
        cfg = self.config
        L, T = cfg.n_layer, cfg.block_size
        flops_per_token = 6 * N + 12 * L * cfg.n_embd * T
        flops_per_iter = flops_per_token * T * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        return flops_achieved / flops_promised

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx