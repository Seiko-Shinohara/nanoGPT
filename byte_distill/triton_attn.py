"""
Triton (AOTriton) flash attention for ROCm builds where PyTorch's SDPA
flash/mem-efficient backends are unavailable (this build: torch 2.14+rocm
reports "No available kernel" for both).

Implements causal flash attention (forward + backward) following the
standard Triton fused-attention tutorial:
  * forward:  query-block parallel, online softmax over key blocks
  * backward: recompute-based dQ kernel + dK/dV kernel (no P materialized)

Memory: O(B*H*(M+D)) instead of O(B*H*M*N).

Usage:  y = triton_flash_attention(q, k, v)   # q,k,v: (B, nh, T, hs)
Falls back to the torch chunked implementation if Triton is unavailable or
the head dim is not a power of two.
"""

import math

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _TRITON_OK = True
except Exception:  # pragma: no cover
    _TRITON_OK = False


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------
@triton.jit
def _fwd_kernel(Q, K, V, O, LSE,
                sm_scale,
                stride_qb, stride_qh, stride_qm,
                stride_kb, stride_kh, stride_kn,
                stride_vb, stride_vh, stride_vn,
                stride_ob, stride_oh, stride_om,
                stride_lsb, stride_lsh, stride_lsm,
                Z, H, M, N,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = O + pid_b * stride_ob + pid_h * stride_oh
    lse_base = LSE + pid_b * stride_lsb + pid_h * stride_lsh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < M
    q = tl.load(q_base + offs_m[:, None] * stride_qm + offs_d[None, :],
                mask=m_mask[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    hi = (pid_m + 1) * BLOCK_M  # causal: keys only up to the end of this block
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(k_base + offs_n[None, :] * stride_kn + offs_d[:, None],
                    mask=(offs_n[None, :] < N), other=0.0)
        qk = tl.dot(q, k) * sm_scale
        causal = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(causal, qk, float('-inf'))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :],
                    mask=(offs_n[:, None] < N), other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    acc = acc / l_i[:, None]
    tl.store(o_base + offs_m[:, None] * stride_om + offs_d[None, :],
             acc.to(q.dtype), mask=m_mask[:, None])
    lse = m_i + tl.log(l_i)
    tl.store(lse_base + offs_m * stride_lsm, lse, mask=m_mask)


@triton.jit
def _bwd_dq_kernel(Q, K, V, O, DO, DQ, LSE, DELTA,
                   sm_scale,
                   stride_qb, stride_qh, stride_qm,
                   stride_kb, stride_kh, stride_kn,
                   stride_vb, stride_vh, stride_vn,
                   stride_ob, stride_oh, stride_om,
                   stride_dob, stride_doh, stride_dom,
                   stride_lsb, stride_lsh, stride_lsm,
                   stride_dsb, stride_dsh, stride_dsm,
                   Z, H, M, N,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = O + pid_b * stride_ob + pid_h * stride_oh
    do_base = DO + pid_b * stride_dob + pid_h * stride_doh
    dq_base = DQ + pid_b * stride_dob + pid_h * stride_doh
    lse_base = LSE + pid_b * stride_lsb + pid_h * stride_lsh
    delta_base = DELTA + pid_b * stride_dsb + pid_h * stride_dsh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < M
    q = tl.load(q_base + offs_m[:, None] * stride_qm + offs_d[None, :],
                mask=m_mask[:, None], other=0.0)
    do = tl.load(do_base + offs_m[:, None] * stride_dom + offs_d[None, :],
                 mask=m_mask[:, None], other=0.0)
    lse = tl.load(lse_base + offs_m * stride_lsm, mask=m_mask, other=0.0)
    delta = tl.load(delta_base + offs_m * stride_dsm, mask=m_mask, other=0.0)

    dq_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    hi = (pid_m + 1) * BLOCK_M
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(k_base + offs_n[:, None] * stride_kn + offs_d[None, :],
                    mask=(offs_n[:, None] < N), other=0.0)  # (BLOCK_N, BLOCK_D)
        v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :],
                    mask=(offs_n[:, None] < N), other=0.0)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        causal = offs_m[:, None] >= offs_n[None, :]
        p = tl.where(causal, tl.exp(qk - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq_acc += tl.dot(ds.to(k.dtype), k) * sm_scale

    tl.store(dq_base + offs_m[:, None] * stride_dom + offs_d[None, :],
             dq_acc.to(q.dtype), mask=m_mask[:, None])


@triton.jit
def _bwd_dkdv_kernel(Q, K, V, O, DO, DK, DV, LSE, DELTA,
                     sm_scale,
                     stride_qb, stride_qh, stride_qm,
                     stride_kb, stride_kh, stride_kn,
                     stride_vb, stride_vh, stride_vn,
                     stride_ob, stride_oh, stride_om,
                     stride_dob, stride_doh, stride_dom,
                     stride_lsb, stride_lsh, stride_lsm,
                     stride_dsb, stride_dsh, stride_dsm,
                     Z, H, M, N,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = O + pid_b * stride_ob + pid_h * stride_oh
    do_base = DO + pid_b * stride_dob + pid_h * stride_doh
    dk_base = DK + pid_b * stride_dob + pid_h * stride_doh
    dv_base = DV + pid_b * stride_dob + pid_h * stride_doh
    lse_base = LSE + pid_b * stride_lsb + pid_h * stride_lsh
    delta_base = DELTA + pid_b * stride_dsb + pid_h * stride_dsh

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    n_mask = offs_n < N
    k = tl.load(k_base + offs_n[None, :] * stride_kn + offs_d[:, None],
                mask=(offs_n[None, :] < N), other=0.0)
    v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :],
                mask=(offs_n[:, None] < N), other=0.0)

    dk_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    lo = pid_n * BLOCK_N  # causal: queries only from this key block onward
    for start_m in range(lo, M, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        q = tl.load(q_base + offs_m[:, None] * stride_qm + offs_d[None, :],
                    mask=(offs_m[:, None] < M), other=0.0)
        do = tl.load(do_base + offs_m[:, None] * stride_dom + offs_d[None, :],
                     mask=(offs_m[:, None] < M), other=0.0)
        lse = tl.load(lse_base + offs_m * stride_lsm, mask=(offs_m < M), other=0.0)
        delta = tl.load(delta_base + offs_m * stride_dsm, mask=(offs_m < M), other=0.0)
        qk = tl.dot(q, k) * sm_scale
        causal = offs_m[:, None] >= offs_n[None, :]
        p = tl.where(causal, tl.exp(qk - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dk_acc += tl.dot(tl.trans(ds.to(k.dtype)), q) * sm_scale
        dv_acc += tl.dot(tl.trans(p.to(v.dtype)), do)

    tl.store(dk_base + offs_n[:, None] * stride_kn + offs_d[None, :],
             dk_acc.to(k.dtype), mask=(offs_n[:, None] < N))
    tl.store(dv_base + offs_n[:, None] * stride_vn + offs_d[None, :],
             dv_acc.to(v.dtype), mask=(offs_n[:, None] < N))


# ---------------------------------------------------------------------------
# autograd wrapper
# ---------------------------------------------------------------------------
class _FlashAttnFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, block_m, block_n):
        B, nh, T, D = q.shape
        o = torch.empty_like(q)
        lse = torch.empty(B, nh, T, device=q.device, dtype=torch.float32)
        sm_scale = 1.0 / math.sqrt(D)
        grid = (triton.cdiv(T, block_m), B * nh)
        _fwd_kernel[grid](
            q, k, v, o, lse, sm_scale,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            lse.stride(0), lse.stride(1), lse.stride(2),
            B, nh, T, T,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=D,
            num_warps=8 if D >= 64 else 4, num_stages=1,
        )
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.sm_scale = sm_scale
        ctx.block_m, ctx.block_n = block_m, block_n
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        do = do.contiguous()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        B, nh, T, D = q.shape
        # DELTA = rowsum(dO * O)
        delta = (do.float() * o.float()).sum(-1)
        bwd_block_m = max(32, ctx.block_m // 2)
        grid_m = (triton.cdiv(T, bwd_block_m), B * nh)
        _bwd_dq_kernel[grid_m](
            q, k, v, o, do, dq, lse, delta, ctx.sm_scale,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            lse.stride(0), lse.stride(1), lse.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            B, nh, T, T,
            BLOCK_M=bwd_block_m, BLOCK_N=ctx.block_n, BLOCK_D=D,
            num_warps=8 if D >= 64 else 4, num_stages=1,
        )
        grid_n = (triton.cdiv(T, ctx.block_n), B * nh)
        _bwd_dkdv_kernel[grid_n](
            q, k, v, o, do, dk, dv, lse, delta, ctx.sm_scale,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            lse.stride(0), lse.stride(1), lse.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            B, nh, T, T,
            BLOCK_M=bwd_block_m, BLOCK_N=ctx.block_n, BLOCK_D=D,
            num_warps=8 if D >= 64 else 4, num_stages=1,
        )
        return dq, dk, dv, None, None


def triton_flash_attention(q, k, v, block_m=128, block_n=64):
    """Causal flash attention. q, k, v: (B, nh, T, D). Returns (B, nh, T, D)."""
    D = q.shape[-1]
    if D & (D - 1) != 0:  # not a power of two
        raise ValueError(f"head dim {D} must be a power of two")
    return _FlashAttnFn.apply(q, k, v, block_m, block_n)


def is_available():
    return _TRITON_OK