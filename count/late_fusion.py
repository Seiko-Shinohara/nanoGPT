"""COUNT's tables used at the *output* instead of the input: Appendix B.1, Equations 5-7.

COUNT is early fusion -- the n-gram statistics enter as f_t on the input embedding. The paper's
comparison arm is the older, output-side family (kNN-LM, Infini-gram, Li et al. 2022): take the
transformer's own next-token distribution p_t(w) = p_theta(w | x_1:t) and blend it with the table's
S(w | h_{t,1}). Two variants, both with the same tables and the same K:

    additive (Eq. 5)        p_add(w)  = (1 - lambda m_t) p(w) + lambda q_t(w)
    multiplicative (Eq. 7)  p_mult(w) proportional to p(w) * qtilde_t(w)^alpha

with q_t and qtilde_t built from the candidates W_t = T_K[h_{t,1}] and their retained mass
A_t = sum_{w in W_t} S(w | h_{t,1]). Two differences from COUNT that the paper insists on and that
this module enforces by construction:

* J = 1. Late fusion uses only the longest matching history h_{t,1}, "as the neural LM cannot
  correct noisy low order predictions in late fusion". So the caller looks up with j=1 and hands
  this module the single order's K candidates -- averaging six orders into an output-side prior is
  not what is being compared.
* mass, not scores. Additive fusion is a probability interpolation, so it is sensitive to A_t: the
  cap `max{1, A_t}` is what keeps a position whose backoff chain raised more than 1.0 of score from
  reweighting p_t by more than lambda, and m_t = min{A_t, 1} scales the whole blend by how much
  usable mass the row actually has. A position that resolves nothing (A_t = 0) is left completely
  alone by both variants: m_t = 0 in Eq. 5, and Eq. 6 degenerates to a uniform 1/|V|, whose
  normalized product with p is p itself.

Paper settings for the inference-only comparison: lambda = 0.1 or alpha = 0.1, K = 8, a standard
transformer checkpoint without embedding amplification.

Everything here is fp32 and dense per row chunk: a (rows, |V|) probability table is the shape these
equations are written over, and at 1024 rows x 50304 vocab it is 206 MB -- the chunking is done by
the caller (model.logits_rows).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


EPS_MASS = 1e-4  # Sec. B.1: multiplicative fusion keeps at least 1e-4 total mass for non-candidates


def retained_mass(candidates: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """A_t = sum over the stored candidates of S(w | h). Padded slots carry score 0 exactly."""
    return scores.masked_fill(candidates < 0, 0.0).sum(dim=-1)


def candidate_union_size(candidates: torch.Tensor) -> torch.Tensor:
    """|W_t|: how many real candidates the row has (padded slots are -1)."""
    return (candidates >= 0).sum(dim=-1)


def additive_prior(
    candidates: torch.Tensor,
    scores: torch.Tensor,
    vocab: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eq. 5's q_t(w) = S(w | h_{t,1}) / max{1, A_t} as a dense (rows, vocab) table, plus m_t.

    Returns the *probability* table q and the per-row scalar m_t = min{A_t, 1}. The blend itself,

        p_add(w) = (1 - lambda m_t) p(w) + lambda q(w),

    is exactly normalized: sum_w q(w) = A_t / max{1, A_t} = m_t, so the two components carry
    (1 - lambda m_t) and lambda m_t of the mass. No renormalization is applied afterwards, because
    Eq. 5 does not ask for one.
    """
    rows = candidates.size(0)
    device = candidates.device
    q = torch.zeros(rows, vocab, device=device, dtype=torch.float32)
    a = retained_mass(candidates, scores)
    denom = torch.clamp(a, min=1.0)  # max{1, A_t}
    valid = (candidates >= 0) & (scores > 0)  # same convention as multiplicative_prior
    safe_idx = candidates.clamp(min=0)
    probs = torch.where(valid, scores, torch.zeros_like(scores)) / denom.unsqueeze(1)
    q.scatter_(1, safe_idx, probs)
    m = torch.clamp(a, max=1.0)
    return q, m


def multiplicative_prior(
    candidates: torch.Tensor,
    scores: torch.Tensor,
    vocab: int,
    eps: float = EPS_MASS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eq. 6's qtilde_t(w) as a dense log-probability table, plus the mass kept for candidates.

    Candidates get S(w) / max{1, A_t / (1 - eps)}; everything else gets (1 - m_e_t) / (|V| - |W_t|)
    with m_e_t = min{A_t, 1 - eps}, i.e. at least `eps` of total mass spread uniformly over the
    non-candidates. Log form, because Eq. 7 raises it to a power and multiplies it into p.
    """
    rows = candidates.size(0)
    device = candidates.device
    a = retained_mass(candidates, scores)
    m_e = torch.clamp(a, max=1.0 - eps)  # m_e_t = min{A_t, 1 - eps}, the mass kept for candidates
    denom = torch.clamp(a / (1.0 - eps), min=1.0)  # max{1, A_t / (1 - eps)}
    # A slot is a candidate for Eq. 6 only if it stores a token *and* carries mass. Padded slots are
    # -1 with score 0, and must not be scattered as -inf: clamping the index to 0 would otherwise
    # zero out vocabulary id 0's fused probability at every short row. Same convention as
    # eval_count.py's `valid = (cand >= 0) & (sc > 0)`.
    valid = (candidates >= 0) & (scores > 0)
    n_cand = valid.sum(dim=-1)
    leftover = (1.0 - m_e) / (vocab - n_cand).clamp(min=1)
    log_leftover = torch.log(leftover)

    logq = log_leftover.unsqueeze(1).expand(rows, vocab).contiguous()
    cand_prob = torch.where(valid, scores, torch.zeros_like(scores)) / denom.unsqueeze(1)
    log_cand = torch.where(cand_prob > 0, torch.log(cand_prob.clamp(min=1e-38)),
                           log_leftover.unsqueeze(1).expand(rows, -1))
    logq.scatter_(1, candidates.clamp(min=0), log_cand)
    return logq, m_e


def fuse_additive(
    log_probs: torch.Tensor,
    candidates: torch.Tensor,
    scores: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Eq. 5 applied to a (rows, vocab) log-probability table. Returns log p_add, normalized."""
    q, m = additive_prior(candidates, scores, log_probs.size(1))
    p = log_probs.exp()
    # (1 - lambda m_t) multiplies the model's own distribution row-wise; lambda q_t is the table's
    # share. Together they sum to 1 exactly, so no renormalization follows.
    p_add = (1.0 - lam * m).unsqueeze(1) * p + lam * q
    return torch.log(p_add.clamp(min=1e-38))


def fuse_multiplicative(
    log_probs: torch.Tensor,
    candidates: torch.Tensor,
    scores: torch.Tensor,
    alpha: float,
    eps: float = EPS_MASS,
) -> torch.Tensor:
    """Eq. 7 applied to a (rows, vocab) log-probability table. Returns the normalized log p_mult."""
    logq, _ = multiplicative_prior(candidates, scores, log_probs.size(1), eps)
    fused = log_probs + alpha * logq
    return fused - torch.logsumexp(fused, dim=-1, keepdim=True)


def fuse_logits(
    log_probs: torch.Tensor,
    candidates: torch.Tensor,
    scores: torch.Tensor,
    mode: str,
    weight: float,
    eps: float = EPS_MASS,
) -> torch.Tensor:
    """One entry point for both variants, so eval_lm.py and sample.py hold a single arm switch.

    mode='additive' uses lambda=weight (Eq. 5), mode='multiplicative' uses alpha=weight (Eq. 7),
    mode='none' returns log_probs untouched -- the baseline every late-fusion delta is measured
    against. weight=0 is a no-op in both modes, which tools/test_late_fusion.py checks.
    """
    if mode in ("none", "", None):
        return log_probs
    if mode == "additive":
        return fuse_additive(log_probs, candidates, scores, weight)
    if mode == "multiplicative":
        return fuse_multiplicative(log_probs, candidates, scores, weight, eps)
    raise ValueError(f"unknown fusion mode {mode!r}: expected none|additive|multiplicative")


def scores_for_targets(
    candidates: torch.Tensor,
    scores: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """S(w_t | h_{t,1}) for the corpus's own next token, 0 where it is not a stored candidate.

    Not needed to fuse -- it is the diagnostic that says *why* a fusion arm moved the loss: the
    table either proposed the true token with some mass or it did not, and only the former can help.
    """
    # The scores are indexed by *candidate slot*, not by token id, so the true token has to be
    # located among the row's K slots rather than used as a gather index.
    match = (candidates >= 0) & (candidates == targets.unsqueeze(1))
    return (scores * match).sum(dim=-1)
