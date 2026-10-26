"""Model-side tests for COUNT: does Eq. 1 reach the forward pass, and only the forward pass?

Run:  venv/bin/python tools/test_count_model.py [--table count_tables/fw5m_lo]

These are the four things a reader of model.py should not have to take on faith:

1. A zero-score count batch is *inert*: the forward pass is bitwise identical to the same forward
   pass with count=None. If that fails, f_t is leaking a constant into the residual stream and
   every COUNT number in the repo is contaminated.
2. A real lookup from a real table changes the logits, and the change is attributable to Eq. 1
   alone (same weights, same batch, only the candidate/score tensors differ).
3. Gradients reach the unembedding. Eq. 1 uses B = the model's own lm_head rows, so f_t is not a
   frozen feature: it trains the input embedding too (nanoGPT ties wte to lm_head, so this is one
   tensor). A COUNT implementation that detached B would look identical at step 0 and diverge
   later, which is the worst kind of bug to find by loss curve.
4. J and K are enforced. forward() must refuse a count batch whose J/K disagree with the model's,
   because the shape difference is silently a different network.

Plus the inference guards and the decode-side arm: generate() without attached tables raises,
attaching tables makes generate() work, and GPT.set_fusion (Appendix B.1) is a no-op at weight 0, a
different distribution at lambda=0.1 with f_t withdrawn, and reads the same longest-history row a
trained J=6 model reads. And the Sec. 5.2 variant is honest: max_order caps J at the number of orders
the lookup may read, cuts the long end of the ladder off, and matches the trained lookup order for
order.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from count.data import open_tables, table_weights
from count.generate import count_batch
from model import GPT, GPTConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', default='count_tables/fw5m_lo')
    ap.add_argument('--tokens', default='data/fineweb-edu/val.bin')
    ap.add_argument('--rows', type=int, default=2)
    ap.add_argument('--seq-len', type=int, default=64)
    args = ap.parse_args()

    torch.manual_seed(0)
    table = open_tables([args.table])[0]
    vocab = min(table.vocab_size, 50304)
    cfg = dict(n_layer=2, n_head=2, n_embd=64, block_size=args.seq_len, vocab_size=vocab,
               bias=False, dropout=0.0, chunked_head_rows=0)

    import numpy as np

    data = np.memmap(args.tokens, dtype=np.uint16, mode='r')
    idx = torch.from_numpy(np.asarray(data[:args.rows * args.seq_len], dtype=np.int64).reshape(
        args.rows, args.seq_len))
    targets = torch.roll(idx, -1, dims=1)

    # 1. inert at zero scores -------------------------------------------------
    plain = GPT(GPTConfig(**cfg, count=False))
    withcount = GPT(GPTConfig(**cfg, count=True))
    withcount.load_state_dict(plain.state_dict())  # identical weights, different config
    zeros = {"candidates": torch.full((args.rows, args.seq_len, 6, 8), 0, dtype=torch.long),
             "scores": torch.zeros(args.rows, args.seq_len, 6, 8, dtype=torch.bfloat16),
             "n_found": torch.ones(args.rows, args.seq_len, dtype=torch.long)}
    with torch.no_grad():
        logits_a, _ = plain(idx)
        logits_b, _ = withcount(idx, count=zeros)
    assert torch.equal(logits_a, logits_b), "a zero-score count batch is not inert"
    print(f"1. zero-score batch inert: logits bitwise equal ({logits_a.shape})")

    # 1b. a stock model refuses a count batch it was not built for -------------
    try:
        GPT(GPTConfig(**cfg, count=True))(idx)
        raise AssertionError("forward() accepted count=None on a count=True model")
    except RuntimeError as e:
        print(f"1b. count=True without a count batch raises: {e}")

    # 2. a real lookup changes the logits --------------------------------------
    c = count_batch([table], [1.0], idx, j=6, k=8)
    assert c["candidates"].shape == (args.rows, args.seq_len, 6, 8)
    with torch.no_grad():
        logits_c, loss_c = withcount(idx, targets, count=c)
    delta = (logits_c - logits_a).abs()
    hit = (c["candidates"] >= 0).sum().item()
    assert hit > 0 and delta.max().item() > 1e-3, "the table lookup did not reach the logits"
    with torch.no_grad():
        loss_plain = plain(idx, targets)[1]
    print(f"2. real lookup ({hit:,} candidates): logits move by max {delta.max():.4f}, "
          f"mean {delta.mean():.5f}; loss {loss_c.item():.4f} vs plain {loss_plain.item():.4f}")

    # 3. gradients reach the unembedding --------------------------------------
    withcount.zero_grad(set_to_none=True)
    _, loss = withcount(idx, targets, count=c)
    loss.backward()
    g = withcount.lm_head.weight.grad
    assert g is not None and g.abs().sum().item() > 0, "no gradient reached the unembedding"
    # Compare against the gradient the head alone would get: the count path must add to it, not
    # replace it, and it must land on the same tensor Eq. 1 reads from.
    plain.zero_grad(set_to_none=True)
    plain(idx, targets)[1].backward()
    g_plain = plain.lm_head.weight.grad
    extra = (g - g_plain).abs()
    assert extra.max().item() > 1e-8, "the count path contributed nothing to the unembedding gradient"
    tied = withcount.transformer.wte.weight is withcount.lm_head.weight
    print(f"3. unembedding gradient: count path adds max {extra.max():.3e} (tied wte==lm_head: "
          f"{tied}); input-embedding grad norm {g.norm().item():.4f}")

    # 3b. amplification is a real, trainable input LayerNorm -------------------
    amp = GPT(GPTConfig(**cfg, count=True, count_amplify=True))
    loaded = amp.load_state_dict(withcount.state_dict(), strict=False)
    # count_amp is the one parameter group a COUNT-free checkpoint cannot supply: it is new, and it
    # starts at LayerNorm's own init (gain 1, bias 0) so sqrt(d) LayerNorm is a pure rescale. Both
    # halves exist because Appendix B.6's LayerNorm has "learned gain and bias" even in a bias=False
    # model -- Table 2 names "embedding-amplification parameters" in the weight-decay exemption.
    assert list(loaded.missing_keys) == ['count_amp.weight', 'count_amp.bias'], loaded.missing_keys
    assert not loaded.unexpected_keys, loaded.unexpected_keys
    assert torch.all(amp.count_amp.weight == 1.0) and torch.all(amp.count_amp.bias == 0.0), \
        "count_amp does not start at LayerNorm's identity init"
    amp_out = amp.count_amp(torch.randn(4, 1, cfg['n_embd']))
    scale = amp_out.norm().item() / (float(cfg['n_embd'] ** 0.5) * 2.0)  # 4 rows: norm of each ~sqrt(d)
    assert abs(scale - 1.0) < 1e-3, f"count_amp is not sqrt(d) LayerNorm (got scale {scale})"
    _, amp_loss = amp(idx, targets, count=c)
    amp_loss.backward()
    gw, gb = amp.count_amp.weight.grad, amp.count_amp.bias.grad
    assert gw is not None and gw.abs().sum() > 0, "no gradient reached the amplification gain"
    assert gb is not None and gb.abs().sum() > 0, "no gradient reached the amplification bias"
    print(f"3b. embedding amplification: sqrt(d) scaling {scale:.4f} of d={cfg['n_embd']}, "
          f"gradients reach gain ({gw.abs().sum():.3e}) and bias ({gb.abs().sum():.3e})")

    # 4. J/K mismatch is refused ---------------------------------------------
    c_small = {"candidates": c["candidates"][:, :, :1, :4], "scores": c["scores"][:, :, :1, :4],
               "n_found": c["n_found"]}
    try:
        withcount(idx, targets, count=c_small)
        raise AssertionError("forward() accepted a J/K mismatch")
    except RuntimeError as e:
        print(f"4. J/K mismatch refused: {e}")

    # 5. generate() guards and works ------------------------------------------
    try:
        withcount.generate(idx[:, :4], 4)
        raise AssertionError("generate() ran without attached tables")
    except RuntimeError as e:
        print(f"5. generate() without tables raises: {e}")
    withcount.attach_count_tables([table], [1.0], j=6, k=8)
    with torch.no_grad():
        out = withcount.generate(idx[:, :4], 8)
    assert out.shape[1] == 12
    print(f"5b. generate() with attached tables: {out.shape[1]} tokens, last 8 ids "
          f"{out[0, 4:].tolist()}")

    # 5c. a sweep arm must not rewrite what the checkpoint was trained at -------
    # An eval run attaches the same tables at several shapes: K=32 for one Appendix B.4 arm, the
    # trained J=6 K=8 for the next. attach_count_tables rewrites config.count_j/count_k for the shape
    # it is running, so the *trained* shape has to be remembered separately -- otherwise the arm that
    # follows a sweep compares itself against the sweep and reports a mismatch that never happened.
    withcount.attach_count_tables([table], [1.0], j=1, k=32, allow_jk_override=True)
    with torch.no_grad():
        _ = withcount.generate(idx[:, :4], 2)
    try:
        withcount.attach_count_tables([table], [1.0], j=6, k=8)
    except RuntimeError as e:
        raise AssertionError(f"a K=32 sweep arm poisoned the trained-shape record: {e}")
    print("5c. J=1 K=32 sweep arm, then the trained J=6 K=8 back again: no false mismatch")

    # 5d. what the checkpoint was trained at is decided by its config, not by arm order --
    # eval_lm.py scores the 'standard' arm first, which sets model.config.count = False. If the
    # trained shape were remembered at first attach, that arm would make every later arm look like
    # COUNT bolted onto a COUNT-free checkpoint, and would drop the J/K guard along with it.
    import io
    from contextlib import redirect_stdout
    withcount.config.count = False # exactly what eval_lm.py does for the standard arm
    buf = io.StringIO()
    with redirect_stdout(buf):
        withcount.attach_count_tables([table], [1.0], j=6, k=8)
    out = buf.getvalue()
    assert "trained without it" not in out, (
        f"a standard arm in front of the COUNT arms made a COUNT checkpoint look COUNT-free: {out!r}")
    try:
        withcount.attach_count_tables([table], [1.0], j=2, k=8)
        raise AssertionError("the J/K guard disappeared after a standard arm ran first")
    except RuntimeError as e:
        assert "trained at J=6 K=8" in str(e), e
    print("5d. standard arm first, COUNT arms after: no false 'trained without COUNT', guard intact")

    # 6. untied head: Eq. 1's basis stops being the input embedding -------------
    # nanoGPT ties wte to lm_head, so a tied COUNT arm weights the *token-embedding* basis --
    # the variant Appendix B.7 / Figure 18 measures against the paper's unembedding basis and
    # finds worse. --tie_embeddings=False is the arm the paper trains, and this checks that it
    # really moves B, not just the parameter count.
    tied = GPT(GPTConfig(**cfg, count=True))
    untied = GPT(GPTConfig(**cfg, count=True, tie_embeddings=False))
    assert tied.transformer.wte.weight.data_ptr() == tied.lm_head.weight.data_ptr()
    assert untied.transformer.wte.weight.data_ptr() != untied.lm_head.weight.data_ptr(), (
        "tie_embeddings=False still shares the two matrices")
    vocab_d = untied.transformer.wte.weight.numel()
    assert untied.get_num_params() == tied.get_num_params() + 0, (
        f"backbone parameter count changed by {untied.get_num_params() - tied.get_num_params():,}: "
        "get_num_params should count the unembedding in both and the input embedding in neither")
    assert sum(p.numel() for p in untied.parameters()) == (
        sum(p.numel() for p in tied.parameters()) + vocab_d)
    C = count_batch([table], [1.0], idx, j=6, k=8)
    def f_of(model):
        with torch.no_grad():
            return model.count_embedding(C['candidates'], C['scores'], C['n_found']).clone()
    f0 = f_of(untied)
    with torch.no_grad():
        untied.lm_head.weight.data += 0.25
    f_head = f_of(untied)
    moved = (f_head - f0).abs().max().item()
    assert moved > 1e-6, "f_t is not built from the unembedding rows"
    with torch.no_grad():
        untied.transformer.wte.weight.data += 7.0
    assert torch.equal(f_of(untied), f_head), (
        "f_t read the input embedding: untying the head did not change the basis")
    logits_u, loss_u = untied(idx, targets=targets, count=C)
    logits_u2, _ = untied(idx, count=C)
    assert logits_u.shape == (args.rows, args.seq_len, untied.config.vocab_size)
    assert torch.allclose(logits_u[:, -1], logits_u2[:, -1], atol=1e-5), (
        "the last-position shortcut and the full head disagree when the head is untied")
    print(f"6. untied head: B is lm_head, not wte (perturbing lm_head moves f_t by "
          f"{moved:.3e}, perturbing wte moves it by 0); +{vocab_d:,} parameters, backbone "
          f"count unchanged at {untied.get_num_params()/1e6:.2f}M")

    # 7. late fusion at decode (Appendix B.1's wiring in GPT.generate) ---------
    fused_model = GPT(GPTConfig(**cfg, count=True))
    fused_model.load_state_dict(plain.state_dict())
    try:
        fused_model.set_fusion('additive', 0.1)
        raise AssertionError("set_fusion accepted a mode with no tables attached")
    except RuntimeError as e:
        assert 'attach_count_tables' in str(e), str(e)
    fused_model.attach_count_tables([table], [1.0], j=6, k=8, score_dtype=torch.float32)
    try:
        fused_model.set_fusion('bogus', 0.1)
        raise AssertionError("set_fusion accepted an unknown mode")
    except ValueError as e:
        assert 'additive|multiplicative' in str(e), str(e)

    # The hook reads slot 0 of a J=1 lookup. That has to be the same row as slot 0 of the trained
    # J=6 lookup: B.1 scores the longest history, and which history that is cannot depend on how
    # many shorter histories the checkpoint was trained with.
    C6 = fused_model._count_lookup(idx)
    C1 = fused_model._count_lookup_at(idx, 1, 8)
    assert C1['candidates'].shape[2] == 1 and C1['candidates'].shape[3] == 8
    assert torch.equal(C1['candidates'][:, :, 0, :], C6['candidates'][:, :, 0, :]), (
        "the J=1 lookup's longest-history row is not the J=6 lookup's: late fusion would be scoring "
        "a different history than the one B.1 defines")

    prompt = idx[:, :16]
    def sample_ids(seed, mode, weight, keep_count):
        torch.manual_seed(seed)
        fused_model.set_fusion(mode, weight, keep_count=keep_count)
        out = fused_model.generate(prompt, 24, temperature=1.0, top_k=None)
        fused_model.set_fusion(None)
        return out

    # Weight 0 in either mode is a no-op, and keep_count=True leaves the trained f_t in place, so
    # this must reproduce plain COUNT decode token for token.
    plain_ids = sample_ids(11, None, 0.0, True)
    zero_add = sample_ids(11, 'additive', 0.0, True)
    zero_mul = sample_ids(11, 'multiplicative', 0.0, True)
    assert torch.equal(plain_ids, zero_add) and torch.equal(plain_ids, zero_mul), (
        "fusion weight 0 changed the sampled tokens: fuse_logits is not a no-op at weight 0, or the "
        "hook is applied somewhere it should not be")
    assert fused_model.config.count is True, "keep_count=True did not leave config.count alone"

    # The paper's configuration: f_t off, table prior on. Same seed, so any difference in what comes
    # out is the fusion doing its job and nothing else.
    fused_ids = sample_ids(11, 'additive', 0.1, False)
    assert fused_model.config.count is False, "set_fusion did not withdraw f_t (the paper's arm)"
    assert not torch.equal(fused_ids[0, 16:], plain_ids[0, 16:]), (
        "lambda=0.1 over a real table left 24 sampled tokens untouched")
    print("7. late fusion at decode: weight 0 reproduces plain COUNT decode exactly, lambda=0.1 with "
          "f_t withdrawn changes the samples, and the J=1 row matches the trained J=6 row")

    # rehost_ngram_tables() must be harmless on a model that has no learned tables at all.
    assert fused_model.rehost_ngram_tables() is fused_model

    # 8. Sec. 5.2's shorter-history variant (max_order) ------------------------
    # Capping the order caps the ladder: the lookup returns one candidate set per order it is allowed
    # to read, so a --max-orders arm carries J = min(J, cap, table.max_order) sets. Slots are ordered
    # longest suffix first, so a capped batch is *not* a prefix of the trained one -- the cap cuts the
    # long end off. What has to hold is that the two agree slot for slot once slots are matched by the
    # order they came from: the ablation reads fewer of the same histories, not different histories.
    from count.table import mix
    tokens = np.asarray(idx.numpy(), dtype=np.uint16)
    full6 = mix([table], [1.0], tokens, j=6, k=8)
    cap_max = min(6, table.max_order)
    checked = 0
    for cap in (1, 2, 3, 5, 6, 7):
        capped = mix([table], [1.0], tokens, j=6, k=8, max_order=cap)
        jj = capped.orders.shape[2]
        assert jj == min(6, cap, table.max_order), (
            f"max_order={cap} produced J={jj}; the lookup is not returning one candidate set per "
            "order it reads")
        assert int(capped.orders.max()) <= min(cap, table.max_order), (
            f"max_order={cap} still resolved a slot of order {int(capped.orders.max())}")
        for r in range(args.rows):
            for p in range(1, args.seq_len):
                full_orders = [int(o) for o in full6.orders[r, p]]
                cap_orders = [int(o) for o in capped.orders[r, p]]
                if any(o == 0 for o in cap_orders):
                    continue  # a slot the cap leaves unresolved: nothing to line up against
                assert set(cap_orders) <= set(full_orders), (
                    f"max_order={cap} resolved order {sorted(cap_orders)} at (row {r}, pos {p}) but "
                    f"the trained lookup resolved {sorted(full_orders)}: the capped batch is not built "
                    "from the histories the trained batch reads")
                for s, o in enumerate(cap_orders):
                    fs = full_orders.index(o)
                    assert np.array_equal(capped.candidates[r, p, s], full6.candidates[r, p, fs]) and \
                        np.allclose(capped.scores[r, p, s], full6.scores[r, p, fs], atol=1e-6), (
                        f"the order-{o} slot differs between max_order={cap} and the trained lookup")
                    checked += 1
    print(f"8. max_order caps J at min(J, cap, table.max_order)={cap_max} and cuts the long end off: "
          f"{checked} capped slots matched the trained lookup's slot for the same order")

    table.close()
    print("\nCOUNT MODEL TESTS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
